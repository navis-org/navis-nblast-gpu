#    This script is part of navis (http://www.github.com/navis-org/navis-nblast-gpu).
#    Copyright (C) 2022 Philipp Schlegel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

import torch
import navis

import numpy as np
import pandas as pd

from torch_cluster import knn
from tqdm.auto import tqdm

__all__ = ['nblast']


def nblast(queries: navis.NeuronList,
           targets: navis.NeuronList,
           device: str = None,
           progress: bool = True):
    """Run NBLAST on GPU using PyTorch KNN.

    The general strategy is this:

    1. Generate one large tensor for all query points.
    2. Iterate over each target and find the nearest neighbours for all
       query points at once.
    3. Do as much of the scoring, etc. as possible on the GPU before
       moving back to CPU.

    Important: because the for loop iterates over the targets, this
    implementation is fastest with many queries and few targets.

    Parameters
    ----------
    query :     navis.NeuronLists
                Dotprops to NBLAST.
    targets :   navis.NeuronLists
                Dotprops to NBLAST against.
    device :    str, optional
                Which device to use for tensors. If ``None`` will use GPU if
                available and fallback to CPU if not.
    progress :  bool
                Whether to show a progress bar.

    Returns
    -------
    scores :      pandas.DataFrame

    """
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not isinstance(queries, navis.NeuronList):
        queries = navis.NeuronList(queries)

    if not isinstance(targets, navis.NeuronList):
        targets = navis.NeuronList(targets)

    nblast_preflight(queries, targets, n_cores=1,
                     req_unique_ids=True, req_dotprops=True,
                     req_microns=True)

    # Need to normalize coordinates - otherwise we will run into issues where
    # the nearest neighbour search on the GPU returns no NN for some points
    co_norm = max(np.abs(queries.bbox).max(), np.abs(targets.bbox).max())

    # Generate tensor for all query points (Q_p)
    Q_p = torch.tensor((np.vstack(queries.points) / co_norm).astype('single', copy=False),
                       device=device)
    # This tells us where to split query points to get arrays for individual
    # neurons later on
    Q_splits = queries.n_points.cumsum()[:-1]

    # Generate array for all query vectors (Q_v)
    Q_v = torch.tensor(np.vstack(queries.vect), device=device)

    # Grab the normal FWCB score function
    fcwb_score_fn = navis.nbl.smat.smat_fcwb(False)
    score_fn = ScoreFunc2D(fcwb_score_fn, device=device)

    # Prepare scores dataframe
    scores = pd.DataFrame(np.empty((len(queries), len(targets)), dtype='float32'),
                          index=queries.id, columns=targets.id)

    # Calculate self-hits for normalization
    self_hits = []
    for n in queries:
        self_hits.append(len(n.points) * score_fn(0, 1.0))
        self_hits = np.array(self_hits)

    for t in tqdm(targets, desc='NBLASTing', disable=not progress):
        # Generate tensors for this target's points (T_p) and vectors (T_v)
        T_p = torch.tensor((t.points / co_norm).astype('single', copy=False),
                           device=device)
        T_v = torch.tensor(t.vect, device=device)

        # Get nearest neighbour
        ix, nn = knn(T_p, Q_p, k=1)

        # Calculate distances (note we're de-normalising here)
        dist = (torch.sqrt(torch.sum((Q_p - T_p[nn]) ** 2, 1)) * co_norm)

        # Calculate vector dotproducts
        dp = torch.abs(torch.sum(Q_v * T_v[nn], 1))

        # Delete tensor from GPU
        del T_p, T_v

        # Calculate scores from distances and dotproducts
        point_scr = score_fn(dist, dp).cpu()

        # Agglomerate over each neuron
        point_scr_grp = np.array([s.sum() for s in np.split(point_scr, Q_splits)])

        # Normalize
        point_scr_grp /= self_hits

        # Write to scores
        scores[t.id] = point_scr_grp

    # Delete tensor from GPU
    del Q_p, Q_v

    return scores


class ScoreFunc2D:
    """2D score function running on tensors.

    Parameters
    ----------
    score_fn :  navis.Lookup2d
                Score function defining bins and cells.
    device :    str

    """
    def __init__(self, score_fn, device):
        self.score_fn = score_fn

        # Turn cells and ax boundaries into tensors
        self.cells = torch.tensor(score_fn.cells, device=device)
        self.ax1_bounds = torch.tensor(score_fn.axes[0].boundaries, device=device)
        self.ax2_bounds = torch.tensor(score_fn.axes[1].boundaries, device=device)

    def __call__(self, v1, v2):
        # Use bucketsize to digitize
        ax1_ix = torch.bucketize(v1, self.ax1_bounds, right=True) - 1
        ax2_ix = torch.bucketize(v2, self.ax2_bounds, right=True) - 1

        # Return values
        return self.cells[ax1_ix, ax2_ix]
