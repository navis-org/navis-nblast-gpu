# NBLAST on GPU(s) [WIP]

In a nutshell, the main bottleneck when running NBLAST is the nearest-neighbor
search. Vanilla NBLAST in `navis` is already using fast compiled KDTree libraries
(`scipy` or `pykdtree`) for this but depending on your setup you'll still spend
up to 80% of the time in the nearest-neighbour search.

This is not so much a problem if you're only ever working with a few hundred
neurons at a time. Existing connectomics datasets, however, already contain
tens of thousands of neurons and we will see the 100k mark being breached within
the next year.

You can throw more cores at it and that obviously helps speed things up. But
wait a minute... don't machine learning folks use dem fancy graphic cards?
Would be nice if we could leverage the power of a GPU for NBLAST!

Turns out we can! [PyTorch-cluster](https://github.com/rusty1s/pytorch_cluster)
provides an easy-to-use KNN implementation that can be run on a GPU.

## Advantages
- fast: hard to compare but 1 high end GPU ~= 20 CPU cores
- runs smoothly on Google Colab GPUs

## Disadvantages
- requires a CUDA-compatible (i.e. NVIDIA) GPU
- tricky to setup on OSX (see e.g. [this](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) article)
- limited by the GPU's memory   

## Install

First install `nblast-gpu` by either cloning and install with `pip3 install -e .`
or directly from Github:

```bash
$ pip3 install git+https://github.com/navis-org/navis-nblast-gpu@main
```

Then you need to install [PyTorch-cluster](https://github.com/rusty1s/pytorch_cluster).
Please see instructions on their website!

## Usage

```Python
>>> import navis
>>> from nblast_gpu import nblast
>>> nl = navis.example_neurons(5)
>>> dp = navis.make_dotprops(nl, k=5)
>>> dp_um = dp / 125
>>> scores = nblast(dp_um, dp_um)
```

## TODO
- scale to and distribute NBLAST onto multiple GPUs
- check GPU memory limits (utility function?)
- look into batching for very large datasets which exceed GPU memory
- work towards feature-parity with vanilla NBLAST
