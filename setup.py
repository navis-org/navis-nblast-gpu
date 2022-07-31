from setuptools import setup, find_packages
from pathlib import Path
from runpy import run_path

from extreqs import parse_requirement_files

HERE = Path(__file__).resolve().parent

verstr = run_path(str(HERE / "nblast_gpu" / "__version__.py"))["__version__"]

install_requires, extras_require = parse_requirement_files(
    HERE / "requirements.txt",
)

with open("README.md") as f:
    long_description = f.read()

setup(
    name='nblast-gpu',
    version=verstr,
    packages=find_packages(include=["nblast_gpu", "nblast_gpu.*"]),
    license='GNU GPL V3',
    description='NBLAST on GPU (PyTorch)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/navis-org/navis-nblast-gpu',
    project_urls={
     "Documentation": "https://github.com/navis-org/navis-nblast-gpu",
     "Source": "https://github.com/navis-org/navis-nblast-gpu",
     "Changelog": "https://github.com/navis-org/navis-nblast-gpu",
    },
    author='Philipp Schlegel',
    author_email='pms70@cam.ac.uk',
    keywords='NBLAST GPU PyTorch neuron comparison',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=install_requires,
    extras_require=dict(extras_require)
    python_requires='>=3.7',
    zip_safe=False,

    include_package_data=True

)
