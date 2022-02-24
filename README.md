![Logo](docs/source/_static/img/hazma_logo_large.png)

-----------------------------------------

[![CircleCI](https://circleci.com/gh/LoganAMorrison/Hazma.svg?style=svg)](https://circleci.com/gh/LoganAMorrison/Hazma)
[![Documentation Status](https://readthedocs.org/projects/hazma/badge/?version=latest)](https://hazma.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3347114.svg)](https://doi.org/10.5281/zenodo.3347114)
[![arXiv](https://img.shields.io/badge/arXiv-1907.11846-b31b1b.svg?style=plastic)](https://arxiv.org/abs/1907.11846)

Hazma is a tool for studying indirect detection of sub-GeV dark. Its main uses are:

- Computing gamma-ray and electron/positron spectra from dark matter annihilations;
- Setting limits on sub-GeV dark matter using existing gamma-ray data;
- Determining the discovery reach of future gamma-ray detectors;
- Deriving accurate CMB constraints.

Hazma comes with [several](https://hazma.readthedocs.io/en/latest/models.html) sub-GeV dark matter models, for which it provides functions to compute dark matter annihilation cross sections and mediator decay widths. A variety of low-level tools are provided to make it straightforward to [define new models](https://hazma.readthedocs.io/en/latest/usage.html#user-defined-models).

## Installation

Hazma can be installed from PyPI using:

    pip install hazma

Alternatively, you can download Hazma directly from this page, navigate to the package directory using the command line and run

    pip install .

or

    python setup.py install

Since Hazma utilizes C to rapidly compute gamma ray, electron and positron spectra, you will need to have the cython package installed.

Another way to run Hazma is by using docker. If you have docker installed on your machine, clone the Hazma repository and in the Hazma directory, run:

    docker build --rm -t jupyter/hazma .

This will build the docker image called `jupyter/hazma`. Then to start a jupyter notebook, run:

    docker run -it -p 8888:8888 -v /path/to/hazma/tutorials:/home/jovyan/work --rm --name jupyter jupyter/hazma

This will start a jupyter kernel.

## Other information

### Citing

If you use Hazma in your own research, please cite [our paper](https://arxiv.org/abs/1907.11846):
```
@article{Coogan:2019qpu,
      author         = "Coogan, Adam and Morrison, Logan and Profumo, Stefano",
      title          = "{Hazma: A Python Toolkit for Studying Indirect Detection
                        of Sub-GeV Dark Matter}",
      year           = "2019",
      eprint         = "1907.11846",
      archivePrefix  = "arXiv",
      primaryClass   = "hep-ph"
}
```
If you use any of the models we've included that rely on chiral perturbation theory, please also cite [the paper](https://arxiv.org/abs/2104.06168) explaining how they were constructed:
```
@article{Coogan:2021sjs,
    author = "Coogan, Adam and Morrison, Logan and Profumo, Stefano",
    title = "{Precision Gamma-Ray Constraints for Sub-GeV Dark Matter Models}",
    eprint = "2104.06168",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "4",
    year = "2021"
}
```

### Papers using `hazma`

- [![arXiv](https://img.shields.io/badge/arXiv-2010.04797-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2010.04797)
- [![arXiv](https://img.shields.io/badge/arXiv-2101.10370-B31B1B.svg?style=plastic)](https://arxiv.org/abs/2101.10370)
- [![arXiv](https://img.shields.io/badge/arXiv-2104.06168-B31B1B.svg?style=plastic)](https://arxiv.org/abs/2104.06168)

Logo design: David Reiman and Adam Coogan; icon from Freepik from [flaticon.com](https://www.flaticon.com/).
