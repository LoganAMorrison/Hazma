![Logo](docs/source/_images/hazma_logo.png)


[![CircleCI](https://circleci.com/gh/LoganAMorrison/Hazma.svg?style=svg)](https://circleci.com/gh/LoganAMorrison/Hazma)

[![Build Status](https://travis-ci.org/LoganAMorrison/Hazma.svg?branch=master)](https://travis-ci.org/LoganAMorrison/Hazma)

# Hazma

`hazma` is a tool for analyzing theories of sub-GeV dark matter. It can compute gamma-ray spectra from dark matter annihilations, set limits using current gamma-ray data and make projects for future gamma-ray detectors. It can generate positron spectra as well, and derive accurate CMB constraints. `hazma` includes several pre-implemented sub-GeV dark matter models, and provides the infrastructure to add custom ones.

## Installation

`hazma` is currently still in development. If you would like to try it anyways, you can install it using the command:

    pip install --index-url https://test.pypi.org/simple/ hazma

Alternatively, you can download `hazma` directly from this page, navigate to the package directory using the command line and run

    pip install .

or

    python setup.py install

Since `hazma` utilizes C to rapidly compute gamma ray, electron and positron spectra, you will need to have the `cython` package installed.

Another way to run `hazma` is by using `docker`. If you have docker installed on your machine, clone the `hazma` repository and in the `hazma` directory, run:

    docker build --rm -t jupyter/hazma .

This will build the docker image called `jupyter/hazma`. Then to start a jupyter notebook, run:
    
    docker run -it -p 8888:8888 -v /path/to/hazma/tutorials:/home/jovyan/work --rm --name jupyter jupyter/hazma

This will start a jupyter kernel. 

## Other information

Logo design: David Reiman and Adam Coogan; icon from Freepik from [flaticon.com](flaticon.com).
