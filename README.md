![Logo](docs/source/_images/hazma_logo.png)
Icon made by Freepik from www.flaticon.com (Logo designed by David Reiman and Adam Coogan)

[![CircleCI](https://circleci.com/gh/LoganAMorrison/Hazma.svg?style=svg)](https://circleci.com/gh/LoganAMorrison/Hazma)

[![Build Status](https://travis-ci.org/LoganAMorrison/Hazma.svg?branch=master)](https://travis-ci.org/LoganAMorrison/Hazma)

# Hazma - Developement Branch
Gamma ray spectrum generator for light standard model particles.

For more information, visit https://loganamorrison.github.io/Hazma/

## Installation

`hazma` is currently still in development. If you would like to try it anyways, you can install it using the command:

    pip install --index-url https://test.pypi.org/simple/ hazma

Alternatively, you can download `hazma` directly from github, navigate to the package directory using the command line and run

    pip install .

or

    python setup.py install

Since `hazma` utilizes C to rapidly compute gamma ray and electron/positron spectra, you will need to have the cython package installed.

Another way to run `hazma` is by using **docker**. If you have docker installed on your machine, clone the `hazma` repository and in the `hazma` directory, run:

    docker build --rm -t jupyter/hazma .

This will build the docker image and will call it *jupyter/hazma*. Then to start a jupyter notebook, run:
    
    docker run -it -p 8888:8888 -v /path/to/hazma/tutorials:/home/jovyan/work --rm --name jupyter jupyter/hazma

This will start a jupyter kernal. 


