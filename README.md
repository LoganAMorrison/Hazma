![Logo](docs/source/_images/hazma_logo.png)
Icon made by Freepik from www.flaticon.com (Logo designed by David Reiman and Adam Coogan)

[![CircleCI](https://circleci.com/gh/LoganAMorrison/Hazma.svg?style=svg)](https://circleci.com/gh/LoganAMorrison/Hazma)

[![Build Status](https://travis-ci.org/LoganAMorrison/Hazma.svg?branch=master)](https://travis-ci.org/LoganAMorrison/Hazma)

# Hazma
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
