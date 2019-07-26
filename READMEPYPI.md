`hazma` is a tool for analyzing theories of sub-GeV dark matter. It can compute gamma-ray spectra from dark matter 
annihilations, set limits using current gamma-ray data and make projects for future gamma-ray detectors. It can generate 
positron spectra as well, and derive accurate CMB constraints. `hazma` includes several pre-implemented sub-GeV dark 
matter models, and provides the infrastructure to add custom ones. Visit https://github.com/LoganAMorrison/Hazma or 
https://hazma.readthedocs.io for more information.

`hazma` can be installed from PyPI using:

    pip install hazma

Alternatively, you can download `hazma` directly from this page, navigate to the package directory using the command line and run

    pip install .

or

    python setup.py install

Since `hazma` utilizes C to rapidly compute gamma ray, electron and positron spectra, you will need to have the `cython` package installed.