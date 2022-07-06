# Hazma

![Logo](docs/source/_static/img/hazma_logo_large.png)

---

| [**Overview**](#overview)
| [**Installation**](#installation)
| [**Documentation**](https://hazma.readthedocs.io/en/latest/)
| [**Usage**](#usage)
| [**Citing Hazma**](#citing)

[![CircleCI](https://circleci.com/gh/LoganAMorrison/Hazma.svg?style=svg)](https://circleci.com/gh/LoganAMorrison/Hazma)
[![Documentation Status](https://readthedocs.org/projects/hazma/badge/?version=latest)](https://hazma.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3347114.svg)](https://doi.org/10.5281/zenodo.3347114)
[![arXiv](https://img.shields.io/badge/arXiv-1907.11846-b31b1b.svg?style=plastic)](https://arxiv.org/abs/1907.11846)

## Overview<a id="overview"></a>

Hazma is a tool for studying indirect detection of sub-GeV dark. Its main uses are:

- Computing gamma-ray and electron/positron spectra from dark matter annihilations;
- Setting limits on sub-GeV dark matter using existing gamma-ray data;
- Determining the discovery reach of future gamma-ray detectors;
- Deriving accurate CMB constraints.

Hazma comes with [several](https://hazma.readthedocs.io/en/latest/models.html) sub-GeV dark matter models, for which it provides functions to compute dark matter annihilation cross sections and mediator decay widths. A variety of low-level tools are provided to make it straightforward to [define new models](https://hazma.readthedocs.io/en/latest/usage.html#user-defined-models).

## 📦 Installation<a id="installation"></a>

Hazma can be installed from PyPI using:

```shell
pip install hazma
```

Alternatively, you can download Hazma directly from this page, navigate to the package directory using the command line and run

```shell
pip install .
```

or

```shell
python setup.py install
```

Since Hazma utilizes C to rapidly compute gamma ray, electron and positron spectra, you will need to have [Cython](https://github.com/cython/cython)
and a c/c++ compiler installed.

## 🚀 Usage<a id="usage"></a>

### Computing Photon, Positron and Neutrino Spectra

`hazma` has built in utilities for generating photon, positron and neutrino
spectra. All spectra generation functions live in `hazma.spectra`. The easiest to use and
most versatile functions for spectrum generation are: `hazma.spectra.dnde_photon`,
`hazma.spectra.dnde_positron` and `hazma.spectra.dnde_neutrino`. As an example, to compute the photon spectrum from 5 neutral pions, use:

```python
import numpy as np
from hazma import spectra
from hazma.parameters import neutral_pion_mass as mpi0

# Spectra from 5 neutral pions
cme = 6 * mpi0 # center-of-mass energy
photon_energies = np.geomspace(1e-3, 1.0) * cme
final_states = ["pi0"] * 5 # 5 neutral pions
dnde = spectra.dnde_photon(
 photon_energies=photon_energies,
 cme=cme,
 final_states=final_states
)
```

By replacing `dnde_photon` with `dnde_positron` or `dnde_neutrino`, you can
compute the positron or neutrino spectra.

You can supply a squared matrix element to improve the accuracy when there are
more than 2 final state particles. For example, suppose you want to compute the
muon decay spectrum into photons (`hazma` has built-in functions for computing
the analytic result, so this is simply for demonstration.) By default, for a
three-body final state, the matrix element is assumed to accept two invariant
masses: `s = (p2 + p3)^2` and `t=(p1+p3)^2`. The user can change this assumption
by using `msqrd_signature`. To compute the muon decay spectrum, use:

```python
import numpy as np
from hazma import spectra
from hazma import parameters

mmu = parameters.muon_mass
me = parameters.electron_mass

# Squared matrix element for mu -> e + nu + nu
# s = (p2 + p3)^2, t = (p1 + p3)^2
# with p1 = pe, p2 = pve, p3 = pvm (same order as `final_states`)
def msqrd(s, t):
    return 16.0 * GF ** 2 * (mmu**2 - t) * (t - me**2)

# Spectra from mu -> e + nu_e + nu_mu
cme = mmu
photon_energies = np.geomspace(1e-3, 1.0) * cme
final_states = ["e", "ve", "vm"]

dnde = spectra.dnde_photon(
    photon_energies=photon_energies,
    cme=cme,
    final_states=final_states,
    msqrd=msqrd,
)
```

The `dnde_photon` (and sibling functions) can also be used to compute spectra
from a single final state. For example, `dnde_positron(positron_energies, cme,
"phi")` will compute the positron spectrum from a `phi` vector meson.

### Working with Lorentz Invariant Phase Space

`hazma` include several functions to integrate over Lorentz invariant phase
space. Notably, `hazma` can integrate over N-body phase space using the `RAMBO`
algorithm. There is also special code for computing three-body phase space
integrals.

To demonstrate, let's consider a silly squared matrix element of a 5 body final
state. We take the squared matrix element to be the product of pairs of final
state momenta.

Before we do so, we need to mention how `hazma` treats four-momenta. For-momenta
are taken to be NumPy arrays with the first-axis containing the energy,
x-momentum, y-momenta and z-momenta. The second axis contains the different
particles. The last axis contains the number of groups of four-momenta we have
(number of 'events'.) For example, if we have 5 particles and 100 events, then
the momenta will be stored in a NumPy are with shape `momenta.shape == (4, 5,100)`. 

- To access all the four-momenta of particle 3, you would use `momenta[:,2]`. 
- To access the energies of all the particles over all events, you would use
`momenta[0]`. 
- To access all the four-momenta of the first event, you would use
`momenta[:,:,0]` or `momenta[...,0]`.

With that out of the way, our squared matrix element will be:

```python
import numpy as np
from hazma.utils import ldot # computes Minkowski dot product of numpy arrays
import itertools

# compute all combinations of two particles
pairs = np.array(list(itertools.combinations(range(5), 2)))

# The below numpy trickery is equivalent to:
# npts = momenta.shape[-1]
# msqrd = np.ones((npts,))
# for i in range(5):
#   for j in range(i+1, 5):
#       msqrd *= ldot(momenta[:, i], momenta[:, j])
# return msqrd
def msqrd(momenta):
    p1s = momenta[:, pairs.T[0], :]
    p2s = momenta[:, pairs.T[1], :]
    return np.prod(ldot(p1s, p2s), axis=0)
```

To integrate over phase space, we create a `Rambo` object from
`hazma.phase_space`. We then call `integrate`, specifying how many Monte-Carlo
points should be used to compute the integral:

```python
from hazma import phase_space
cme = 20.0 # Center-of-mass energy
masses = [1.0, 2.0, 3.0, 4.0, 5.0] # Masses of the final state particles
rambo = phase_space.Rambo(cme=cme, masses=masses, msqrd=msqrd)
# Integrate! We use 2^14 points, this take ~20ms
rambo.integrate(n=1<<14)
```

### Vector Form Factors

In version 2.0, we introduced vector form factors for a large set of mesonic
final states. These form factors are available in `hazma.form_factors.vector`.
All the form factors have a similar interface and similar functionality. We
provide functions to compute:

- the raw form-factor (scalar function coefficients of the Lorentz structures), 
- integrals of the form factors over phase space, 
- decay widths of a massive vector or cross section of dark matter annihilation,
- energy distributions and invariant mass distributions of the final state mesons

Examples:

```python
import hazma.form_factors.vector as vff

# compute pi-pi electromagnetic form-factor between 300 MeV and 1 GeV
ff_pipi = vff.VectorFormFactorPiPi()
energies = np.linspace(300.0, 1000.0, 100)
ff_pipi.form_factor(q=energies, gvuu=2.0/3.0, gvdd=-1.0/3.0)

# Integrate the pi-pi-pi0 form-factor over phase-space between 450 MeV and 1 GeV
ff_pipipi0 = vff.VectorFormFactorPiPiPi0()
energies = np.linspace(450.0, 1000.0, 100)
ff_pipipi0.integrated_form_factor(q=energies, gvuu=2.0/3.0, gvdd=-1.0/3.0, gvss=-1.0/3.0)

# Generate energy distributions of the pi0-k-k form factor at 1 GeV
ff_pikk = vff.VectorFormFactorPi0KpKm()
ff_pikk.energy_distributions(q=1000.0, gvuu=2.0/3.0, gvdd=-1.0/3.0, gvss=-1.0/3.0, nbins=100)
```

## Other information

### Citing<a id="citing"></a>

If you use Hazma in your own research, please cite [our paper](https://arxiv.org/abs/1907.11846):

```bibtex
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

If you use any of the models we've included that rely on chiral perturbation
theory, please also cite [the paper](https://arxiv.org/abs/2104.06168)
explaining how they were constructed:

```bibtex
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

Logo design: David Reiman and Adam Coogan; icon from Freepik from
[flaticon.com](https://www.flaticon.com/).
