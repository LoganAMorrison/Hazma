# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.2'
#       jupytext_version: 1.1.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# **This notebook collects all the code snippets from the paper introducing Hazma.** Commented lines indicate the expected output.

# %%
import numpy as np

# %% [markdown] {"heading_collapsed": true}
# ## Section 3: Particle Physics Framework

# %% {"hidden": true}
from hazma.scalar_mediator import ScalarMediator

sm = ScalarMediator(mx=150.0, ms=1e3, gsxx=1.0, gsff=0.1, gsGG=0.1, gsFF=0.1, lam=2e5)
print(sm.gsff)
# 0.1
sm.gsff = 0.5
print(sm.gsff)
# 0.5

# %% {"hidden": true}
from hazma.scalar_mediator import HiggsPortal, HeavyQuark

hp = HiggsPortal(mx=150.0, ms=1e3, gsxx=1.0, stheta=1e-3)
hq = HeavyQuark(mx=150.0, ms=1e3, gsxx=1.0, gsQ=1.0, mQ=1e6, QQ=1.0)

# %% {"hidden": true}
from hazma.vector_mediator import VectorMediator

vm = VectorMediator(
    mx=150.0, mv=1e3, gvxx=1.0, gvuu=0.1, gvdd=0.2, gvss=0.3, gvee=0.4, gvmumu=0.5
)

# %% {"hidden": true}
from hazma.vector_mediator import KineticMixing

km = KineticMixing(mx=150.0, mv=1e3, gvxx=1.0, eps=0.1)
print(km.gvuu)
# -0.020187846690459792
km.gvuu = 0.1
# AttributeError: Cannot set gvuu

# %% {"hidden": true}
print(ScalarMediator.list_annihilation_final_states())
# ['mu mu', 'e e', 'g g', 'pi0 pi0', 'pi pi', 's s']
print(VectorMediator.list_annihilation_final_states())
# ['mu mu', 'e e', 'pi pi', 'pi0 g', 'pi0 v', 'v v']

# %% {"hidden": true}
from hazma.scalar_mediator import ScalarMediator

sm = ScalarMediator(mx=180.0, ms=190.0, gsxx=1.0, gsff=0.1, gsGG=0.1, gsFF=0.1, lam=2e5)
e_cm = 400.0
sm.annihilation_cross_sections(e_cm)  # MeV^-2
# {'mu mu': 1.0186780542223538e-16,
#  'e e': 3.892649866478948e-21,
#  'g g': 4.8762938612605775e-21,
#  'pi0 pi0': 1.5950084474945102e-16,
#  'pi pi': 3.0338694778562897e-16,
#  's s': 3.578493120517737e-07,
#  'total': 3.5784931261653806e-07}

# %% {"hidden": true}
from hazma.scalar_mediator import ScalarMediator

sm = ScalarMediator(mx=120.0, ms=280.0, gsxx=1.0, gsff=0.1, gsGG=0.1, gsFF=0.1, lam=2e5)
sm.partial_widths()  # MeV
# {'g g': 7.363085017295395e-14,
#  'pi0 pi0': 6.523373994152967e-09,
#  'pi pi': 1.8664869506864194e-09,
#  'x x': 0.380609030857039,
#  'e e': 1.1995401279596814e-13,
#  'mu mu': 1.4482244706501055e-09,
#  'total': 0.38060904069531803}

# %% [markdown] {"heading_collapsed": true}
# ## Section 4: Building Blocks of MeV Gamma-Ray Spectra

# %% {"hidden": true}
from hazma.scalar_mediator import HeavyQuark

sm = HeavyQuark(mx=140.0, ms=1e3, gsxx=1.0, gsQ=0.1, mQ=1e3, QQ=0.1)
print(sm.gamma_ray_lines(e_cm=300.0))
# {'g g': {'energy': 150.0, 'bf': 1.285653242415087e-08}}
from hazma.vector_mediator import QuarksOnly

vm = QuarksOnly(mx=140.0, mv=1e3, gvxx=1.0, gvuu=0.1, gvdd=0.1, gvss=0.0)
print(vm.gamma_ray_lines(e_cm=300.0))
# {'pi0 g': {'energy': 119.63552908740002, 'bf': 1.0}}

# %% {"hidden": true}
from hazma.decay import neutral_pion as dnde_pi0

e_gams = np.array([100.0, 125.0, 150])  # photon energies
e_pi0 = 180.0  # pion energy
dnde_pi0(e_gams, e_pi0)
# array([0.0165965, 0.0165965, 0.       ])

# %% {"hidden": true}
from hazma.decay import muon as dnde_mu

e_gams = np.array([1.0, 10.0, 100.0])  # photon energies
e_mu = 130.0  # muon energy
dnde_mu(e_gams, e_mu)
# array([1.76076858e-02, 1.34063877e-03, 4.64775301e-08])

# %% {"hidden": true}
from hazma.decay import charged_pion as dnde_pi

e_gams = np.array([1.0, 10.0, 100.0])  # photon energies
e_pi = 150.0  # pion energy
dnde_pi(e_gams, e_pi)
# array([2.54329145e-02, 1.70431824e-03, 2.71309637e-08])

# %% {"hidden": true}
from hazma.scalar_mediator import HiggsPortal

e_cm = 305.0  # DM center of mass energy
e_gams = np.array([1.0, 10.0, 100.0])  # photon energies
hp = HiggsPortal(mx=150.0, ms=1e3, gsxx=0.7, stheta=0.1)
hp.total_spectrum(e_gams, e_cm)

# %% {"hidden": true}
hp.spectra(e_gams, e_cm)

# %% {"hidden": true}
print(hp.dnde_ee(e_gams, e_cm, spectrum_type="fsr"))
# array([0.05435176, 0.00509829, 0.00028842])
print(hp.dnde_ee(e_gams, e_cm, spectrum_type="decay"))
# array([0., 0., 0.])  # electrons don't decay
print(hp.dnde_pipi(e_gams, e_cm, spectrum_type="fsr"))
# array([1.01492577e-03, 5.00245201e-05, 0.00000000e+00])
print(hp.dnde_pipi(e_gams, e_cm, spectrum_type="decay"))
# array([5.08808459e-02, 3.42094713e-03, 2.02687537e-07])
print(hp.dnde_pipi(e_gams, e_cm))
# array([5.18957717e-02, 3.47097165e-03, 2.02687537e-07])

# %% [markdown] {"heading_collapsed": true}
# ## Section 5: Gamma Ray Spectra from DM Annihilation

# %% {"hidden": true}
from hazma.scalar_mediator import HiggsPortal

e_cm = 305.0  # DM center of mass energy
e_min, e_max = 1.0, 100.0  # define energy range
energy_res = lambda e: 0.05  # 5% energy resolution function
hp = HiggsPortal(mx=150.0, ms=1e3, gsxx=0.7, stheta=0.1)
dnde_conv = hp.total_conv_spectrum_fn(e_min, e_max, e_cm, energy_res, n_pts=1000)

# %% {"hidden": true}
dnde_conv(25.0)
# array(0.00048767)

# %% {"hidden": true}
dnde_conv.integral(25.0, 85.0)
# 0.8691133833417997

# %% [markdown] {"heading_collapsed": true}
# ## Section 6: Electron and Positron Spectra from DM annihilation

# %% {"hidden": true}
from hazma.positron_spectra import muon as dnde_p_mu

e_mu = 150.0  # muon energy
e_p = np.array([1.0, 10.0, 100.0])  # positron energies
dnde_p_mu(e_p, e_mu)
# array([4.86031362e-05, 4.56232320e-03, 4.45753994e-03])

# %% {"hidden": true}
from hazma.positron_spectra import charged_pion as dnde_p_pi

e_pi = 150.0  # charged pion energy
e_p = np.array([1.0, 10.0, 100.0])  # positron energies
dnde_p_pi(e_p, e_pi)
# array([3.84163631e-05, 3.85242442e-03, 2.55578895e-05])

# %% {"hidden": true}
from hazma.scalar_mediator import HiggsPortal

e_cm = 305.0  # DM center of mass energy
e_ps = np.array([1.0, 10.0, 100.0])  # positron energies
hp = HiggsPortal(mx=150.0, ms=1e3, gsxx=0.7, stheta=0.1)
print(hp.total_positron_spectrum(e_ps, e_cm))
# [2.60677406e-05 2.58192085e-03 4.09383294e-04]
print(hp.positron_spectra(e_ps, e_cm))
# {'mu mu': array([3.25408271e-06, 3.03854662e-04, 3.08297785e-04]),
#  'pi pi': array([2.28136579e-05, 2.27806619e-03, 1.01085509e-04]),
#  's s': array([0., 0., 0.]),
#  'total': array([2.60677406e-05, 2.58192085e-03, 4.09383294e-04])}
print(hp.positron_lines(e_cm))
# {'e e': {'energy': 152.5, 'bf': 4.109750194098895e-06}}

# %% {"hidden": true}
e_cm = 305.0  # DM center of mass energy
e_p_min, e_p_max = 1.0, 100.0  # define energy range
energy_res = lambda e: 0.05
dnde_p_conv = hp.total_conv_positron_spectrum_fn(e_p_min, e_p_max, e_cm, energy_res)
print(dnde_p_conv(20.0))
# 0.008336136582135356
print(dnde_p_conv.integral(10, 100))  # integrate spectrum
# 0.6369485109837103

# %% [markdown] {"heading_collapsed": true}
# ## Section 7: Gamma Ray Limits

# %% {"hidden": true}
from hazma.scalar_mediator import ScalarMediator
from hazma.gamma_ray_parameters import egret_diffuse, TargetParams

egret_diffuse.target  # target region information
TargetParams(J=3.79e27, dOmega=6.584844306798711)
sm = ScalarMediator(mx=150.0, ms=1e4, gsxx=1.0, gsff=0.1, gsGG=0.5, gsFF=0, lam=1e5)
sm.binned_limit(egret_diffuse)
# 7.842099364343105e-27  # cm^3 / s

# %% {"hidden": true}
from hazma.gamma_ray_parameters import A_eff_e_astrogam
from hazma.gamma_ray_parameters import energy_res_e_astrogam
from hazma.gamma_ray_parameters import T_obs_e_astrogam
from hazma.gamma_ray_parameters import gc_target, gc_bg_model

print(gc_target.J)
# 1.795e+29
print(gc_target.dOmega)  # target region information
# 0.12122929761504078
sm.unbinned_limit(
    A_eff_e_astrogam,  # effective area
    energy_res_e_astrogam,  # energy resolution
    T_obs_e_astrogam,  # observing time
    gc_target,  # target region
    gc_bg_model,
)  # background model
# 5.635716615303109e-30  # cm^3 / s

# %% [markdown] {"heading_collapsed": true}
# ## Section 8: Cosmic Microwave Background limits

# %% {"hidden": true}
from hazma.scalar_mediator import ScalarMediator

sm = ScalarMediator(mx=150.0, ms=1e4, gsxx=1.0, gsff=0.1, gsGG=0.5, gsFF=0, lam=1e5)
x_kd = 1e-4
sm.f_eff(x_kd), sm.f_eff_ep(x_kd), sm.f_eff_g(x_kd)
# (0.4371171534060625, 0.1657568724934657, 0.2713602809125968)

# %% {"hidden": true}
from hazma.cmb import p_ann_planck_temp_pol as p_ann
from hazma.scalar_mediator import HiggsPortal

p_ann
# 3.5e-31  # cm^3 / MeV / s
x_kd = 1e-4
sm = HiggsPortal(mx=150.0, ms=1e4, gsxx=1.0, stheta=0.01)
sm.cmb_limit(x_kd, p_ann)
# 1.2215415859552705e-28  # cm^3 / s

# %% [markdown] {"heading_collapsed": true}
# ## Appendix B: Basic Usage

# %% {"hidden": true}
import numpy as np
from hazma.vector_mediator import KineticMixing

params = {"mx": 250.0, "mv": 1e6, "gvxx": 1.0, "eps": 1e-3}
km = KineticMixing(**params)

# %% {"hidden": true}
km.list_annihilation_final_states()
# ['mu mu', 'e e', 'pi pi', 'pi0 g', 'pi0 v', 'v v']

# %% {"hidden": true}
cme = 2.0 * km.mx * (1.0 + 0.5 * 1e-6)
print(km.annihilation_cross_sections(cme))
# {'mu mu': 8.94839775021393e-25,
#  'e e': 9.064036692829845e-25,
#  'pi pi': 1.2940469635262499e-25,
#  'pi0 g': 5.206158864833925e-29,
#  'pi0 v': 0.0,
#  'v v': 0.0,
#  'total': 1.9307002022456507e-24}
print(km.annihilation_branching_fractions(cme))
# {'mu mu': 0.46347940191883763,
#  'e e': 0.4694688839980031,
#  'pi pi': 0.06702474894968717,
#  'pi0 g': 2.6965133472190545e-05,
#  'pi0 v': 0.0,
#  'v v': 0.0}

# %% {"hidden": true}
km.partial_widths()

# %% {"hidden": true}
photon_energies = np.array([cme / 4])
km.spectra(photon_energies, cme)
# {'mu mu': array([2.94759389e-05]),
#  'e e': array([0.00013171]),
#  'pi pi': array([2.14636322e-06]),
#  'pi0 g': array([2.29931655e-07]),
#  'pi0 v': array([0.]),
#  'v v': array([0.]),
#  'total': array([0.00016357])}

# %% {"hidden": true}
spec_funs = km.spectrum_funcs()
print(spec_funs["mu mu"](photon_energies, cme))
# [6.35970849e-05]
mumu_bf = km.annihilation_branching_fractions(cme)["mu mu"]
print(mumu_bf * spec_funs["mu mu"](photon_energies, cme))
# [2.94759389e-05]

# %% {"hidden": true}
km.total_spectrum(photon_energies, cme)
# array([0.00016357])

# %% {"hidden": true}
km.gamma_ray_lines(cme)
# {'pi0 g': {'energy': 231.78145156177675, 'bf': 2.6965133472190545e-05}}

# %% {"hidden": true}
min_photon_energy = 1e-3
max_photon_energy = cme
energy_resolution = lambda photon_energy: 1.0
number_points = 1000

spec = km.total_conv_spectrum_fn(
    min_photon_energy, max_photon_energy, cme, energy_resolution, number_points
)
spec(cme / 4)
# array(0.00168782)

# %% {"hidden": true}
from hazma.parameters import electron_mass as me

positron_energies = np.logspace(np.log10(me), np.log10(cme), num=100)
km.positron_spectra(positron_energies, cme)
km.positron_lines(cme)
km.total_positron_spectrum(positron_energies, cme)
dnde_pos = km.total_conv_positron_spectrum_fn(
    min(positron_energies),
    max(positron_energies),
    cme,
    energy_resolution,
    number_points,
)

# %% {"hidden": true}
from hazma.gamma_ray_parameters import egret_diffuse

mxs = np.linspace(me / 2.0, 250.0, num=100)
limits = np.zeros(len(mxs), dtype=float)
for i, mx in enumerate(mxs):
    km.mx = mx
    limits[i] = km.binned_limit(egret_diffuse)

# %% {"hidden": true}
from hazma.gamma_ray_parameters import gc_target, gc_bg_model
from hazma.gamma_ray_parameters import A_eff_e_astrogam, energy_res_e_astrogam
from hazma.gamma_ray_parameters import T_obs_e_astrogam

mxs = np.linspace(me / 2.0, 250.0, num=100)
limits = np.zeros(len(mxs), dtype=float)
for i, mx in enumerate(mxs):
    km.mx = mx
    limits[i] = km.unbinned_limit(
        A_eff_e_astrogam,
        energy_res_e_astrogam,
        T_obs_e_astrogam,
        gc_target,
        gc_bg_model,
    )

# %% [markdown] {"heading_collapsed": true}
# ## Appendix C: Advanced Usage

# %% {"hidden": true}
from hazma.gamma_ray_parameters import TargetParams

tp = TargetParams(J=1e29, dOmega=0.1)

# %% {"hidden": true}
from hazma.background_model import BackgroundModel

bg = BackgroundModel(e_range=[0.5, 1e4], dPhi_dEdOmega=lambda e: 2.7e-3 / e ** 2)

# %% [markdown] {"hidden": true}
# See the notebook ``hazma_example.ipynb`` for the custom model implementation.

# %% {"hidden": true}
