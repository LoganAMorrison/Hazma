# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# **This notebook collects all the code snippets from the paper introducing Hazma.** Commented lines indicate the expected output.

# %%
import numpy as np

# %% [markdown] heading_collapsed=true
# ## Section 3: Particle Physics Framework

# %% hidden=true
from hazma.scalar_mediator import ScalarMediator

sm = ScalarMediator(mx=150.0, ms=1e3, gsxx=1.0, gsff=0.1, gsGG=0.1, gsFF=0.1, lam=2e5)
display(sm.gsff)
# 0.1
sm.gsff = 0.5
display(sm.gsff)
# 0.5

# %% hidden=true
from hazma.scalar_mediator import HiggsPortal, HeavyQuark

hp = HiggsPortal(mx=150.0, ms=1e3, gsxx=1.0, stheta=1e-3)
hq = HeavyQuark(mx=150.0, ms=1e3, gsxx=1.0, gsQ=1.0, mQ=1e6, QQ=1.0)

# %% hidden=true
from hazma.vector_mediator import VectorMediator

vm = VectorMediator(
    mx=150.0, mv=1e3, gvxx=1.0, gvuu=0.1, gvdd=0.2, gvss=0.3, gvee=0.4, gvmumu=0.5
)

# %% hidden=true
from hazma.vector_mediator import KineticMixing

km = KineticMixing(mx=150.0, mv=1e3, gvxx=1.0, eps=0.1)
display(km.gvuu)
# -0.020187846690459792
km.gvuu = 0.1
# AttributeError: Cannot set gvuu

# %% hidden=true
display(ScalarMediator.list_annihilation_final_states())
# ['mu mu', 'e e', 'g g', 'pi0 pi0', 'pi pi', 's s']
display(VectorMediator.list_annihilation_final_states())
# ['mu mu', 'e e', 'pi pi', 'pi0 g', 'pi0 v', 'v v']

# %% hidden=true
from hazma.scalar_mediator import ScalarMediator

sm = ScalarMediator(mx=180.0, ms=190.0, gsxx=1.0, gsff=0.1, gsGG=0.1, gsFF=0.1, lam=2e5)
e_cm = 400.0
sm.annihilation_cross_sections(e_cm)  # MeV^-2
# {'mu mu': 1.018678054222354e-16,
#  'e e': 3.892649866478948e-21,
#  'g g': 2.4381469306302895e-21,
#  'pi0 pi0': 7.975042237472552e-17,
#  'pi pi': 1.5169347389281449e-16,
#  's s': 1.78924656025887e-07,
#  'total': 1.7892465635920504e-07}

# %% hidden=true
from hazma.scalar_mediator import ScalarMediator

sm = ScalarMediator(mx=120.0, ms=280.0, gsxx=1.0, gsff=0.1, gsGG=0.1, gsFF=0.1, lam=2e5)
sm.partial_widths()  # MeV
# {'g g': 1.472617003459079e-13,
#  'pi0 pi0': 3.261686997076484e-09,
#  'pi pi': 1.8664869506864194e-09,
#  'x x': 1.522436123428156,
#  'e e': 4.798160511838726e-13,
#  'mu mu': 5.792897882600423e-09,
#  'total': 1.522436134349855}

# %% [markdown] heading_collapsed=true
# ## Section 4: Building Blocks of MeV Gamma-Ray Spectra

# %% hidden=true
from hazma.scalar_mediator import HeavyQuark

sm = HeavyQuark(mx=140.0, ms=1e3, gsxx=1.0, gsQ=0.1, mQ=1e3, QQ=0.1)
display(sm.gamma_ray_lines(e_cm=300.0))
# {'g g': {'energy': 150.0, 'bf': 1.2856532424150869e-08}}
from hazma.vector_mediator import QuarksOnly

vm = QuarksOnly(mx=140.0, mv=1e3, gvxx=1.0, gvuu=0.1, gvdd=0.1, gvss=0.0)
display(vm.gamma_ray_lines(e_cm=300.0))
# {'pi0 g': {'energy': 119.63552908740002, 'bf': 1.0}}

# %% hidden=true
from hazma.decay import neutral_pion as dnde_pi0

e_gams = np.array([100.0, 125.0, 150])  # photon energies
e_pi0 = 180.0  # pion energy
dnde_pi0(e_gams, e_pi0)
# array([0.0165965, 0.0165965, 0.       ])

# %% hidden=true
from hazma.decay import muon as dnde_mu

e_gams = np.array([1.0, 10.0, 100.0])  # photon energies
e_mu = 130.0  # muon energy
dnde_mu(e_gams, e_mu)
# array([1.76076858e-02, 1.34063877e-03, 4.64775301e-08])

# %% hidden=true
from hazma.decay import charged_pion as dnde_pi

e_gams = np.array([1.0, 10.0, 100.0])  # photon energies
e_pi = 150.0  # pion energy
dnde_pi(e_gams, e_pi)
# array([1.76949944e-02, 1.32675207e-03, 1.16607174e-09])

# %% hidden=true
from hazma.scalar_mediator import HiggsPortal

e_cm = 305.0  # DM center of mass energy
e_gams = np.array([1.0, 10.0, 100.0])  # photon energies
hp = HiggsPortal(mx=150.0, ms=1e3, gsxx=0.7, stheta=0.1)
hp.total_spectrum(e_gams, e_cm)
# array([0.02484114, 0.00186874, 0.01827828])

# %% hidden=true
hp.spectra(e_gams, e_cm)
# {'mu mu': array([4.87977497e-03, 3.83154301e-04, 1.22977086e-06]),
#  'e e': array([4.19129970e-07, 3.93151015e-08, 2.22413275e-09]),
#  'pi0 pi0': array([0.        , 0.        , 0.01827702]),
#  'pi pi': array([1.99609455e-02, 1.48554303e-03, 2.28115016e-08]),
#  's s': array([0., 0., 0.]),
#  'total': array([0.02484114, 0.00186874, 0.01827828])}

# %% hidden=true
display(hp.dnde_ee(e_gams, e_cm, spectrum_type="fsr"))
# array([0.05435176, 0.00509829, 0.00028842])
display(hp.dnde_ee(e_gams, e_cm, spectrum_type="decay"))
# array([0., 0., 0.])  # electrons don't decay
display(hp.dnde_pipi(e_gams, e_cm, spectrum_type="fsr"))
# array([1.01492577e-03, 5.00245201e-05, 0.00000000e+00])
display(hp.dnde_pipi(e_gams, e_cm, spectrum_type="decay"))
# array([3.53972084e-02, 2.65985674e-03, 4.16120296e-08])
display(hp.dnde_pipi(e_gams, e_cm))
# array([3.64121342e-02, 2.70988126e-03, 4.16120296e-08])

# %% [markdown] heading_collapsed=true
# ## Section 5: Gamma Ray Spectra from DM Annihilation

# %% hidden=true
from hazma.scalar_mediator import HiggsPortal

e_cm = 305.0  # DM center of mass energy
e_min, e_max = 1.0, 100.0  # define energy range
energy_res = lambda e: 0.05  # 5% energy resolution function
hp = HiggsPortal(mx=150.0, ms=1e3, gsxx=0.7, stheta=0.1)
dnde_conv = hp.total_conv_spectrum_fn(e_min, e_max, e_cm, energy_res, n_pts=1000)

# %% hidden=true
dnde_conv(25.0)
# array(0.00046904)

# %% hidden=true
dnde_conv.integral(25.0, 85.0)
# 0.815998464406668

# %% [markdown] heading_collapsed=true
# ## Section 6: Electron and Positron Spectra from DM annihilation

# %% hidden=true
from hazma.positron_spectra import muon as dnde_p_mu

e_mu = 150.0  # muon energy
e_p = np.array([1.0, 10.0, 100.0])  # positron energies
dnde_p_mu(e_p, e_mu)
# array([4.86031362e-05, 4.56232320e-03, 4.45753994e-03])

# %% hidden=true
from hazma.positron_spectra import charged_pion as dnde_p_pi

e_pi = 150.0  # charged pion energy
e_p = np.array([1.0, 10.0, 100.0])  # positron energies
dnde_p_pi(e_p, e_pi)
# array([3.84163631e-05, 3.85242442e-03, 2.55578895e-05])

# %% hidden=true
from hazma.scalar_mediator import HiggsPortal

e_cm = 305.0  # DM center of mass energy
e_ps = np.array([1.0, 10.0, 100.0])  # positron energies
hp = HiggsPortal(mx=150.0, ms=1e3, gsxx=0.7, stheta=0.1)
display(hp.total_positron_spectrum(e_ps, e_cm))
# array([2.75093729e-05, 2.70739898e-03, 6.73319534e-04])
display(hp.positron_spectra(e_ps, e_cm))
# {'mu mu': array([6.10588051e-06, 5.70145391e-04, 5.78482356e-04]),
#  'pi pi': array([2.14034924e-05, 2.13725359e-03, 9.48371777e-05]),
#  's s': array([0., 0., 0.]),
#  'total': array([2.75093729e-05, 2.70739898e-03, 6.73319534e-04])}
display(hp.positron_lines(e_cm))
# {'e e': {'energy': 152.5, 'bf': 7.711433862697906e-06}}

# %% hidden=true
e_cm = 305.0  # DM center of mass energy
e_p_min, e_p_max = 1.0, 100.0  # define energy range
energy_res = lambda e: 0.05
dnde_p_conv = hp.total_conv_positron_spectrum_fn(e_p_min, e_p_max, e_cm, energy_res)
display(dnde_p_conv(20.0))
# array(0.00864851)
display(dnde_p_conv.integral(10, 100))  # integrate spectrum
# 0.6538810882108401

# %% [markdown] heading_collapsed=true
# ## Section 7: Gamma Ray Limits

# %% hidden=true
from hazma.scalar_mediator import ScalarMediator
from hazma.gamma_ray_parameters import egret_diffuse, TargetParams

egret_diffuse.target  # target region information
TargetParams(J=3.79e27, dOmega=6.584844306798711)
sm = ScalarMediator(mx=150.0, ms=1e4, gsxx=1.0, gsff=0.1, gsGG=0.5, gsFF=0, lam=1e5)
sm.binned_limit(egret_diffuse)
# 2.2809005757912853e-27  # cm^3 / s

# %% hidden=true
from hazma.gamma_ray_parameters import A_eff_e_astrogam
from hazma.gamma_ray_parameters import energy_res_e_astrogam
from hazma.gamma_ray_parameters import gc_targets, gc_bg_model

T_obs_e_astrogam = 1e7
gc_target = gc_targets["nfw"]["1 arcmin cone"]

display(gc_target.J)
# 6.972e+32
display(gc_target.dOmega)  # target region information
# 2.66e-07
sm.unbinned_limit(
    A_eff_e_astrogam,  # effective area
    energy_res_e_astrogam,  # energy resolution
    T_obs_e_astrogam,  # observing time
    gc_target,  # target region
    gc_bg_model,
)  # background model
# 1.8521346229682218e-30  # cm^3 / s

# %% [markdown] heading_collapsed=true
# ## Section 8: Cosmic Microwave Background limits

# %% hidden=true
from hazma.scalar_mediator import HiggsPortal

sm = HiggsPortal(mx=150.0, ms=1.5*150, gsxx=1.0, stheta=0.1)
x_kd = 1e-6
sm.f_eff(x_kd), sm.f_eff_ep(x_kd), sm.f_eff_g(x_kd)
# (0.4234440957173111, 0.18622537016754176, 0.23721872554976936)

# %% hidden=true
from hazma.cmb import p_ann_planck_temp_pol as p_ann
from hazma.scalar_mediator import HiggsPortal

p_ann
# 3.5e-31  # cm^3 / MeV / s
x_kd = 1e-6
sm = HiggsPortal(mx=150.0, ms=1.5*150, gsxx=1.0, stheta=0.1)
sm.cmb_limit(x_kd, p_ann)
# 1.2398330861377438e-28  # cm^3 / s

# %% [markdown] heading_collapsed=true
# ## Appendix B: Basic Usage

# %% hidden=true
import numpy as np
from hazma.vector_mediator import KineticMixing

params = {"mx": 250.0, "mv": 1e6, "gvxx": 1.0, "eps": 1e-3}
km = KineticMixing(**params)

# %% hidden=true
km.list_annihilation_final_states()
# ['mu mu', 'e e', 'pi pi', 'pi0 g', 'pi0 v', 'v v']

# %% hidden=true
cme = 2.0 * km.mx * (1.0 + 0.5 * 1e-6)
display(km.annihilation_cross_sections(cme))
# {'mu mu': 8.998735387276086e-25,
#  'e e': 9.115024836416874e-25,
#  'pi pi': 1.3013263970304e-25,
#  'pi0 g': 1.7451483984993156e-29,
#  'pi0 v': 0.0,
#  'v v': 0.0,
#  'total': 1.941526113556321e-24}
display(km.annihilation_branching_fractions(cme))
# {'mu mu': 0.4634877339245762,
#  'e e': 0.46947732367713324,
#  'pi pi': 0.06702595385888176,
#  'pi0 g': 8.988539408840104e-06,
#  'pi0 v': 0.0,
#  'v v': 0.0}

# %% hidden=true
km.partial_widths()
# {'pi pi': 0.0006080948890354345,
#  'pi0 g': 0.23374917012731816,
#  'x x': 26525.823848648604,
#  'e e': 0.0024323798404358825,
#  'mu mu': 0.0024323798404358803,
#  'total': 26526.0630706733}

# %% hidden=true
# Specify photon energies to compute spectrum at
photon_energies = np.array([cme / 4])
km.spectra(photon_energies, cme)
# {'mu mu': array([2.8227189e-05]),
#  'e e': array([0.00013172]),
#  'pi pi': array([2.1464018e-06]),
#  'pi0 g': array([7.66452627e-08]),
#  'pi0 v': array([0.]),
#  'v v': array([0.]),
#  'total': array([0.00016217])}

# %% hidden=true
spec_funs = km.spectrum_funcs()
display(spec_funs["mu mu"](photon_energies, cme))
# array([6.09016958e-05])
mumu_bf = km.annihilation_branching_fractions(cme)["mu mu"]
display(mumu_bf * spec_funs["mu mu"](photon_energies, cme))
# array([2.8227189e-05])

# %% hidden=true
km.total_spectrum(photon_energies, cme)
# array([0.00016217])

# %% hidden=true
km.gamma_ray_lines(cme)
# {'pi0 g': {'energy': 231.78145156177675, 'bf': 8.988539408840104e-06}}

# %% hidden=true
min_photon_energy = 1e-3
max_photon_energy = cme
energy_resolution = lambda photon_energy: 1.0
number_points = 1000

spec = km.total_conv_spectrum_fn(
    min_photon_energy, max_photon_energy, cme, energy_resolution, number_points
)
spec(cme / 4)
# array(0.00167605)

# %% hidden=true
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

# %% hidden=true
from hazma.gamma_ray_parameters import egret_diffuse

mxs = np.linspace(me / 2.0, 250.0, num=100)
limits = np.zeros(len(mxs), dtype=float)
for i, mx in enumerate(mxs):
    km.mx = mx
    limits[i] = km.binned_limit(egret_diffuse)

# %% hidden=true
from hazma.gamma_ray_parameters import gc_targets, gc_bg_model
from hazma.gamma_ray_parameters import A_eff_e_astrogam, energy_res_e_astrogam

gc_target = gc_targets["nfw"]["1 arcmin cone"]
T_obs_e_astrogam = 1e7  # s

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

# %% hidden=true
from hazma.scalar_mediator import ScalarMediator
from hazma.parameters import vh

class HiggsPortal(ScalarMediator):
    def __init__(self, mx, ms, gsxx, stheta):
        self._lam = vh
        self._stheta = stheta
        super(HiggsPortal, self).__init__(
            mx, ms, gsxx, stheta, 3.*stheta, -5.*stheta/6., vh
        )

    @property
    def stheta(self):
        return self._stheta

    @stheta.setter
    def stheta(self, stheta):
        self._stheta = stheta
        self.gsff = stheta
        self.gsGG = 3. * stheta
        self.gsFF = - 5. * stheta / 6.

    # Hide underlying properties' setters
    @ScalarMediator.gsff.setter
    def gsff(self, gsff):
        raise AttributeError("Cannot set gsff")

    @ScalarMediator.gsGG.setter
    def gsGG(self, gsGG):
        raise AttributeError("Cannot set gsGG")

    @ScalarMediator.gsFF.setter
    def gsFF(self, gsFF):
        raise AttributeError("Cannot set gsFF")


# %% [markdown] heading_collapsed=true
# ## Appendix C: Advanced Usage

# %% hidden=true
from hazma.gamma_ray_parameters import TargetParams

tp = TargetParams(J=1e29, dOmega=0.1)

# %% hidden=true
from hazma.background_model import BackgroundModel

bg = BackgroundModel(e_range=[0.5, 1e4], dPhi_dEdOmega=lambda e: 2.7e-3 / e ** 2)

# %% [markdown] hidden=true
# See the notebook ``hazma_example.ipynb`` for the custom model implementation.

# %% hidden=true
