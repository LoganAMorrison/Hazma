from hazma import gamma_ray
import warnings
import numpy as np

warnings.filterwarnings('ignore')


def test_mu_mu_mu():
    isp_masses = np.array([100., 100.])
    particles = np.array(['muon', 'muon', 'muon'])
    cme = 5000.
    eng_gams = np.logspace(0., np.log10(5000), num=200, dtype=np.float64)
    gamma_ray.gamma_ray(isp_masses, particles, cme, eng_gams)


def test_ck_mu_mu():
    isp_masses = np.array([100., 100.])
    particles = np.array(['charged_kaon', 'muon', 'muon'])
    cme = 5000.
    eng_gams = np.logspace(0., np.log10(5000), num=200, dtype=np.float64)
    gamma_ray.gamma_ray(isp_masses, particles, cme, eng_gams)


def test_sk_mu_mu():
    isp_masses = np.array([100., 100.])
    particles = np.array(['short_kaon', 'muon', 'muon'])
    cme = 5000.
    eng_gams = np.logspace(0., np.log10(5000), num=200, dtype=np.float64)
    gamma_ray.gamma_ray(isp_masses, particles, cme, eng_gams)


def test_lk_mu_mu():
    isp_masses = np.array([100., 100.])
    particles = np.array(['long_kaon', 'muon', 'muon'])
    cme = 5000.
    eng_gams = np.logspace(0., np.log10(5000), num=200, dtype=np.float64)
    gamma_ray.gamma_ray(isp_masses, particles, cme, eng_gams)


def test_cp_mu_mu():
    isp_masses = np.array([100., 100.])
    particles = np.array(['charged_pion', 'muon', 'muon'])
    cme = 5000.
    eng_gams = np.logspace(0., np.log10(5000), num=200, dtype=np.float64)
    gamma_ray.gamma_ray(isp_masses, particles, cme, eng_gams)


def test_np_mu_mu():
    isp_masses = np.array([100., 100.])
    particles = np.array(['neutral_pion', 'muon', 'muon'])
    cme = 5000.
    eng_gams = np.logspace(0., np.log10(5000), num=200, dtype=np.float64)
    gamma_ray.gamma_ray(isp_masses, particles, cme, eng_gams)
