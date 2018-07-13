import numpy as np

from .pseudo_scalar_mediator_fsr import dnde_xx_to_p_to_ffg
from .pseudo_scalar_mediator_cross_sections import branching_fractions

from ..decay import muon

from ..parameters import muon_mass as mmu
from ..parameters import electron_mass as me
from ..parameters import charged_pion_mass as mpi
from ..parameters import neutral_pion_mass as mpi0

from .pseudo_scalar_mediator_mat_elem_sqrd_rambo import msqrd_xx_to_p_to_000
from .pseudo_scalar_mediator_mat_elem_sqrd_rambo import msqrd_xx_to_p_to_pm0

from ..gamma_ray import gamma_ray

# Stuff needed to compute FSR from x xbar -> P -> pip pim pi0
# from ..gamma_ray import gamma_ray_rambo
# from .pseudo_scalar_mediator_mat_elem_sqrd_rambo import msqrd_xx_to_p_to_pm0g


# TODO: pp spectrum. Gonna need Logan to do this since it requires cython...
def dnde_pp(egams, Q, params, mode="total"):
    eng_p = Q / 2.

    pass


def dnde_ee(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return (dnde_ee(egams, cme, params, 'FSR') +
                dnde_ee(egams, cme, params, 'Decay'))
    elif spectrum_type == 'FSR':
        return dnde_xx_to_p_to_ffg(egams, cme, me, params)
    elif spectrum_type == 'Decay':
        return np.array([0.0 for _ in range(len(egams))])
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_mumu(egams, cme, params, spectrum_type='All'):
    if spectrum_type == 'All':
        return (dnde_mumu(egams, cme, params, 'FSR') +
                dnde_mumu(egams, cme, params, 'Decay'))
    elif spectrum_type == 'FSR':
        return dnde_xx_to_p_to_ffg(egams, cme, mmu, params)
    elif spectrum_type == 'Decay':
        return 2. * muon(egams, cme / 2.0)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_pi0pipi(egams, cme, params, spectrum_type='All'):

    if cme < 2. * mpi + mpi0:
        return np.array([0.0 for _ in range(len(egams))])

    if spectrum_type == 'All':
        return (dnde_pi0pipi(egams, cme, params, 'FSR') +
                dnde_pi0pipi(egams, cme, params, 'Decay'))

    elif spectrum_type == 'FSR':
        # Define the tree level and radiative matrix element squared for
        # RAMBO. These need to be of the form double(*func)(np.ndarray) where
        # the np.ndarray is a list of 4-momenta. Note msqrd_xx_to_p_to_pm0
        # takes params as the second argument. The first and second FS
        # particles must be the charged pions and the third a neutral pion.

        # NOTE: I am removing this because it takes too long and need to
        # be extrapolated and evaluated at the correct egams

        """
        def msqrd_tree(momenta):
            return msqrd_xx_to_p_to_pm0(momenta, params)

        def msqrd_rad(momenta):
            return msqrd_xx_to_p_to_pm0g(momenta, params)

        isp_masses = np.array([params.mx, params.mx])
        fsp_masses = np.array([mpi, mpi, mpi0, 0.0])

        return gamma_ray_rambo(isp_masses, fsp_masses, cme,
                               num_ps_pts=50000, num_bins=150,
                               mat_elem_sqrd_tree=msqrd_tree,
                               mat_elem_sqrd_rad=msqrd_rad)
        """

        return np.array([0.0 for _ in range(len(egams))])

    elif spectrum_type == 'Decay':
        # Define the matrix element squared for RAMBO. This needs to be
        # of the form double(*func)(np.ndarray) where the np.ndarray is
        # a list of 4-momenta. Note msqrd_xx_to_p_to_pm0 takes params as the
        # second argument. The first and second FS particles must be the
        # charged pions and the third a neutral pion.
        def msqrd_tree(momenta):
            return msqrd_xx_to_p_to_pm0(momenta, params)

        return gamma_ray(["charged_pion", "charged_pion", "neutral_pion"],
                         cme, egams, num_ps_pts=1000,
                         mat_elem_sqrd=msqrd_tree)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def dnde_pi0pi0pi0(egams, cme, params, spectrum_type='All'):
    """
    Return the gamma ray spectrum for dark matter annihilations into
    three neutral pions.
    """
    if cme < 3. * mpi0:
        return np.array([0.0 for _ in range(len(egams))])

    if spectrum_type == 'All':
        return dnde_pi0pi0pi0(egams, cme, params, 'Decay')

    elif spectrum_type == 'FSR':
        return np.array([0.0 for _ in range(len(egams))])

    elif spectrum_type == 'Decay':
        # Define the matrix element squared for RAMBO. This needs to be
        # of the form double(*func)(np.ndarray) where the np.ndarray is
        # a list of 4-momenta. Note msqrd_xx_to_p_to_000 takes params as the
        # second argument.
        def msqrd_tree(momenta):
            return msqrd_xx_to_p_to_000(momenta, params)

        return gamma_ray(["neutral_pion", "neutral_pion", "neutral_pion"],
                         cme, egams, num_ps_pts=1000,
                         mat_elem_sqrd=msqrd_tree)
    else:
        raise ValueError("Type {} is invalid. Use 'All', 'FSR' or \
                         'Decay'".format(spectrum_type))


def spectra(egams, cme, params):
    """Compute the total spectrum from two fermions annihilating through a
    scalar mediator to mesons and leptons.

    Parameters
    ----------
    cme : float
        Center of mass energy.
    egams : array-like, optional
        Gamma ray energies to evaluate the spectrum at.

    Returns
    -------
    specs : dictionary
        Dictionary of the spectra. The keys are 'total', 'mu mu', 'e e',
        'pi0 pi0', 'pi pi', 'k k', 'k0 k0'.
    """
    # Compute branching fractions
    bfs = branching_fractions(cme, params)

    # Only compute the spectrum if the channel's branching fraction is nonzero
    def spec_helper(bf, specfn):
        if bf != 0:
            return bf * specfn(egams, cme, params)
        else:
            return np.zeros(egams.shape)

    # Pions. TODO: use rambo to compute this.
    pi0pipi_spec = spec_helper(bfs['pi0 pi pi'], dnde_pi0pipi)
    pi0pi0pi0_spec = spec_helper(bfs['pi0 pi0 pi0'], dnde_pi0pi0pi0)

    # Leptons
    mumu_spec = spec_helper(bfs['mu mu'], dnde_mumu)
    ee_spec = spec_helper(bfs['e e'], dnde_ee)

    # Mediator
    pp_spec = spec_helper(bfs['p p'], dnde_pp)

    # Compute total spectrum
    total = pi0pipi_spec + mumu_spec + ee_spec + pp_spec + pi0pi0pi0_spec

    # Define dictionary for spectra
    specs = {'total': total,
             'pi0 pi pi': pi0pipi_spec,
             'pi0 pi0 pi0': pi0pi0pi0_spec,
             'mu mu': mumu_spec,
             'e e': ee_spec,
             'p p': pp_spec}

    return specs


def gamma_ray_lines(cme, params):
    bf = branching_fractions(cme, params)["g g"]

    return {"g g": {"energy": cme / 2.0, "bf": bf}}
