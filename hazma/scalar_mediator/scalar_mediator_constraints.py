import numpy as np
from ..parameters import (charged_B_mass as mB, charged_kaon_mass as mk,
                          charged_pion_mass as mpi, down_quark_mass as mdq,
                          bottom_quark_mass as mbq, strange_quark_mass as msq,
                          long_kaon_mass as mkl, top_quark_mass as mtq,
                          neutral_pion_mass as mpi0, GF, vh, Vts, Vtb, Vtd)
from ..constraint_parameters import (B_k_invis_obs, k_pi_invis_obs,
                                     kl_pi0_mu_mu_obs, kl_pi0_e_e_obs,
                                     B_k_mu_mu_obs, B_k_e_e_obs,
                                     cm_to_inv_MeV)

# TODO
# * 3-body phase space integration for s -> x x contribution to invisible
#   decays


def _lambda_ps(a, b, c):
    """Phase space factor."""
    return (a - b - c)**2 - 4. * b * c


def width_B_k_s(self):
    f0B = 0.33 / (1. - self.ms**2 / (38.0e3)**2)
    # FCNC couplings
    hSsb = (3. * np.sqrt(2) * (msq + mbq) * GF * mtq**2 * Vts.conjugate() *
            Vtb / (32. * np.pi**2 * vh))

    return (1. / (16. * np.pi * mB**3) * ((mB**2 - mk**2) / (mbq - msq))**2 *
            f0B**2 * abs(hSsb)**2 * self.gsff**2 *
            np.sqrt(_lambda_ps(mB**2, mk**2, self.ms**2)))


def constraint_B_k_invis(self):
    """Constraint function for B+ -> K+ invis."""
    ms = self.ms
    width_contr = 0.

    # Make sure scalar mass doesn't fall outside of kinematic bounds
    if np.any([s[0] <= ms**2 <= s[1] for s in B_k_invis_obs.s_bounds]):
        widths_s = self.partial_widths()
        width_s = widths_s["total"]
        width_s_sm = width_s - widths_s["x x"]  # Gamma_{S->SM}

        # Magnitude of S' 3-momentum
        ps = np.sqrt((mB - mk - ms) * (mB + mk - ms) * (mB - mk + ms) *
                     (mB + mk + ms)) / (2. * mB)
        # Probability that S decays outside the detector
        pr_invis = np.exp(-B_k_invis_obs.r_max * cm_to_inv_MeV * width_s * ms /
                          ps)

        # Compute the total contribution to the invisible decay width
        width_contr = (self.width_B_k_s() *
                       (widths_s["x x"] + pr_invis * width_s_sm) / width_s)

    return B_k_invis_obs.width_bound - width_contr


def constraint_B_k_mu_mu(self):
    """Constraint function for B+ -> K+ mu+ mu-."""
    ms = self.ms
    width_contr = 0.

    # Make sure scalar mass doesn't fall outside of kinematic bounds
    if np.any([s[0] <= ms**2 <= s[1] for s in B_k_mu_mu_obs.s_bounds]):
        widths_s = self.partial_widths()
        width_s = widths_s["total"]

        # Magnitude of S' 3-momentum
        ps = np.sqrt((mB - mk - ms) * (mB + mk - ms) * (mB - mk + ms) *
                     (mB + mk + ms)) / (2. * mB)
        # Probability that S decays close to the primary vertex
        pr_vis = 1. - np.exp(-B_k_mu_mu_obs.r_max * cm_to_inv_MeV * width_s *
                             ms / ps)

        # print pr_vis
        # print widths_s["mu mu"] / width_s

        # Compute the contribution to the mu mu decay width
        width_contr = self.width_B_k_s() * widths_s["mu mu"] / width_s * pr_vis

    return B_k_mu_mu_obs.width_bound - width_contr


def constraint_B_k_e_e(self):
    """Constraint function for B+ -> K+ e+ e-."""
    ms = self.ms
    width_contr = 0.

    # Make sure scalar mass doesn't fall outside of kinematic bounds
    if np.any([s[0] <= ms**2 <= s[1] for s in B_k_e_e_obs.s_bounds]):
        widths_s = self.partial_widths()
        width_s = widths_s["total"]

        # Magnitude of S' 3-momentum
        ps = np.sqrt((mB - mk - ms) * (mB + mk - ms) * (mB - mk + ms) *
                     (mB + mk + ms)) / (2. * mB)
        # Probability that S decays close to the primary vertex
        pr_vis = 1. - np.exp(-B_k_e_e_obs.r_max * cm_to_inv_MeV * width_s *
                             ms / ps)

        # Compute the contribution to the e e decay width
        width_contr = self.width_B_k_s() * widths_s["e e"] / width_s * pr_vis

    return B_k_e_e_obs.width_bound - width_contr


def width_k_pi_s(self):
    # FCNC couplings
    hSsd = (3. * np.sqrt(2) * (msq + mdq) * GF * mtq**2 * Vts.conjugate() *
            Vtd / (32. * np.pi**2 * vh))

    return (1. / (16. * np.pi * mk**3) * ((mk**2 - mpi**2) / (msq - mdq))**2 *
            abs(hSsd)**2 * self.gsff**2 *
            np.sqrt(_lambda_ps(mk**2, mpi**2, self.ms**2)))


def constraint_k_pi_invis(self):
    """Constraint function for K+ -> pi+ invis."""
    width_contr = 0.
    ms = self.ms

    # Make sure scalar mass doesn't fall outside of kinematic bounds
    if np.any([s[0] <= ms**2 <= s[1] for s in k_pi_invis_obs.s_bounds]):
        widths_s = self.partial_widths()
        width_s = widths_s["total"]
        width_s_sm = width_s - widths_s["x x"]  # Gamma_{S->SM}

        # Magnitude of S' 3-momentum
        ps = np.sqrt((mk - mpi - ms) * (mk + mpi - ms) * (mk - mpi + ms) *
                     (mk + mpi + ms)) / (2. * mk)
        # Probability that S decays outside the detector
        pr_invis = np.exp(-k_pi_invis_obs.r_max * cm_to_inv_MeV * width_s *
                          ms / ps)

        # Compute the total contribution to the invisible decay width
        width_contr = (self.width_k_pi_s() *
                       (widths_s["x x"] + pr_invis * width_s_sm) / width_s)

    return k_pi_invis_obs.width_bound - width_contr


def width_kl_pi0_s(self):
    # FCNC couplings
    hSsd = (3. * np.sqrt(2) * (msq + mdq) * GF * mtq**2 * Vts.conjugate() *
            Vtd / (32. * np.pi**2 * vh))

    return (1. / (16. * np.pi * mkl**3) * ((mkl**2 - mpi0**2)/(msq - mdq))**2 *
            hSsd.imag**2 * self.gsff *
            np.sqrt(_lambda_ps(mk**2, mpi0**2, self.ms**2)))


def constraint_kl_pi0_mu_mu(self):
    """Constraint function for K_L -> pi0 mu+ mu-."""
    width_contr = 0.
    ms = self.ms

    # Make sure scalar mass doesn't fall outside of kinematic bounds
    if np.any([s[0] <= ms**2 <= s[1] for s in kl_pi0_mu_mu_obs.s_bounds]):
        widths_s = self.partial_widths()
        width_s = widths_s["total"]

        # Magnitude of S' 3-momentum
        ps = np.sqrt((mkl - mpi0 - ms) * (mkl + mpi0 - ms) *
                     (mkl - mpi0 + ms) * (mkl + mpi0 + ms)) / (2. * mkl)
        # Probability that S decays close to the primary vertex
        pr_vis = 1. - np.exp(-kl_pi0_mu_mu_obs.r_max * cm_to_inv_MeV *
                             width_s * ms / ps)

        # Compute the contribution to the mu mu decay width
        width_contr = (self.width_kl_pi0_s() * widths_s["mu mu"] / width_s *
                       pr_vis)

    return kl_pi0_mu_mu_obs.width_bound - width_contr


def constraint_kl_pi0_e_e(self):
    """Constraint function for K_L -> pi0 e+ e-."""
    width_contr = 0.
    ms = self.ms

    # Make sure scalar mass doesn't fall outside of kinematic bounds
    if np.any([s[0] <= ms**2 <= s[1] for s in kl_pi0_e_e_obs.s_bounds]):
        widths_s = self.partial_widths()
        width_s = widths_s["total"]

        # Magnitude of S' 3-momentum
        ps = np.sqrt((mkl - mpi0 - ms) * (mkl + mpi0 - ms) *
                     (mkl - mpi0 + ms) * (mkl + mpi0 + ms)) / (2. * mkl)
        # Probability that S decays close to the primary vertex
        pr_vis = 1. - np.exp(-kl_pi0_e_e_obs.r_max * cm_to_inv_MeV *
                             width_s * ms / ps)
        # Compute the contribution to the e e decay width
        width_contr = (self.width_kl_pi0_s() * widths_s["e e"] / width_s *
                       pr_vis)

    return kl_pi0_e_e_obs.width_bound - width_contr


def constraints(self):
    return {"B -> k invis": self.constraint_B_k_invis,
            "B -> k mu mu": self.constraint_B_k_mu_mu,
            # "B -> k e e": self.constraint_B_k_e_e,  # worthlessly weak!
            "k -> pi invis": self.constraint_k_pi_invis,
            "kl -> pi0 mu mu": self.constraint_kl_pi0_mu_mu,
            "kl -> pi0 e e": self.constraint_kl_pi0_e_e}
