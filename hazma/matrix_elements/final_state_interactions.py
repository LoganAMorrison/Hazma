from ..parameters import vh, b0, alpha_em, fpi
from ..parameters import charged_pion_mass as mpi
from ..parameters import up_quark_mass as muq
from ..parameters import down_quark_mass as mdq

import numpy as np

import os


DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "unitarization",
                         "final_state_int_unitarizated.dat")

unitarizated_data = np.genfromtxt(DATA_PATH, delimiter=',')

metric_diag = np.array([1.0, -1.0, -1.0, -1.0])


def minkowski_dot(fv1, fv2):
    return np.sum(metric_diag[:] * fv1[:] * fv2[:])


def unit_matrix_elem_sqrd(cme):
    t_mod_sqrd = np.interp(cme, unitarizated_data[0], unitarizated_data[1])
    additional_factor = (2 * cme**2 - mpi**2) / (32. * fpi**2 * np.pi)
    return t_mod_sqrd / additional_factor**2


def xx_to_s_to_pipi(moms, mx, ms, cxxs, cffs, cggs, vs):

    pp, pm = moms

    s = minkowski_dot(pp + pm, pp + pm)

    mat_elem_sqrd = (-2 * cxxs**2 * (4 * mx**2 - s) *
                     (2 * cggs * (2 * mpi**2 - s) *
                      (-9 * vh - 9 * cffs * vs + 2 * cggs * vs) *
                      (9 * vh + 8 * cggs * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * cggs * vs) *
                      (54 * cggs * vh - 32 * cggs**2 * vs + 9 * cffs *
                         (9 * vh + 16 * cggs * vs)))**2) / \
        ((ms**2 - s)**2 *
         (9 * vh + 9 * cffs * vs - 2 * cggs * vs) ** 2 *
         (9 * vh + 4 * cggs * vs)**2 * (9 * vh + 8 * cggs * vs)**2)

    return mat_elem_sqrd * unit_matrix_elem_sqrd(np.sqrt(s))


def xx_to_s_to_pipi2(moms, mx, ms, cxxs, cffs, cggs, vs):

    pp, pm = moms

    s = minkowski_dot(pp + pm, pp + pm)

    mat_elem_sqrd = (-2 * cxxs**2 * (4 * mx**2 - s) *
                     (2 * cggs * (2 * mpi**2 - s) *
                      (-9 * vh - 9 * cffs * vs + 2 * cggs * vs) *
                      (9 * vh + 8 * cggs * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * cggs * vs) *
                      (54 * cggs * vh - 32 * cggs**2 * vs + 9 * cffs *
                         (9 * vh + 16 * cggs * vs)))**2) / \
        ((ms**2 - s)**2 *
         (9 * vh + 9 * cffs * vs - 2 * cggs * vs) ** 2 *
         (9 * vh + 4 * cggs * vs)**2 * (9 * vh + 8 * cggs * vs)**2)

    return mat_elem_sqrd


def xx_to_s_to_pipig(moms, mx, ms, cxxs, cffs, cggs, vs):

    pp, pm, pg = moms

    Q = (np.sum(moms, 0))[0]

    s = minkowski_dot(pp + pm, pp + pm)
    t = minkowski_dot(pg + pm, pg + pm)
    u = minkowski_dot(pg + pp, pg + pp)

    e = np.sqrt(4 * np.pi * alpha_em)

    mat_elem_sqrd = (-16 * cxxs**2 *
                     (-4 * mx**2 + Q**2) *
                     (4 * mpi**6 - s * t * u + mpi**2 *
                      (t + u) * (s + t + u) - mpi**4 * (s + 4 * (t + u))) *
                     (2 * cggs * (4 * mpi**2 - s - t - u) *
                      (-9 * vh - 9 * cffs * vs + 2 * cggs * vs) *
                      (9 * vh + 8 * cggs * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * cggs * vs) *
                      (54 * cggs * vh - 32 * cggs**2 * vs + 9 * cffs *
                       (9 * vh + 16 * cggs * vs)))**2 * e**2) / \
        ((ms**2 - Q**2)**2 *
         (mpi**2 - t)**2 * (mpi**2 - u)**2 *
         (9 * vh + 9 * cffs * vs - 2 * cggs * vs)**2 *
         (9 * vh + 4 * cggs * vs)**2 * (9 * vh + 8 * cggs * vs)**2)

    pions_inv_mass = np.sqrt(minkowski_dot(pp + pm, pp + pm))

    return unit_matrix_elem_sqrd(pions_inv_mass) * mat_elem_sqrd


def xx_to_s_to_pipig2(moms, mx, ms, cxxs, cffs, cggs, vs):

    pp, pm, pg = moms

    Q = (np.sum(moms, 0))[0]

    s = minkowski_dot(pp + pm, pp + pm)
    t = minkowski_dot(pg + pm, pg + pm)
    u = minkowski_dot(pg + pp, pg + pp)

    e = np.sqrt(4 * np.pi * alpha_em)

    mat_elem_sqrd = (-16 * cxxs**2 *
                     (-4 * mx**2 + Q**2) *
                     (4 * mpi**6 - s * t * u + mpi**2 *
                      (t + u) * (s + t + u) - mpi**4 * (s + 4 * (t + u))) *
                     (2 * cggs * (4 * mpi**2 - s - t - u) *
                      (-9 * vh - 9 * cffs * vs + 2 * cggs * vs) *
                      (9 * vh + 8 * cggs * vs) +
                      b0 * (mdq + muq) * (9 * vh + 4 * cggs * vs) *
                      (54 * cggs * vh - 32 * cggs**2 * vs + 9 * cffs *
                       (9 * vh + 16 * cggs * vs)))**2 * e**2) / \
        ((ms**2 - Q**2)**2 *
         (mpi**2 - t)**2 * (mpi**2 - u)**2 *
         (9 * vh + 9 * cffs * vs - 2 * cggs * vs)**2 *
         (9 * vh + 4 * cggs * vs)**2 * (9 * vh + 8 * cggs * vs)**2)

    return mat_elem_sqrd
