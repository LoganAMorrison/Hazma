import numpy as np


def sigma_f1f1_to_f2f2(Q, mf1, mf2, ms=0, mp=0, mv=0, ma=0,
                       gsff1=0, gsff2=0, gpff1=0, gpff2=0,
                       gvff1=0, gvff2=0, gaff1=0, gaff2=0):

    uf1, uf2 = mf1 / Q, mf2 / Q
    us, up = ms / Q, mp / Q
    uv, ua = mv / Q, ma / Q

    return ((np.sqrt(1 - 4*uf2**2) *
             ((8*gaff1**2*gaff2**2 *
               (1 + 6*uf1**4 - 4*uf2**2 + 6*uf2**4 + 4*uf1**2 *
                (-1 + 4*uf2**2))) / (3.*(-1 + ua**2)**2) -
              (2*gpff1**2*gpff2**2 *
               (-1 + 4*uf1**4 - 8*uf1**2*uf2**2 + 4*uf2**4)) /
              (-1 + up**2)**2 - (16*gaff1*gaff2*gpff1*gpff2*uf1*uf2) /
              ((-1 + ua**2)*(-1 + up**2)) +
              (2*gsff1**2*gsff2**2*(-1 + 2*uf1**2 + 2*uf2**2)**2) /
              (-1 + us**2)**2 -
              (8*gvff1**2*gvff2**2 *
               (-1 + 6*uf1**4 - 2*uf2**2 + 6*uf2**4 -
                2*uf1**2*(1 + 8*uf2**2)))/(3.*(-1 + uv**2)**2))) /
            (32.*np.pi*Q**2*np.sqrt(1 - 4*uf1**2)))
