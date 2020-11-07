class TargetParams:
    """
    Container for information about a target region.

    Parameters
    ----------
    J : float
        (Averaged) J-factor for DM annihilation in MeV^2 cm^-5.
    D : float
        (Averaged) D-factor for DM decay in MeV cm^-2.
    dOmega : float
        Angular size in sr.
    vx : float
        Average DM velocity in target in units of c. Defaults to 1e-3, the
        Milky Way velocity dispersion.
    """

    def __init__(self, J=None, D=None, dOmega=None, vx=1e-3):
        self.J = J
        self.D = D
        self.dOmega = dOmega
        self.vx = vx
