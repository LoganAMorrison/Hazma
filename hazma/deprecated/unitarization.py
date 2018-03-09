def amp_bethe_salpeter_kk_to_kk(cme):
    """
    Returns the unitarized matrix element for kk -> kk in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I = 0 channel.

    Parameters
    ----------
    Q: float
        Invariant mass of the pions.

    Returns
    -------
    Mu: float
        Unitarized matrix element for kk -> kk in the zero isospin
        channel.
    """

    s = cme**2 + 1j * eps
    return (-24 * np.pi**2 * s *
            (-((2 * mpi**2 - 3 * s) *
               np.sqrt(4 * mpi**2 - s) *
               np.arctan((q_max * np.sqrt(s)) /
                         np.sqrt((mpi**2 + q_max**2) * (4 * mpi**2 - s)))) +
             np.sqrt(s) * (32 * fpi**2 * np.pi**2 +
                           (-2 * mpi**2 + 3 * s) *
                           np.log(mpi / (q_max + np.sqrt(mpi**2 +
                                                         q_max**2)))))) / \
        (-(np.sqrt(4 * mpi**2 - s) *
           np.arctan((q_max * np.sqrt(s)) /
                     np.sqrt((mpi**2 + q_max**2) * (4 * mpi**2 - s))) *
           (64 * fpi**2 * np.pi**2 * (mpi**2 - 2 * s) +
            3 * (2 * mpi**2 - 3 * s) * s *
            np.log(mk / (q_max + np.sqrt(mk**2 + q_max**2))))) +
         3 * np.sqrt(4 * mk**2 - s) * np.sqrt(s) *
         np.arctan((q_max * np.sqrt(s)) /
                   np.sqrt((mk**2 + q_max**2) * (4 * mk**2 - s))) *
         (-((2 * mpi**2 - 3 * s) * np.sqrt(4 * mpi**2 - s) *
            np.arctan((q_max * np.sqrt(s)) /
                      np.sqrt((mpi**2 + q_max**2) * (4 * mpi**2 - s)))) +
          np.sqrt(s) * (32 * fpi**2 * np.pi**2 +
                        (-2 * mpi**2 + 3 * s) *
                        np.log(mpi / (q_max + np.sqrt(mpi**2 + q_max**2))))) +
         np.sqrt(s) * (64 * fpi**2 * np.pi**2 *
                       (16 * fpi**2 * np.pi**2 -
                        (mpi**2 - 2 * s) *
                        np.log(mpi / (q_max + np.sqrt(mpi**2 + q_max**2)))) +
                       3 * s *
                       np.log(mk / (q_max + np.sqrt(mk**2 + q_max**2))) *
                       (32 * fpi**2 * np.pi**2 + (-2 * mpi**2 + 3 * s) *
                        np.log(mpi / (q_max + np.sqrt(mpi**2 + q_max**2))))))


def amp_bethe_salpeter_pipi_to_kk(cme):
    """
    Returns the unitarized matrix element for pipi -> kk in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I = 0 channel.

    Parameters
    ----------
    Q: float
        Invariant mass of the pions.

    Returns
    -------
    Mu: float
        Unitarized matrix element for pipi -> kk in the zero isospin
        channel.
    """

    s = cme**2 + 1j * eps
    return (-256 * np.sqrt(3) * fpi**2 * np.pi**4 * s**1.5) / \
        (-(np.sqrt(4 * mpi**2 - s) *
           np.arctan((q_max * np.sqrt(s)) /
                     np.sqrt((mpi**2 + q_max**2) * (4 * mpi**2 - s))) *
           (64 * fpi**2 * np.pi**2 * (mpi**2 - 2 * s) +
            3 * (2 * mpi**2 - 3 * s) * s *
            np.log(mk / (q_max + np.sqrt(mk**2 + q_max**2))))) +
         3 * np.sqrt(4 * mk**2 - s) * np.sqrt(s) *
         np.arctan((q_max * np.sqrt(s)) /
                   np.sqrt((mk**2 + q_max**2) * (4 * mk**2 - s))) *
         (-((2 * mpi**2 - 3 * s) * np.sqrt(4 * mpi**2 - s) *
            np.arctan((q_max * np.sqrt(s)) /
                      np.sqrt((mpi**2 + q_max**2) * (4 * mpi**2 - s)))) +
            np.sqrt(s) *
            (32 * fpi**2 * np.pi**2 +
             (-2 * mpi**2 + 3 * s) *
             np.log(mpi / (q_max + np.sqrt(mpi**2 + q_max**2))))) +
         np.sqrt(s) * (64 * fpi**2 * np.pi**2 *
                       (16 * fpi**2 * np.pi**2 -
                        (mpi**2 - 2 * s) *
                        np.log(mpi / (q_max + np.sqrt(mpi**2 + q_max**2)))) +
                       3 * s *
                       np.log(mk / (q_max + np.sqrt(mk**2 + q_max**2))) *
                       (32 * fpi**2 * np.pi**2 + (-2 * mpi**2 + 3 * s) *
                        np.log(mpi / (q_max + np.sqrt(mpi**2 + q_max**2))))))


def amp_bethe_salpeter_pipi_to_pipi(cme):
    """
    Returns the unitarized matrix element for pipi -> pipi in the zero isospin
    channel.

    This is computed by resumming an infinite number of bubble
    intermediate states. The intermediate states used were pions and kaons
    in the I = 0 channel.

    Parameters
    ----------
    Q: float
        Invariant mass of the pions.

    Returns
    -------
    Mu: float
        Unitarized matrix element for pipi -> pipi in the zero isospin
        channel.
    """

    s = cme**2 + 1j * eps
    return (8 * np.pi**2 * np.sqrt(s) *
            (64 * fpi**2 * np.pi**2 * (mPI**2 - 2 * s) -
             3 * np.sqrt(4 * mK**2 - s) *
             np.sqrt(s) * (-2 * mPI**2 + 3 * s) *
             np.arctan((q_max * np.sqrt(s)) /
                       np.sqrt((mK**2 + q_max**2) * (4 * mK**2 - s))) +
             3 * (2 * mPI**2 - 3 * s) * s *
             np.log(mK / (q_max + np.sqrt(mK**2 + q_max**2))))) /\
        (-(np.sqrt(4 * mPI**2 - s) *
           np.arctan((q_max * np.sqrt(s)) /
                     np.sqrt((mPI**2 + q_max**2) * (4 * mPI**2 - s))) *
           (64 * fpi**2 * np.pi**2 * (mPI**2 - 2 * s) +
            3 * (2 * mPI**2 - 3 * s) * s *
            np.log(mK / (q_max + np.sqrt(mK**2 + q_max**2))))) +
         3 * np.sqrt(4 * mK**2 - s) * np.sqrt(s) *
         np.arctan((q_max * np.sqrt(s)) /
                   np.sqrt((mK**2 + q_max**2) * (4 * mK**2 - s))) *
         (-((2 * mPI**2 - 3 * s) * np.sqrt(4 * mPI**2 - s) *
            np.arctan((q_max * np.sqrt(s)) /
                      np.sqrt((mPI**2 + q_max**2) * (4 * mPI**2 - s)))) +
            np.sqrt(s) * (32 * fpi**2 * np.pi**2 +
                          (-2 * mPI**2 + 3 * s) *
                          np.log(mPI / (q_max +
                                        np.sqrt(mPI**2 + q_max**2))))) +
         np.sqrt(s) * (64 * fpi**2 * np.pi**2 *
                       (16 * fpi**2 * np.pi**2 -
                        (mPI**2 - 2 * s) *
                        np.log(mPI / (q_max + np.sqrt(mPI**2 + q_max**2)))) +
                       3 * s *
                       np.log(mK / (q_max + np.sqrt(mK**2 + q_max**2))) *
                       (32 * fpi**2 * np.pi**2 + (-2 * mPI**2 + 3 * s) *
                        np.log(mPI / (q_max + np.sqrt(mPI**2 + q_max**2))))))
