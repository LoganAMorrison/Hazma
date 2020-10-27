import numpy as np
from scipy.special import factorial


def pdk(Q, m1, m2):
    """
    Compute the magnitude of momentum for particles of mass `m1` and `m2` in
    center-of-mass frame with center-of-mass energy `Q`.
    """
    return np.sqrt(
        np.abs((Q - m1 - m2) * (Q + m1 + m2) * (Q - m1 + m2) * (Q + m1 - m2))
    ) / (2 * Q)


def boost(fv, beta):
    """
    Boost a four-vector along the boost vector.

    Parameters
    ----------
    fv: np.array
        Four-vector to boost.
    beta: np.array
        Boost vector.

    Returns
    -------
    fvp: np.array
        Boosted four-vector.
    """
    b = np.sqrt(beta[0] ** 2 + beta[1] ** 2 + beta[2] ** 2)
    if b == 0.0:
        return
    assert 0 <= beta and beta < 1, "Norm of boost vector must be less than 1."

    gam = 1.0 / np.sqrt(1.0 - beta ** 2)

    xp = fv[1:]
    tp = fv[0]

    x = xp + ((gam - 1) / b ** 2 * np.dot(beta, xp)) * beta + (gam * tp) * beta
    t = gam * (tp + np.dot(beta, xp))

    return np.array([t, x[0], x[1], x[2]])


class PhaseSpaceIntegrator:
    def __init__(self, masses, ecm, beta, msqrd):
        self.masses = np.array(masses)
        self.ecm = ecm
        self.beta = np.array(beta)
        self.msqrd = msqrd

        self.nfsp = len(masses)
        self.mass_sum = np.sum(masses)

        self.events = []

    def generate(self, rnds):
        """
        Fill four-momenta and compute weight of a monte-carlo phase-space
        event given final-state particle masses, center-of-mass energy,
        boost-vector and random-numbers produced by PyCuba.
        """
        weight = 1.0

        # random numbers of intermediate particle masses
        rno = np.array([0.0 for _ in range(self.nfsp)])
        rno[-1] = 1.0
        if self.nfsp > 2:
            rno[1:-1] = rnds[:-2]
            rno = np.sort(rno)
            weight /= factorial(self.nfsp - 2)

        # calculate values of intermediate particle masses M[k]:
        #   M[k] = (√s - Σm) × x[k] + sum(M[i] for i in 1:j), k = 2:N-1
        # where Σm =  sum(m[i] for i in 1:N) and x[k] sorted random numbers such
        # that 0 <= x[k-1] <= x[k] <= 1 for k = 3:N-1. `invmass[k]==M[k]`.
        invmass = np.array([0.0 for _ in range(self.nfsp)])
        summ = 0.0
        for n in range(self.nfsp):
            summ += self.masses[n]
            invmass[n] = rno[n] * (self.ecm - self.mass_sum) + summ

            if n == 0 or n == self.nfsp - 1:
                continue

            weight *= 2 * invmass[n] * (self.ecm - self.mass_sum)

        # Compute weight
        wt = 1.0
        pd = np.array([0.0 for _ in range(self.nfsp - 1)])
        for n in range(self.nfsp - 1):
            pd[n] = pdk(invmass[n + 1], invmass[n], self.masses[n + 1])
            wt *= pd[n] * np.pi / (invmass[n + 1] * (2.0 * np.pi) ** 3)
        wt *= weight

        momenta = [np.zeros(4) for _ in range(self.nfsp)]

        # boost
        momenta[0][0] = np.sqrt(pd[0] ** 2 + self.masses[0] ** 2)
        momenta[0][1] = 0.0
        momenta[0][2] = 0.0
        momenta[0][3] = pd[0]

        i = 1
        while True:
            momenta[i][0] = np.sqrt(pd[i - 1] ** 2 + self.masses[i] ** 2)
            momenta[i][1] = 0.0
            momenta[i][2] = 0.0
            momenta[i][3] = -pd[i - 1]

            # start grabing random nums from nfsp-1 (since we already used
            # rnds[1:nfsp-2] )
            cY = 2 * rnds[self.nfsp - 1 + 2 * (i - 2) + 0] - 1
            sY = np.sqrt(1 - cY ** 2)
            angZ = 2.0 * np.pi * rnds[self.nfsp - 1 + 2 * (i - 2) + 1]
            cZ = np.cos(angZ)
            sZ = np.sin(angZ)

            for j in range(i):
                x = momenta[j][1]
                z = momenta[j][3]
                # rotation around Y
                momenta[j][3] = cY * z - sY * x
                momenta[j][1] = sY * z + cY * x
                x = momenta[j][1]
                # rotation around Z
                y = momenta[j][2]
                momenta[j][1] = cZ * x - sZ * y
                momenta[j][2] = sZ * x + cZ * y

            if i == self.nfsp - 1:
                break

            beta = pd[i] / np.sqrt(pd[i] ** 2 + invmass[i] ** 2)
            for j in range(i):
                momenta[j] = boost(momenta[j], np.array([0.0, 0.0, beta]))
            i += 1

        # final boost of all particles
        for n in range(self.nfsp):
            momenta[n] = boost(momenta[n], beta)

        # store the weight and event
        self.events.append({"momenta": momenta, "weight": wt})

        return wt

    def integrand(self, ndim, x, ncomp, f):
        self.generate(x)
        f[0] = (
            2.0
            * np.pi
            * self.msqrd(self.events[-1]["momenta"])
            * self.events[-1]["weight"]
        )
        return 0
