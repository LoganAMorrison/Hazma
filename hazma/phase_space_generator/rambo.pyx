import numpy as np
cimport numpy as np
from scipy.optimize import newton
from functools import partial
from libc.math cimport log, M_PI, sqrt, tgamma
import cython

# ctypedef np.float np.float
# ctypedef np.np.int np.int

# np.float = np.float
# np.int = np.np.int

ctypedef double (*f_type)(np.ndarray)

cdef class Rambo:

    def __init__(self):
        pass

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass


    cdef __initilize(self, np.int num_phase_space_pts, \
                     np.float64_t[:] masses, np.float64_t cme):
        np.random.seed()
        self.__num_fsp = len(masses)
        self.__masses = np.array(masses, dtype=np.float64)
        self.__num_phase_space_pts = num_phase_space_pts
        self.__cme = cme

        self.__weight_array = np.zeros(num_phase_space_pts, dtype=np.float64)
        self.__phase_space_array = np.zeros(
            (num_phase_space_pts, len(masses), 4), dtype=np.float64)

        self.__event_count = 0
        self.__q_list = np.zeros((len(masses), 4), dtype=np.float64)
        self.__p_list = np.zeros((len(masses), 4), dtype=np.float64)
        self.__k_list = np.zeros((len(masses), 4), dtype=np.float64)
        self.__weight = 1.0
        self.__randoms = np.random.random(num_phase_space_pts * len(masses) * 4)

    # ROOT FINDING
    @cython.cdivision(True)
    cdef func_xi(self, xi):
        cdef np.int i
        cdef np.float64_t val = 0.0

        # prnp.int('num_fsp = {}'.format(self.__num_fsp))

        for i in range(self.__num_fsp):
            val += sqrt(self.__masses[i]**2 \
                           + xi**2 * self.__p_list[i, 0]**2)
            # prnp.int('__masses[{}] = {}'.format(i, self.__masses[i]))
            # prnp.int('__p_list[{}] = {}'.format(i, self.__p_list[i, 0]))


        # prnp.int('val - cme = {}'.format(val - self.__cme))
        return val - self.__cme


    @cython.cdivision(True)
    cdef deriv_func_xi(self, xi):
        cdef np.int i
        cdef np.float64_t val = 0.0
        cdef np.float64_t denom

        for i in range(self.__num_fsp):
            denom = sqrt(self.__masses[i]**2 \
                            + xi**2 * self.__p_list[i, 0]**2)
            val += xi * self.__p_list[i, 0]**2 / denom
        return val


    @cython.cdivision(True)
    cdef np.float64_t __find_root(self):
        cdef np.float64_t mass_sum = 0.0
        cdef np.int i = 0

        for i in range(self.__num_fsp):
            mass_sum += self.__masses[i]

        f = partial(self.func_xi, self)
        df = partial(self.deriv_func_xi, self)

        xi_0 = sqrt(1.0 - (mass_sum / self.__cme)**2)

        return newton(f, xi_0, df)


    # CONVINIENCE FUNCTIONS
    cdef __get_mass(self, fv):
        return sqrt(fv[0]**2 - fv[1]**2 - fv[2]**2 - fv[3]**2)


    # FV GENERATORS
    @cython.cdivision(True)
    cdef __generate_qs(self):
        """
        Generate four-momenta Q_i.

        Return:
            Returns a numpy array containing self.__num_fsp four-momenta.

        Details:
            These four-momenta are isotropic and have energies which are
            distributed according to:
                q_0 * exp(-q_0)
        """

        cdef np.int i
        cdef np.float64_t rho_1, rho_2, rho_3, rho_4
        cdef np.float64_t c, phi
        cdef np.float64_t q_e, q_x, q_y, q_z

        for i in range(self.__num_fsp):
            rho_1 = self.__randoms[self.__num_fsp * i + self.__event_count + 0]
            rho_2 = self.__randoms[self.__num_fsp * i + self.__event_count + 1]
            rho_3 = self.__randoms[self.__num_fsp * i + self.__event_count + 2]
            rho_4 = self.__randoms[self.__num_fsp * i + self.__event_count + 3]

            c = 2.0 * rho_1 - 1.0
            phi = 2.0 * M_PI * rho_2

            q_e = -log(rho_3 * rho_4)
            q_x = q_e * sqrt(1.0 - c**2.0) * np.cos(phi)
            q_y = q_e * sqrt(1.0 - c**2.0) * np.sin(phi)
            q_z = q_e * c

            self.__q_list[i][0] = q_e
            self.__q_list[i][1] = q_x
            self.__q_list[i][2] = q_y
            self.__q_list[i][3] = q_z


    @cython.cdivision(True)
    cdef __generate_ps(self):
        """
        Generates P's from Q's.

        Arguments:
            q_list: A list of the Q four-momenta.

        Returns:
            Returns a numpy array containing self.__num_fsp four-momenta.

        Details:
            The Q's are transformed np.into P's, which, when summed, have the
            correct center of mass energy self.__cme and zero total
            three-momenta.
        """

        cdef np.float64_t mass_Q
        cdef np.float64_t b_x, b_y, b_z
        cdef np.float64_t x, gamma, a
        cdef np.float64_t qi_e, qi_x, qi_y, qi_z
        cdef np.float64_t b_dot_qi
        cdef np.float64_t pi_e, pi_x, pi_y, pi_z

        # Create the Q FV. This is the sum of all the q's.
        q_list_sum = np.sum(self.__q_list, 0)

        # Transform q's np.into p's

        # Q mass
        mass_Q = self.__get_mass(q_list_sum)

        # Boost factor
        b_x = -q_list_sum[1] / mass_Q
        b_y = -q_list_sum[2] / mass_Q
        b_z = -q_list_sum[3] / mass_Q
        x = self.__cme / mass_Q
        gamma = q_list_sum[0] / mass_Q
        a = 1.0 / (1.0 + gamma)

        for i in range(self.__num_fsp):
            qi_e = self.__q_list[i, 0]
            qi_x = self.__q_list[i, 1]
            qi_y = self.__q_list[i, 2]
            qi_z = self.__q_list[i, 3]

            b_dot_qi = b_x * qi_x + b_y * qi_y + b_z * qi_z

            pi_e = x * (gamma * qi_e + b_dot_qi)
            pi_x = x * (qi_x + b_x * qi_e + a * b_dot_qi * b_x)
            pi_y = x * (qi_y + b_y * qi_e + a * b_dot_qi * b_y)
            pi_z = x * (qi_z + b_z * qi_e + a * b_dot_qi * b_z)

            self.__p_list[i][0] = pi_e
            self.__p_list[i][1] = pi_x
            self.__p_list[i][2] = pi_y
            self.__p_list[i][3] = pi_z


        self.__weight = (M_PI / 2.0)**(self.__num_fsp - 1.0) \
            * (self.__cme)**(2.0 * self.__num_fsp - 4.0) \
            / tgamma(self.__num_fsp) \
            / tgamma(self.__num_fsp - 1) \
            * (2.0 * M_PI)**(4.0 - 3.0 * self.__num_fsp)


    @cython.cdivision(True)
    cdef __generate_ks(self):
        """
        Generates K's from P's.

        Arguments:
            p_list: A list of the P four-momenta.

        Returns:
            Returns a numpy array containing self.__num_fsp four-momenta.

        Details:
            The P's are transformed np.into K's, which have the correct __masses.
        """
        cdef np.float64_t xi
        cdef np.float64_t k_e, k_x, k_y, k_z

        cdef np.float64_t term1 = 0.0
        cdef np.float64_t term2 = 0.0
        cdef np.float64_t term3 = 1.0
        cdef np.float64_t modulus

        # Find the scaling factor xi.
        xi = self.__find_root()


        for i in range(self.__num_fsp):
            k_e = sqrt(self.__masses[i]**2 + xi**2 * self.__p_list[i, 0]**2)
            k_x = xi * self.__p_list[i, 1]
            k_y = xi * self.__p_list[i, 2]
            k_z = xi * self.__p_list[i, 3]

            self.__k_list[i][0] = k_e
            self.__k_list[i][1] = k_x
            self.__k_list[i][2] = k_y
            self.__k_list[i][3] = k_z

            modulus = sqrt(k_x**2.0 + k_y**2.0 + k_z**2.0)

            term1 += modulus / self.__cme
            term2 += modulus**2 / self.__k_list[i, 0]
            term3 = term3 * modulus / self.__k_list[i, 0]

        term1 = term1**(2.0 * self.__num_fsp - 3.0)
        term2 = term2**(-1.0)

        self.__weight = self.__weight * term1 * term2 * term3 * self.__cme

    cdef __normalize_weights(self):
        """
        Normalizes the 'self.__weight_array' so that the sum of all the entries is unity.
        """
        cdef np.float64_t weight_sum = 0.0
        cdef np.int i

        for i in range(self.__num_phase_space_pts):
            weight_sum += self.__weight_array[i]

        for i in range(self.__num_phase_space_pts):
            self.__weight_array[i] = self.__weight_array[i] / weight_sum

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def generate_phase_space(self, int num_phase_space_pts, \
                             np.ndarray masses, double cme,
                             mat_elem_sqrd=lambda klist: 1, normalize=True):
        """
        Creates 'num_phase_space_pts' number of phase space ponp.ints with final state particles with masses 'masses' and center of mass energy 'cme'.

        Arguments:
            num_phase_space_pts: Number of phase space ponp.ints to generate.

            masses: List of the final state particle masses.

            cme: Center of mass energy of the process.

            mat_elem_sqrd : Function matrix element squared. Arguments are
            a list of the four momenta.

        Returns:
            phase_space_array (np.ndarray): Array containing the four momenta of each particles for each event. The shape is (num_phase_space_pts, len(masses), 4)

            wieght_array (np.ndarray): Array containing the weight of each event. The shape is (num_phase_space_pts)


        """
        self.__initilize(num_phase_space_pts, masses, cme)
        self.__event_count = 0
        cdef np.int i, j

        while self.__event_count < self.__num_phase_space_pts:
            # Transform p's = q's np.into the desired p's, which have the correct
            # center of mass energy and zero total three momenta. This function
            # also returns massless weight weight0.

            self.__generate_qs()
            self.__generate_ps()
            self.__generate_ks()

            self.__weight_array[self.__event_count] \
                = self.__weight * mat_elem_sqrd(self.__k_list)

            for i in range(self.__num_fsp):
                for j in range(4):
                    self.__phase_space_array[self.__event_count, i, j] \
                        = self.__k_list[i, j]

            self.__event_count += 1

        if normalize == True:
            self.__normalize_weights()

        return self.__phase_space_array, self.__weight_array


    def generate_energy_histogram(self, int num_phase_space_pts, \
                                  np.ndarray masses, double cme, \
                                  int num_bins, mat_elem_sqrd=lambda klist: 1,
                                  logscale=False, normalize=True):
        """
        Returns the energy probability distributions for each particle. These
        distributions are stored in (num_particles, 2, num_bins) array called probs. For example, probs[1, 0, :], probs[1, 1, :] are the energies and their probabilities, respectively, for the second particle.

        Arguments:
            num_phase_space_pts: Number of phase space ponp.ints to generate.

            masses: List of the final state particle masses.

            cme: Center of mass energy of the process.

            num_bins: Number of bins to use for the probability distributions.

        Returns:
            probs (np.ndarray): Array containing the energy probability distributions for each final state particle. The shape is (len(masses), 2, num_bins)
        """
        self.generate_phase_space(num_phase_space_pts, masses, cme,
                                  mat_elem_sqrd, normalize)

        cdef np.int i, j
        cdef np.ndarray energy_array = np.zeros((num_phase_space_pts, \
            len(masses)), dtype=float)
        cdef np.ndarray hist = np.zeros((len(masses), num_bins),\
            dtype=float)
        cdef np.ndarray bins = np.zeros((len(masses), num_bins + 1),\
            dtype=float)
        cdef np.ndarray engs = np.zeros((len(masses), num_bins),\
            dtype=float)
        cdef np.ndarray probs = np.zeros((len(masses), 2, num_bins),\
            dtype=float)


        for i in range(num_phase_space_pts):
            for j in range(len(masses)):
                energy_array[i, j] = self.__phase_space_array[i, j, 0]

        if logscale == True:
            for i in range(len(masses)):
                eng_min = np.min(energy_array[i, :])
                eng_max = np.max(energy_array[i, :])
                engs = np.logspace(np.log10(eng_min), np.log10(eng_max),
                                   num=num_bins+1)
                hist[i, :], bins[i, :] \
                    = np.histogram(energy_array[:, i], bins=engs,
                                   weights=self.__weight_array[:])
        if logscale == False:
            for i in range(len(masses)):
                hist[i, :], bins[i, :] = np.histogram(energy_array[:, i], bins=num_bins, weights=self.__weight_array[:])

        engs = (bins[:, :-1] + bins[:, 1:]) / 2

        for i in range(len(masses)):
            for j in range(num_bins):
                probs[i, 0, j] = engs[i, j]
                probs[i, 1, j] = hist[i, j]

        return probs
