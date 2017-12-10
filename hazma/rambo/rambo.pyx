import numpy as np
cimport numpy as np
from scipy.optimize import newton
from functools import partial
from libc.math cimport log, M_PI, sqrt, tgamma
import cython

ctypedef np.float64_t DBL_T
ctypedef np.int_t INT_T

DBL = np.float64
INT = np.int

cdef class Rambo:

    cdef INT_T __num_fsp, __num_phase_space_pts, __event_count
    cdef DBL_T __cme
    cdef DBL_T[:] __masses, __weight_array, __randoms
    cdef DBL_T[:, :] __q_list, __p_list, __k_list
    cdef DBL_T[:, :, :] __phase_space_array
    cdef DBL_T __weight


    def __init__(self):
        pass

    def __cinit__(self):
        pass

    def __dealloc__(self):
        pass


    cdef __initilize(self, INT_T num_phase_space_pts, \
                     DBL_T[:] masses, DBL_T cme):
        np.random.seed()
        self.__num_fsp = len(masses)
        self.__masses = np.array(masses, dtype=DBL)
        self.__num_phase_space_pts = num_phase_space_pts
        self.__cme = cme

        self.__weight_array = np.zeros(num_phase_space_pts, dtype=DBL)
        self.__phase_space_array = np.zeros(
            (num_phase_space_pts, len(masses), 4), dtype=DBL)

        self.__event_count = 0
        self.__q_list = np.zeros((len(masses), 4), dtype=DBL)
        self.__p_list = np.zeros((len(masses), 4), dtype=DBL)
        self.__k_list = np.zeros((len(masses), 4), dtype=DBL)
        self.__weight = 1.0
        self.__randoms = np.random.random(num_phase_space_pts * len(masses) * 4)

    # ROOT FINDING
    @cython.cdivision(True)
    cdef DBL_T func_xi(self, DBL_T xi):
        cdef INT_T i
        cdef DBL_T val = 0.0

        # print('num_fsp = {}'.format(self.__num_fsp))

        for i in range(self.__num_fsp):
            val += sqrt(self.__masses[i]**2 \
                           + xi**2 * self.__p_list[i, 0]**2)
            # print('__masses[{}] = {}'.format(i, self.__masses[i]))
            # print('__p_list[{}] = {}'.format(i, self.__p_list[i, 0]))


        # print('val - cme = {}'.format(val - self.__cme))
        return val - self.__cme


    @cython.cdivision(True)
    cdef DBL_T deriv_func_xi(self, DBL_T xi):
        cdef INT_T i
        cdef DBL_T val = 0.0
        cdef DBL_T denom

        for i in range(self.__num_fsp):
            denom = sqrt(self.__masses[i]**2 \
                            + xi**2 * self.__p_list[i, 0]**2)
            val += xi * self.__p_list[i, 0]**2 / denom
        return val


    @cython.cdivision(True)
    cdef DBL_T __find_root(self):
        cdef DBL_T mass_sum = 0.0
        cdef INT_T i = 0

        for i in range(self.__num_fsp):
            mass_sum += self.__masses[i]

        f = partial(self.func_xi, self)
        df = partial(self.deriv_func_xi, self)

        xi_0 = sqrt(1.0 - (mass_sum / self.__cme)**2)

        return newton(f, xi_0, df)


    # CONVINIENCE FUNCTIONS
    def __get_mass(self, fv):
        return sqrt(fv[0]**2 - fv[1]**2 - fv[2]**2 - fv[3]**2)


    # FV GENERATORS
    @cython.cdivision(True)
    def __generate_qs(self):
        """
        Generate four-momenta Q_i.

        Return:
            Returns a numpy array containing self.__num_fsp four-momenta.

        Details:
            These four-momenta are isotropic and have energies which are
            distributed according to:
                q_0 * exp(-q_0)
        """

        cdef INT_T i
        cdef DBL_T rho_1, rho_2, rho_3, rho_4
        cdef DBL_T c, phi
        cdef DBL_T q_e, q_x, q_y, q_z

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
    def __generate_ps(self):
        """
        Generates P's from Q's.

        Arguments:
            q_list: A list of the Q four-momenta.

        Returns:
            Returns a numpy array containing self.__num_fsp four-momenta.

        Details:
            The Q's are transformed into P's, which, when summed, have the
            correct center of mass energy self.__cme and zero total
            three-momenta.
        """

        cdef DBL_T mass_Q
        cdef DBL_T b_x, b_y, b_z
        cdef DBL_T x, gamma, a
        cdef DBL_T qi_e, qi_x, qi_y, qi_z
        cdef DBL_T b_dot_qi
        cdef DBL_T pi_e, pi_x, pi_y, pi_z

        # Create the Q FV. This is the sum of all the q's.
        q_list_sum = np.sum(self.__q_list, 0)

        # Transform q's into p's

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
    def __generate_ks(self):
        """
        Generates K's from P's.

        Arguments:
            p_list: A list of the P four-momenta.

        Returns:
            Returns a numpy array containing self.__num_fsp four-momenta.

        Details:
            The P's are transformed into K's, which have the correct __masses.
        """
        cdef DBL_T xi
        cdef DBL_T k_e, k_x, k_y, k_z

        cdef DBL_T term1 = 0.0
        cdef DBL_T term2 = 0.0
        cdef DBL_T term3 = 1.0
        cdef DBL_T modulus

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


    def generate_phase_space(self, num_phase_space_pts, masses, cme):

        self.__initilize(num_phase_space_pts, masses, cme)
        self.__event_count = 0
        cdef INT_T i, j

        while self.__event_count < self.__num_phase_space_pts:
            # Transform p's = q's into the desired p's, which have the correct
            # center of mass energy and zero total three momenta. This function
            # also returns massless weight weight0.

            self.__generate_qs()
            self.__generate_ps()
            self.__generate_ks()

            self.__weight_array[self.__event_count] = self.__weight

            for i in range(self.__num_fsp):
                for j in range(4):
                    self.__phase_space_array[self.__event_count, i, j] \
                        = self.__k_list[i, j]

            self.__event_count += 1

        self.normalize_weights()

        return self.__phase_space_array, self.__weight_array


    cdef normalize_weights(self):
        cdef DBL_T weight_sum = 0.0
        cdef INT_T i

        for i in range(self.__num_phase_space_pts):
            weight_sum += self.__weight_array[i]

        for i in range(self.__num_phase_space_pts):
            self.__weight_array[i] = self.__weight_array[i] / weight_sum
