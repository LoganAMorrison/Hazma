"""
Module containing various utility function using throughout `hazma`.

Q < m1 + m2 + ... + mn or Q < m_a + ...

Q < m1 + m2 + ... + mn or Q < m_a + ...

"""
import functools

import numpy as np


def kin_check_sigma(q, mass_sum_in, mass_sum_out):
    return q > mass_sum_in and q > mass_sum_out


def kin_check_width(m, mass_sum_out):
    return m > mass_sum_out


def kinematic_check(mass_sum_in=np.inf, mass_sum_out=np.inf, energy_arg=1):
    """
    Returns a decorator that applies a kinematic check to a function. That is,
    if (center-of-mass energy < sum of in masses) or
    (center-of-mass energy < sum of in masses) then the result is zero.
    """
    def wrap(func):
        @functools.wraps(func)
        def wrapped_function(*args, **kwargs):
            if type(args[energy_arg]) in [np.ndarray, list]:
                args[energy_arg] = np.array(args[energy_arg])
                ret = np.zeros_like(args[energy_arg])
                mask = (args[energy_arg] < mass_sum_in) or (
                    args[energy_arg] < mass_sum_out)
                args[energy_arg] = args[energy_arg][mask]
                ret[mask] = func(*args, **kwargs)
            else:
                if args[energy_arg] < mass_sum_in or args[energy_arg] < mass_sum_out:
                    return 0.0
                return func(*args, **kwargs)
        return wrapped_function
    return wrap
