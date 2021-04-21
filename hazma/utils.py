"""
Module containing various utility function using throughout `hazma`.

Q < m1 + m2 + ... + mn or Q < m_a + ...

Q < m1 + m2 + ... + mn or Q < m_a + ...

"""
import functools


def kinematic_check(q, mass_sum_in, mass_sum_out):
    """
    Returns a decorator that applies a kinematic check to a function. That is,
    if (center-of-mass energy < sum of in masses) or
    (center-of-mass energy < sum of in masses) then the result is zero.
    """
    def wrap(func):
        @functools.wraps(func)
        def wrapped_function(*args, **kwargs):
            if q < mass_sum_in or q < mass_sum_out:
                return 0  # should return something that matches return of func
            else:
                return func(*args, **kwargs)
