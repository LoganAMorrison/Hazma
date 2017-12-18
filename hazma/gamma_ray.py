from .gamma_ray_helper_functions.gamma_ray_generator import gamma, gamma_point


def gamma_ray(particles, cme, eng_gams,
              mat_elem_sqrd=lambda k_list: 1.0,
              num_ps_pts=1000, num_bins=25):

    if hasattr(eng_gams, '__len__'):
        return gamma(particles, cme, eng_gams, mat_elem_sqrd,
                     num_ps_pts, num_bins)
    return gamma_point(particles, cme, eng_gams, mat_elem_sqrd,
                       num_ps_pts, num_bins)
