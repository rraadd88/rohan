import numpy as np

def get_intersection_of_gaussians(m1, s1, m2, s2):
    # coefficients of quadratic equation ax^2 + bx + c = 0
    a = (s1**2.0) - (s2**2.0)
    b = 2 * (m1 * s2**2.0 - m2 * s1**2.0)
    c = m2**2.0 * s1**2.0 - m1**2.0 * s2**2.0 - 2 * s1**2.0 * s2**2.0 * np.log(s1/s2)
    x1 = (-b + np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    x2 = (-b - np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    return x1, x2