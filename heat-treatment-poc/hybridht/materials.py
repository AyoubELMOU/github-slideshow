"""Published material models and empirical correlations (with sources).

Every function cites its source; do not add a model here without one.
"""
import numpy as np

R_CAL = 1.98720  # cal / (mol K)


def tibbetts_diffusivity(c_wt, T_K):
    """Carbon diffusivity in austenite [m^2/s].

    G. G. Tibbetts, "Diffusivity of carbon in iron and steels at high temperatures",
    J. Appl. Phys. 51 (1980) 4813:
        D = 0.47 exp(-1.6 C) exp(-(37000 - 6600 C) / (R T))  cm^2/s,  C in wt%.
    """
    c = np.maximum(np.asarray(c_wt, dtype=float), 0.0)
    return 0.47e-4 * np.exp(-1.6 * c) * np.exp(-(37000.0 - 6600.0 * c) / (R_CAL * T_K))


def harris_case_depth_mm(t_hours, T_C):
    """Harris empirical carburizing case depth [mm].

    F. E. Harris, "Case depth - an attempt at a practical definition",
    Metal Progress 44 (1943) 265:
        d [in] = 31.6 sqrt(t) / 10^(6700 / T),  t in hours, T in degrees Rankine.
    At 927 C (1700 F) this reduces to the familiar d = 0.025 sqrt(t) in = 0.635 sqrt(t) mm.
    Harris' depth is a *total* case depth (carbon approaching the core level).
    """
    T_R = (T_C * 9.0 / 5.0 + 32.0) + 459.67
    return 25.4 * 31.6 * np.sqrt(t_hours) / 10.0 ** (6700.0 / T_R)
