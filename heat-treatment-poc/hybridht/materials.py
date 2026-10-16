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


def nitrogen_diffusivity(T_K):
    """Nitrogen diffusivity in austenite [m^2/s]: D = 0.91 exp(-168600 / (R T)) cm^2/s.

    Grieveson & Turkdogan, Trans. Metall. Soc. AIME 230 (1964), as used in
    carbonitriding models (e.g. Slycke & Ericsson, J. Heat Treating, 1981).
    TODO: confirm constants and page reference against the original paper before publication.
    """
    return 0.91e-4 * np.exp(-168600.0 / (8.31446 * T_K))


def andrews_ac1_ac3(comp):
    """Equilibrium Ac1 and Ac3 [deg C] from composition (wt%), K. W. Andrews,
    "Empirical formulae for the calculation of some transformation temperatures",
    J. Iron Steel Inst. 203 (1965) 721.

    These are equilibrium (slow-heating) values; the rapid heating of induction hardening
    shifts the effective austenitization temperature upwards (see the Ac3 margin in P4).
    """
    g = lambda k: comp.get(k, 0.0)
    ac1 = 723 - 10.7 * g("Mn") - 16.9 * g("Ni") + 29.1 * g("Si") + 16.9 * g("Cr") + 290 * g("As") + 6.38 * g("W")
    ac3 = (910 - 203 * np.sqrt(g("C")) - 15.2 * g("Ni") + 44.7 * g("Si") + 104 * g("V") + 31.5 * g("Mo")
           + 13.1 * g("W") - 30 * g("Mn") - 11 * g("Cr") - 20 * g("Cu") + 700 * g("P") + 400 * g("Al")
           + 120 * g("As") + 400 * g("Ti"))
    return float(ac1), float(ac3)


# Nominal mid-range compositions (wt%) within the EN 10083-3 / EN 10084 ranges; not a heat analysis.
STEELS = {
    "42CrMo4": dict(C=0.42, Si=0.25, Mn=0.75, Cr=1.05, Mo=0.22),
    "20MnCr5": dict(C=0.20, Si=0.25, Mn=1.25, Cr=1.15),
}
