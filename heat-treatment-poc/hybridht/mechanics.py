"""Quench and residual stresses in a plate heat-treated on both faces (layer model).

Kinematics: half thickness L, depth z in [0, L] (z = 0 surface, z = L mid-plane, symmetry).
In-plane strain is uniform through the thickness (plane sections stay plane, no bending by
symmetry); stress is equi-biaxial in-plane, zero normal to the surface:
    sigma(z, t) = E_b(z, t) (eps0(t) - eps_free - eps_p - eps_trip),   E_b = E / (1 - nu)
    equilibrium: integral_0^L sigma dz = 0   -> one scalar unknown eps0(t) per step.
Plasticity: equi-biaxial von Mises (sigma_eq = |sigma|), perfectly plastic with
temperature/phase-dependent yield; radial return.
TRIP (Leblond / Greenwood-Johnson): d eps_trip = K sigma (1 - f) df for martensite fraction f
(equi-biaxial form of 3/2 K s Phi'(f) df with Phi = f (2 - f)).
Free strain (thermal + transformation) from phase specific volumes:
    eps_free = (V(T, phases, C) / V_ref)^(1/3) - 1.

This is the standard 1D model for case-hardened plates; the 2D tooth needs the FreeFEM
thermo-elasto-plastic model (next milestone).

Specific volumes [cm^3/g], C in wt% (Moyer & Ansell, Metall. Trans. A 6 (1975) 1785, as tabulated
in heat-treatment simulation literature; TODO verify against the original before publication):
    austenite  : 0.12282 + 8.56e-6 (T - 20) + 2.15e-3 C
    martensite : 0.12708 + 4.448e-6 (T - 20) + 2.79e-3 C
    ferrite-carbide base : 0.12708 + 5.528e-6 (T - 20)
"""
import numpy as np
from scipy.optimize import brentq


def specific_volume(T, aus, mart, base, C):
    va = 0.12282 + 8.56e-6 * (T - 20.0) + 2.15e-3 * C
    vm = 0.12708 + 4.448e-6 * (T - 20.0) + 2.79e-3 * C
    vb = 0.12708 + 5.528e-6 * (T - 20.0)
    return aus * va + mart * vm + base * vb


def free_strain(T, ph):
    """(nt, nz) free linear strain relative to the initial (t = 0) state of each layer."""
    C = ph["c"][None, :]
    V = specific_volume(T, ph["aus"], ph["mart"], ph["base"], C)
    return np.cbrt(V / V[0:1, :]) - 1.0


# Representative mechanical properties [MPa]; assumptions to be replaced by measured data.
def youngs_modulus(T):
    return 210e3 * np.clip(1.0 - 4.5e-4 * T, 0.3, None)


def yield_strength(T, ph, sy_base_rt=700.0):
    Cm = np.minimum(ph["c"][None, :], 0.8)
    sy_a = np.maximum(250.0 - 0.25 * T, 20.0)
    sy_m = (1300.0 + 1100.0 * Cm) * np.clip(1.0 - T / 900.0, 0.05, None)
    sy_b = sy_base_rt * np.clip(1.0 - T / 1000.0, 0.03, None)
    return ph["aus"] * sy_a + ph["mart"] * sy_m + ph["base"] * sy_b


def trapezoid_weights(z):
    w = np.zeros_like(z)
    dz = np.diff(z)
    w[:-1] += dz / 2
    w[1:] += dz / 2
    return w


def solve_layers(z, eps_free, E, sy, mart=None, K_trip=5e-5, nu=0.3):
    """Incremental elasto-plastic layer solution. All field arrays are (nt, nz).

    Returns sigma (nt, nz), eps0 (nt,), eps_p and eps_trip at the end, and the equilibrium
    residual max |integral sigma dz| / (max sy * L) over all steps.
    """
    nt, nz = eps_free.shape
    w = trapezoid_weights(z)
    eps_p = np.zeros(nz)
    eps_tp = np.zeros(nz)
    sigma = np.zeros((nt, nz))
    eps0 = np.zeros(nt)
    res = 0.0
    s_prev = np.zeros(nz)
    for n in range(nt):
        Eb = E[n] / (1.0 - nu)
        if mart is not None and n > 0:
            df = np.maximum(mart[n] - mart[n - 1], 0.0)
            eps_tp = eps_tp + K_trip * s_prev * (1.0 - mart[n - 1]) * df
        base = eps_free[n] + eps_p + eps_tp

        def stress(e0):
            return np.clip(Eb * (e0 - base), -sy[n], sy[n])

        g = lambda e0: float(np.dot(w, stress(e0)))
        lo, hi = base.min() - 1e-12, base.max() + 1e-12
        if abs(g(lo)) < 1e-12 or abs(hi - lo) < 1e-15:
            e0 = lo
        else:
            e0 = brentq(g, lo, hi, xtol=1e-16, rtol=1e-14, maxiter=200)
        s_tr = Eb * (e0 - base)
        s = np.clip(s_tr, -sy[n], sy[n])
        eps_p = eps_p + (s_tr - s) / Eb
        sigma[n], eps0[n], s_prev = s, e0, s
        res = max(res, abs(np.dot(w, s)) / (sy[n].max() * z[-1]))
    return dict(sigma=sigma, eps0=eps0, eps_p=eps_p, eps_trip=eps_tp, equilibrium_residual=res)
