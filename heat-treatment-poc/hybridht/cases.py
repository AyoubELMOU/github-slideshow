"""Verification (V*) and validation (P*) cases: FreeFEM vs analytical / published references.

Each case returns a dict with the arrays to plot and scalar metrics.
"""
import numpy as np

from . import analytical as an
from . import freefem as ff
from . import materials as mat

HOUR = 3600.0


def _relerr(num, ref, scale):
    return float(np.max(np.abs(num - ref)) / scale)


# ---------------------------------------------------------------- verification

def v1_diffusion_dirichlet(nx=400, dt=10.0):
    """Carburizing, constant D, fixed surface carbon -> erfc."""
    p = dict(L=5e-3, nx=nx, dt=dt, tend=4 * HOUR, c0=0.2, cs=1.0, beta=-1, dmodel="const", D=1.5e-11)
    x, c = ff.final_profile(ff.run("diffusion_1d.edp", **p))
    ref = an.diffusion_dirichlet(x, p["tend"], p["D"], p["c0"], p["cs"])
    return dict(name="V1 diffusion, Dirichlet (erfc)", params=p, x=x, num=c, ref=ref,
                err=_relerr(c, ref, p["cs"] - p["c0"]))


def v2_diffusion_robin(nx=400, dt=10.0, tend=4 * HOUR):
    """Carburizing, constant D, mass-transfer surface condition -> Crank/Carslaw-Jaeger."""
    p = dict(L=5e-3, nx=nx, dt=dt, tend=tend, c0=0.2, cs=1.0, beta=2e-7, dmodel="const", D=1.5e-11)
    x, c = ff.final_profile(ff.run("diffusion_1d.edp", **p))
    ref = an.diffusion_robin(x, p["tend"], p["D"], p["beta"], p["c0"], p["cs"])
    return dict(name="V2 diffusion, Robin (mass transfer)", params=p, x=x, num=c, ref=ref,
                err=_relerr(c, ref, p["cs"] - p["c0"]))


def v3_eddy_current(regime="below_curie", nx=200):
    """Induction: harmonic eddy current in a slab -> sinh solution, skin depth."""
    mats = {
        # Representative orders of magnitude for a low-alloy steel; not grade-specific data.
        "below_curie": dict(sigma=5e6, mur=100.0),
        "above_curie": dict(sigma=1e6, mur=1.0),
    }[regime]
    f = 10e3
    delta = an.skin_depth(f, mats["sigma"], mats["mur"])
    p = dict(L=10 * delta, nx=nx, freq=f, **mats)
    r = ff.run("eddy_current_1d.edp", **p)
    A_ref, p_ref = an.eddy_current_slab(r["x"], p["L"], f, mats["sigma"], mats["mur"])
    # Skin depth recovered from the decay of the Joule power over the first 3 delta.
    m = (r["x"] > 0.2 * delta) & (r["x"] < 3 * delta)
    slope = np.polyfit(r["x"][m], np.log(r["p"][m]), 1)[0]
    delta_num = -2.0 / slope
    return dict(name=f"V3 eddy current ({regime.replace('_', ' ')})", params=p, x=r["x"],
                num=r["p"], ref=p_ref, delta=delta, delta_num=delta_num,
                err=_relerr(r["Are"] + 1j * r["Aim"], A_ref, 1.0),
                delta_err=abs(delta_num - delta) / delta)


def v4_quench_convective(nx=600, dt=0.01):
    """Quench: semi-infinite steel, convective cooling -> same Robin solution for T."""
    k, rho, cp, h = 25.0, 7800.0, 600.0, 5000.0  # representative steel / oil-quench values
    alpha = k / (rho * cp)
    p = dict(L=0.06, nx=nx, dt=dt, tend=5.0, c0=850.0, cs=60.0, beta=h / (rho * cp),
             dmodel="const", D=alpha)
    x, T = ff.final_profile(ff.run("diffusion_1d.edp", **p))
    ref = an.diffusion_robin(x, p["tend"], alpha, p["beta"], p["c0"], p["cs"])
    return dict(name="V4 quench, convective surface", params=p, x=x, num=T, ref=ref,
                err=_relerr(T, ref, p["c0"] - p["cs"]))


def v5_convergence(levels=(25, 50, 100, 200, 400), tend=2 * HOUR):
    """Mesh/time refinement on V2 with dt ~ h^2 -> observed order of accuracy."""
    hs, errs = [], []
    for nx in levels:
        dt = 80.0 * (levels[0] / nx) ** 2
        r = v2_diffusion_robin(nx=nx, dt=dt, tend=tend)
        hs.append(r["params"]["L"] / nx)
        errs.append(r["err"])
    hs, errs = np.array(hs), np.array(errs)
    orders = np.log(errs[:-1] / errs[1:]) / np.log(hs[:-1] / hs[1:])
    return dict(name="V5 convergence (V2, dt ~ h^2)", h=hs, err=errs, orders=orders)


# ---------------------------------------------------------------- validation

def p1_harris_case_depth(times_h=(2, 4, 8, 16), T_C=927.0, beta=2e-7, c0=0.2, cp=1.0,
                         nx=300, dt=60.0, total_case_excess=0.05):
    """Nonlinear carburizing (Tibbetts D(C,T) + mass transfer) vs Harris (1943) correlation.

    Harris' correlation was fitted to measured case depths; it is a *total* case depth,
    taken here as the depth where carbon falls to c0 + `total_case_excess` wt%.
    beta (surface mass-transfer coefficient) is an assumed, typical endogas value;
    identifying it from measured profiles is the job of the inverse PINN step.
    """
    T_K = T_C + 273.15
    # One run to the longest time, with snapshots every `step_h` hours.
    step_h = float(np.gcd.reduce([int(t) for t in times_h]))
    t_end = max(times_h)
    prm = dict(L=6e-3, nx=nx, dt=dt, tend=t_end * HOUR, c0=c0, cs=cp, beta=beta,
               dmodel="tibbetts", T=T_K, picard=2, nout=int(round(t_end / step_h)))
    res = ff.run("diffusion_1d.edp", **prm)
    rows, profiles = [], []
    for th in times_h:
        m = np.isclose(res["t"], th * HOUR)
        x, c = res["x"][m], res["c"][m]
        d_tot = ff.depth_at(x, c, c0 + total_case_excess) * 1e3
        d_04 = ff.depth_at(x, c, 0.4) * 1e3
        d_h = float(mat.harris_case_depth_mm(th, T_C))
        rows.append(dict(t_h=th, surface_c=float(c[0]), depth_total_mm=d_tot,
                         depth_0p4_mm=d_04, harris_mm=d_h, rel_diff=(d_tot - d_h) / d_h))
        profiles.append((th, x, c))
    t = np.array([r["t_h"] for r in rows], float)
    d = np.array([r["depth_total_mm"] for r in rows])
    exponent = np.polyfit(np.log(t), np.log(d), 1)[0]
    return dict(name="P1 case depth vs Harris (1943)", rows=rows, profiles=profiles,
                sqrt_t_exponent=float(exponent), T_C=T_C, beta=beta, c0=c0, cp=cp)
