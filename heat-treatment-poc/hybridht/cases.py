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


# ================================================================ milestone 2
# Multi-stage carbonitriding and coupled induction heating.

def _cn(stages, outputs=("out", "mass"), **kw):
    prm = dict(L=6e-3, nx=300, dt=60.0, C0=0.2, N0=0.0, dmodel="tibbetts", picard=2, nout=1)
    prm.update(kw)
    return ff.run_multi("carbonitriding_1d.edp", outputs, files={"stages": ff.stages_text(stages)}, **prm)


def _last(res, key):
    m = res["t"] == res["t"].max()
    return res["x"][m], res[key][m]


def v6_stage_split():
    """Splitting one stage into two identical halves must not change the result."""
    s = dict(T_C=930.0, C_pot=1.0, beta_C=2e-7, N_pot=0.3, beta_N=1e-7)
    a = _cn([dict(hours=4, **s)])["out"]
    b = _cn([dict(hours=2, **s), dict(hours=2, **s)])["out"]
    err = max(np.max(np.abs(_last(a, k)[1] - _last(b, k)[1])) for k in ("C", "N"))
    return dict(name="V6 stage split invariance", err=float(err))


def v7_two_species_analytical():
    """C and N together, constant D: each species must follow its own Robin solution."""
    DC, DN, bC, bN = 1.5e-11, 0.8e-11, 2e-7, 1e-7
    st = [dict(hours=4, T_C=900.0, C_pot=1.0, beta_C=bC, N_pot=0.4, beta_N=bN)]
    r = _cn(st, dmodel="const", DC=DC, DN=DN, L=5e-3, nx=400, dt=10.0)["out"]
    x, C = _last(r, "C")
    _, N = _last(r, "N")
    refC = an.diffusion_robin(x, 4 * HOUR, DC, bC, 0.2, 1.0)
    refN = an.diffusion_robin(x, 4 * HOUR, DN, bN, 0.0, 0.4)
    return dict(name="V7 C + N co-diffusion vs analytical", x=x, C=C, N=N, refC=refC, refN=refN,
                err=float(max(np.max(np.abs(C - refC)) / 0.8, np.max(np.abs(N - refN)) / 0.4)))


CN_CYCLE = [  # illustrative boost/diffuse carbonitriding cycle; potentials and beta_N are assumptions
    dict(hours=3.0, T_C=860.0, C_pot=1.0, beta_C=2e-7, N_pot=0.35, beta_N=1e-7),
    dict(hours=1.0, T_C=860.0, C_pot=0.8, beta_C=2e-7, N_pot=0.25, beta_N=1e-7),
]


def v8_conservation(stages=CN_CYCLE):
    """Absorbed amount from the surface flux = increase of the integral of the profile."""
    m = _cn(stages, nout=8)["mass"]
    errC = np.max(np.abs(m["inC"] - m["massC"])) / np.max(np.abs(m["massC"]))
    errN = np.max(np.abs(m["inN"] - m["massN"])) / np.max(np.abs(m["massN"]))
    return dict(name="V8 C + N mass conservation (boost/diffuse)", err=float(max(errC, errN)))


def v9_cross_code():
    """carbonitriding_1d.edp (C only) must reproduce diffusion_1d.edp (Tibbetts, Robin)."""
    T_K = 927.0 + 273.15
    a = ff.run("diffusion_1d.edp", L=6e-3, nx=300, dt=60.0, tend=4 * HOUR, c0=0.2, cs=1.0, beta=2e-7,
               dmodel="tibbetts", T=T_K, picard=2)
    b = _cn([dict(hours=4, T_C=927.0, C_pot=1.0, beta_C=2e-7)])["out"]
    return dict(name="V9 cross-code consistency", err=float(np.max(np.abs(ff.final_profile(a)[1] - _last(b, "C")[1]))))


IH_DEFAULT = dict(L=0.02, nx=200, freq=30e3, H0=1.5e5, mur0=20.0, Tc=760.0, dTc=15.0, sigma20=4.5e6,
                  aR=5.1e-3, k=30.0, rhocp=4.0e6, T0=20.0, Tamb=20.0, hheat=0.0, hq=2e4)


def _ih(**kw):
    prm = dict(IH_DEFAULT, theat=3.0, tend=3.0, dt=5e-3, nout=6)
    prm.update(kw)
    return ff.run_multi("induction_heating_1d.edp", ("out", "energy"), **prm)


def v10_em_surface_power():
    """Induction power per unit area with imposed surface field: P = H0^2 omega mu delta / 4."""
    out = {}
    for regime, T0 in (("cold", 20.0), ("hot", 900.0)):
        e = _ih(T0=T0, Tamb=T0, tend=1e-3, dt=1e-3, theat=1.0, nx=400)["energy"]
        mur = 1 + (IH_DEFAULT["mur0"] - 1) * 0.5 * (1 - np.tanh((T0 - IH_DEFAULT["Tc"]) / IH_DEFAULT["dTc"]))
        sig = IH_DEFAULT["sigma20"] / (1 + IH_DEFAULT["aR"] * (T0 - 20))
        w = 2 * np.pi * IH_DEFAULT["freq"]
        P = IH_DEFAULT["H0"] ** 2 * w * an.MU0 * mur * an.skin_depth(IH_DEFAULT["freq"], sig, mur) / 4
        out[regime] = dict(P_num=float(e["Psurf"][0]), P_ref=float(P), err=float(abs(e["Psurf"][0] - P) / P))
    return dict(name="V10 induction surface power vs analytical", regimes=out,
                err=max(v["err"] for v in out.values()))


def v11_energy_conservation():
    """Coupled EM-thermal heating + quench: Joule energy in = stored + lost through the surface."""
    e = _ih(theat=1.5, tend=2.5, dt=5e-3)["energy"]
    bal = np.abs(e["Ejoule"] - e["Estored"] - e["Eloss"]) / e["Ejoule"].max()
    return dict(name="V11 coupled energy conservation", t=e["t"], energy=e, err=float(bal.max()))


def v12_time_convergence(dts=(0.02, 0.01, 0.005, 0.0025)):
    """Surface temperature at the end of 1 s of heating vs time step -> observed order (Richardson)."""
    Ts = []
    for dt in dts:
        e = _ih(theat=1.0, tend=1.0, dt=dt)["energy"]
        Ts.append(float(e["Tsurf"][-1]))
    Ts = np.array(Ts)
    d = np.abs(np.diff(Ts))
    orders = np.log(d[:-1] / d[1:]) / np.log(2.0)
    return dict(name="V12 coupled time convergence", dt=np.array(dts), Tsurf=Ts, orders=orders)


# ---------------------------------------------------------------- process demonstrations (model only)

def p3_carbonitriding_cycle(stages=CN_CYCLE, threshold=0.35):
    """Boost/diffuse carbonitriding of a 0.2 %C steel; case depth at C + N = threshold wt%."""
    r = _cn(stages, nout=8)["out"]
    times = np.unique(r["t"])
    depth = []
    for t in times:
        m = r["t"] == t
        depth.append(ff.depth_at(r["x"][m], r["C"][m] + r["N"][m], threshold) * 1e3)
    x, C = _last(r, "C")
    _, N = _last(r, "N")
    return dict(name="P3 carbonitriding cycle", stages=stages, x=x, C=C, N=N, t_h=times / HOUR,
                depth_mm=np.array(depth), threshold=threshold)


def p4_induction_hardening(freqs=(10e3, 30e3, 100e3), steel="42CrMo4", margins=(0.0, 100.0), theat=3.0,
                           P_cold=8e6):
    """Induction heating of 42CrMo4 at equal initial (cold) surface power density P_cold [W/m^2].

    Hardened depth = depth where the peak temperature exceeded Ac3 (Andrews) + margin, as a function
    of heating time. The margin brackets the upward shift of austenitization under rapid heating.
    """
    ac1, ac3 = mat.andrews_ac1_ac3(mat.STEELS[steel])
    d0 = IH_DEFAULT
    res = {}
    for f in freqs:
        w = 2 * np.pi * f
        mu = an.MU0 * d0["mur0"]
        H0 = np.sqrt(4 * P_cold / (w * mu * an.skin_depth(f, d0["sigma20"], d0["mur0"])))
        out = _ih(freq=f, H0=H0, theat=theat, tend=theat, dt=5e-3, nout=30)
        r, e = out["out"], out["energy"]
        times = np.unique(r["t"])
        d = {mg: [] for mg in margins}
        for t in times:
            m = r["t"] == t
            for mg in margins:
                d[mg].append(ff.depth_at(r["x"][m], r["Tmax"][m], ac3 + mg) * 1e3)
        res[f] = dict(H0=float(H0), t=times, depth_mm={mg: np.nan_to_num(np.array(v)) for mg, v in d.items()},
                      energy=e, Tsurf_max=float(e["Tsurf"].max()),
                      profile=(r["x"][r["t"] == times[-1]], r["T"][r["t"] == times[-1]]))
    return dict(name=f"P4 induction hardening ({steel})", steel=steel, ac1=ac1, ac3=ac3, margins=margins,
                P_cold=P_cold, runs=res)


# ================================================================ milestone 3
# Phase transformations, hardness, retained austenite and residual stress.

from . import mechanics as mech  # noqa: E402
from . import metallurgy as metal  # noqa: E402


def _field(res, key):
    """Reshape snapshot CSV output (t, x, key) into t (nt,), x (nz,), F (nt, nz)."""
    t = np.unique(res["t"])
    x = res["x"][res["t"] == t[0]]
    return t, x, res[key].reshape(len(t), len(x))


def v13_thermoelastic_plate(nz=201, L=0.01):
    """Elastic plate, quadratic temperature: sigma = -E_b alpha (T - mean T) exactly."""
    z = np.linspace(0, L, nz)
    alpha, dT, Ec, nu = 1.2e-5, 300.0, 200e3, 0.3
    T = dT * (1 - z / L) ** 2                       # mean over [0, L] is dT / 3
    eps = alpha * T[None, :]
    r = mech.solve_layers(z, eps, np.full((1, nz), Ec), np.full((1, nz), 1e9), nu=nu)
    ref = -Ec / (1 - nu) * alpha * (T - dT / 3.0)
    return dict(name="V13 thermo-elastic plate vs analytical", z=z, num=r["sigma"][0], ref=ref,
                err=float(np.max(np.abs(r["sigma"][0] - ref)) / np.max(np.abs(ref))))


def v14_uniform_fields(nz=101):
    """Uniform temperature and transformation history -> no stress at all (exact)."""
    z = np.linspace(0, 0.01, nz)
    t = np.linspace(0, 100, 201)
    T = np.repeat((860.0 - 8.4 * t)[:, None], nz, axis=1)
    comp = mat.STEELS["20MnCr5"]
    ph = metal.phase_history(t, T, comp, 700.0, 830.0)
    r = mech.solve_layers(z, mech.free_strain(T, ph), mech.youngs_modulus(T), mech.yield_strength(T, ph),
                          mart=ph["mart"])
    return dict(name="V14 uniform history gives zero stress", err=float(np.max(np.abs(r["sigma"]))))


CN_QUENCH = dict(L=0.01, nx=200, dt=0.1, tend=600.0, T0=860.0, Tf=60.0, h=2000.0, k=30.0, rhocp=4.0e6, nout=1200)


def chain_carbonitriding(stages=CN_CYCLE, steel="20MnCr5", q=CN_QUENCH, K_trip=5e-5):
    """Furnace (C + N) -> oil quench (heat) -> phases -> hardness -> residual stress."""
    comp = mat.STEELS[steel]
    cn = _cn(stages, nout=1)["out"]
    xs, C = _last(cn, "C")
    _, N = _last(cn, "N")
    th = ff.run("diffusion_1d.edp", L=q["L"], nx=q["nx"], dt=q["dt"], tend=q["tend"], c0=q["T0"], cs=q["Tf"],
                beta=q["h"] / q["rhocp"], dmodel="const", D=q["k"] / q["rhocp"], nout=q["nout"])
    t, z, T = _field(th, "c")
    # Furnace state at t = 0; after the oil bath the load air-cools (slowly, uniformly) to 20 degC.
    t_air = t[-1] + np.linspace(60.0, 1800.0, 30)
    T_air = T[-1][None, :] + (20.0 - T[-1][None, :]) * np.linspace(0, 1, 31)[1:, None]
    t = np.concatenate([[0.0], t, t_air])
    T = np.vstack([np.full(len(z), q["T0"]), T, T_air])
    cint = np.interp(z, xs, C + N, right=comp["C"])
    ac1, ac3 = mat.andrews_ac1_ac3(comp)
    ph = metal.phase_history(t, T, comp, ac1, ac3, c_interstitial=cint)
    return _finish(dict(name=f"carbonitriding ({steel})", steel=steel, comp=comp, t=t, z=z, T=T, ph=ph,
                        C=np.interp(z, xs, C, right=comp["C"]), N=np.interp(z, xs, N, right=0.0),
                        hv_base=300.0, sy_base=400.0), K_trip)


def chain_induction(freq=30e3, theat=2.0, steel="42CrMo4", P_cold=8e6, tq=120.0, ac_shift=0.0, K_trip=5e-5,
                    nx=200, hq=None):
    """Induction heating -> spray quench -> phases -> hardness -> residual stress."""
    comp = mat.STEELS[steel]
    d0 = IH_DEFAULT
    H0 = np.sqrt(4 * P_cold / (2 * np.pi * freq * an.MU0 * d0["mur0"] * an.skin_depth(freq, d0["sigma20"], d0["mur0"])))
    extra = {} if hq is None else dict(hq=hq)
    out = _ih(freq=freq, H0=H0, theat=theat, tend=theat + tq, dt=5e-3, nout=int((theat + tq) / 0.05), nx=nx, **extra)
    t, z, T = _field(out["out"], "T")
    t = np.concatenate([[0.0], t])
    T = np.vstack([np.full(len(z), d0["T0"]), T])
    ac1, ac3 = mat.andrews_ac1_ac3(comp)
    ph = metal.phase_history(t, T, comp, ac1, ac3, ac_shift=ac_shift)
    return _finish(dict(name=f"induction {freq/1e3:.0f} kHz, {theat:g} s ({steel})", steel=steel, comp=comp,
                        t=t, z=z, T=T, ph=ph, hv_base=300.0, sy_base=700.0, ac1=ac1, ac3=ac3), K_trip)


def _finish(r, K_trip):
    ph, T = r["ph"], r["T"]
    r["hv"] = metal.hardness(ph, r["comp"], hv_base=r["hv_base"])
    m = mech.solve_layers(r["z"], mech.free_strain(T, ph), mech.youngs_modulus(T),
                          mech.yield_strength(T, ph, sy_base_rt=r["sy_base"]), mart=ph["mart"], K_trip=K_trip)
    r["mech"] = m
    r["sigma_res"] = m["sigma"][-1]
    r["ra"] = ph["aus"][-1]
    sy_end = mech.yield_strength(T, ph, sy_base_rt=r["sy_base"])[-1]
    r["yield_violation"] = float(np.max(np.abs(r["sigma_res"]) - sy_end))
    r["T_final_spread"] = float(np.ptp(T[-1]))
    return r


def summarize_chain(r):
    z_mm, hv, s = r["z"] * 1e3, r["hv"], r["sigma_res"]
    out = dict(surface_hv=float(hv[0]), core_hv=float(hv[-1]), surface_ra_pct=float(100 * r["ra"][0]),
               max_ra_pct=float(100 * r["ra"].max()), surface_sigma_MPa=float(s[0]),
               min_sigma_MPa=float(s.min()), max_tensile_MPa=float(s.max()),
               max_tensile_depth_mm=float(z_mm[np.argmax(s)]),
               equilibrium_residual=r["mech"]["equilibrium_residual"], yield_violation_MPa=r["yield_violation"],
               T_final_spread_C=r["T_final_spread"])
    out["chd550_mm"] = metal.depth_at_hv(z_mm, hv, 550.0)                 # ISO 2639 (case hardening)
    out["shd_mm"] = metal.depth_at_hv(z_mm, hv, 0.8 * hv[0])              # ISO 3754 style: 0.8 x surface HV
    return out


def chain_checks(chains):
    """V15/V16: equilibrium and yield-surface consistency of every chain run."""
    eq = max(r["mech"]["equilibrium_residual"] for r in chains)
    yv = max(r["yield_violation"] for r in chains)
    return dict(name="V15 equilibrium (all chains)", err=float(eq)), dict(name="V16 yield consistency (all chains)", err=float(max(yv, 0.0)))


def v17_mesh_sensitivity():
    """Residual stress / hardness metrics at nx and 2 nx (thermal + mechanics grid)."""
    out = {}
    pairs = {
        "carbonitriding": (chain_carbonitriding(), chain_carbonitriding(q=dict(CN_QUENCH, nx=400))),
        "induction": (chain_induction(), chain_induction(nx=400)),
    }
    for k, (a, b) in pairs.items():
        sa, sb = summarize_chain(a), summarize_chain(b)
        out[k] = {m: dict(nx=sa[m], nx2=sb[m], rel=abs(sb[m] - sa[m]) / max(abs(sb[m]), 1e-9))
                  for m in ("surface_sigma_MPa", "max_tensile_MPa", "surface_hv", "chd550_mm")}
    err = max(v["rel"] for d in out.values() for v in d.values())
    return dict(name="V17 mesh sensitivity of process outputs", metrics=out, err=float(err))


def sensitivity_m3():
    """One-at-a-time sensitivity of the key outputs to the assumed inputs."""
    rows = []
    def add(process, param, value, r):
        s_ = summarize_chain(r)
        rows.append(dict(process=process, param=param, value=value, surface_sigma_MPa=s_["surface_sigma_MPa"],
                         max_tensile_MPa=s_["max_tensile_MPa"], surface_hv=s_["surface_hv"],
                         surface_ra_pct=s_["surface_ra_pct"], depth_mm=s_["chd550_mm"]))
    for K in (0.0, 5e-5, 1e-4):
        add("carbonitriding", "K_trip [1/MPa]", K, chain_carbonitriding(K_trip=K))
    for h in (1000.0, 2000.0, 4000.0):
        add("carbonitriding", "oil HTC [W/m2K]", h, chain_carbonitriding(q=dict(CN_QUENCH, h=h)))
    for K in (0.0, 5e-5, 1e-4):
        add("induction", "K_trip [1/MPa]", K, chain_induction(K_trip=K))
    for a in (0.0, 50.0, 100.0):
        add("induction", "Ac shift [C]", a, chain_induction(ac_shift=a))
    for h in (1e4, 2e4, 4e4):
        add("induction", "spray HTC [W/m2K]", h, chain_induction(hq=h))
    return rows
