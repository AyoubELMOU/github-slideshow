#!/usr/bin/env python3
"""Run verification (analytical), validation (published) and PINN cases; write results/.

    python3 scripts/run_all.py                 # everything
    python3 scripts/run_all.py --skip-pinn     # FreeFEM-only (fast)
    python3 scripts/run_all.py --pinn-steps 4000
"""
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hybridht import analytical as an  # noqa: E402
from hybridht import cases, published  # noqa: E402
from hybridht import freefem as ff  # noqa: E402

OUT = os.path.join(ROOT, "results")
HOUR = 3600.0

# Reference palette slots 1-3 (blue, orange, aqua); references in secondary ink.
C1, C2, C3, REF = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "lines.linewidth": 2,
    "legend.frameon": False, "font.size": 9,
})

# Acceptance criteria (agreed up front; see README).
TOL = dict(profile=1e-3, eddy=1e-3, skin=1e-3, order_min=1.8, harris=0.20, pinn_fwd=0.03, pinn_inv=0.15)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def verification(summary):
    v1, v2, v4 = cases.v1_diffusion_dirichlet(), cases.v2_diffusion_robin(), cases.v4_quench_convective()
    v3a, v3b = cases.v3_eddy_current("below_curie"), cases.v3_eddy_current("above_curie")
    v5 = cases.v5_convergence()

    fig, ax = plt.subplots(2, 2, figsize=(9, 6.5))
    for a, r, xs, xl, yl in [(ax[0, 0], v1, 1e3, "depth [mm]", "carbon [wt%]"),
                             (ax[0, 1], v2, 1e3, "depth [mm]", "carbon [wt%]"),
                             (ax[1, 0], v4, 1e3, "depth [mm]", "temperature [°C]")]:
        a.plot(r["x"] * xs, r["num"], color=C1, label="FreeFEM")
        a.plot(r["x"] * xs, r["ref"], "--", color=REF, lw=1.5, label="analytical")
        a.set(title=f"{r['name']}\nmax error {r['err']:.1e} (of range)", xlabel=xl, ylabel=yl)
        a.set_xlim(0, 3 if r is not v4 else 25)
        a.legend()
    a = ax[1, 1]
    for r, col in [(v3a, C1), (v3b, C2)]:
        d = r["delta"]
        a.semilogy(r["x"] / d, r["num"] / r["num"][0], color=col,
                   label=f"FreeFEM, {r['name'][17:-1]} (δ = {d*1e3:.3g} mm)")
        a.semilogy(r["x"] / d, r["ref"] / r["ref"][0], "--", color=REF, lw=1.5)
    a.semilogy([], [], "--", color=REF, lw=1.5, label="analytical")
    a.set(title="V3 induction: Joule power, 10 kHz\nskin depth error "
                f"{max(v3a['delta_err'], v3b['delta_err']):.1e}",
          xlabel="depth / skin depth δ", ylabel="p / p(0)", xlim=(0, 6), ylim=(1e-5, 1.5))
    a.legend()
    save(fig, "verification_profiles.png")

    fig, a = plt.subplots(figsize=(4.8, 3.6))
    a.loglog(v5["h"] * 1e6, v5["err"], "o-", color=C1, ms=6, label="FreeFEM (V2)")
    a.loglog(v5["h"] * 1e6, v5["err"][-1] * (v5["h"] / v5["h"][-1]) ** 2, "--", color=REF,
             lw=1.5, label="slope 2")
    a.set(title=f"V5 convergence, observed order {v5['orders'][-1]:.2f}",
          xlabel="element size h [µm] (Δt ∝ h²)", ylabel="max error (of range)")
    a.legend()
    save(fig, "verification_convergence.png")

    checks = {
        v1["name"]: (v1["err"], TOL["profile"]),
        v2["name"]: (v2["err"], TOL["profile"]),
        v3a["name"]: (max(v3a["err"], v3a["delta_err"]), TOL["eddy"]),
        v3b["name"]: (max(v3b["err"], v3b["delta_err"]), TOL["eddy"]),
        v4["name"]: (v4["err"], TOL["profile"]),
    }
    summary["verification"] = {k: dict(error=e, tol=t, passed=bool(e <= t)) for k, (e, t) in checks.items()}
    summary["verification"][v5["name"]] = dict(
        h_m=v5["h"].tolist(), error=v5["err"].tolist(), orders=v5["orders"].tolist(),
        tol=TOL["order_min"], passed=bool(v5["orders"][-1] >= TOL["order_min"]))


def validation_harris(summary):
    runs = {b: cases.p1_harris_case_depth(beta=b) for b in (1e-7, 2e-7, 5e-7)}
    base = runs[2e-7]
    t = np.array([r["t_h"] for r in base["rows"]])
    tt = np.linspace(1, 17, 100)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    a = ax[0]
    a.plot(tt, cases.mat.harris_case_depth_mm(tt, base["T_C"]), "--", color=REF, lw=1.5,
           label="Harris (1943) correlation")
    a.fill_between(tt, 0.8 * cases.mat.harris_case_depth_mm(tt, base["T_C"]),
                   1.2 * cases.mat.harris_case_depth_mm(tt, base["T_C"]), color=REF, alpha=0.08,
                   lw=0, label="±20 % acceptance band")
    for (b, r), col in zip(runs.items(), (C2, C1, C3)):
        d = [row["depth_total_mm"] for row in r["rows"]]
        a.plot(t, d, "o-", color=col, ms=6, label=f"FreeFEM, β = {b:.0e} m/s")
    a.set(title=f"P1 total case depth at {base['T_C']:.0f} °C", xlabel="time [h]",
          ylabel="total case depth [mm]")
    a.legend()
    a = ax[1]
    cols = [C1, C2, C3, "#6250d6"]
    for (th, x, c), col in zip(base["profiles"], cols):
        a.plot(x * 1e3, c, color=col, label=f"{th} h")
    a.axhline(base["c0"] + 0.05, color=REF, lw=1, ls=":")
    a.text(3.9, base["c0"] + 0.07, "total-case threshold (c₀ + 0.05)", ha="right", color=REF, fontsize=8)
    a.set(title="Carbon profiles, Tibbetts D(C,T), β = 2e-7 m/s", xlabel="depth [mm]",
          ylabel="carbon [wt%]", xlim=(0, 4))
    a.legend(title="time")
    save(fig, "validation_harris.png")

    summary["validation_harris"] = {
        f"beta={b:.0e}": dict(rows=r["rows"], sqrt_t_exponent=r["sqrt_t_exponent"],
                              max_rel_diff=max(abs(row["rel_diff"]) for row in r["rows"]))
        for b, r in runs.items()}
    summary["validation_harris"]["passed"] = bool(
        summary["validation_harris"]["beta=2e-07"]["max_rel_diff"] <= TOL["harris"])
    summary["validation_harris"]["tol"] = TOL["harris"]


def validation_published(summary):
    files = published.datasets()
    summary["validation_published"] = []
    for path in files:
        r = published.compare(path)
        entry = dict(file=os.path.relpath(path, ROOT), source=r["meta"]["source"], doi=r["meta"]["doi"])
        if "skipped" in r:
            entry["skipped"] = r["skipped"]
        else:
            entry.update(rmse=r["rmse"], bias=r["bias"], max_abs=r["max_abs"], beta=r["beta"],
                         beta_assumed=r["beta_assumed"])
            fig, a = plt.subplots(figsize=(4.8, 3.6))
            a.plot(r["x_mm"], r["c"], color=C1, label="FreeFEM")
            unc = r["data"]["uncertainty"] if "uncertainty" in r["data"].dtype.names else None
            a.errorbar(r["data"]["depth_mm"], r["data"]["value"], yerr=unc, fmt="o", color=REF,
                       ms=5, label="measured")
            a.set(title=f"{os.path.basename(path)}\nRMSE {r['rmse']:.3f} wt%", xlabel="depth [mm]",
                  ylabel="carbon [wt%]", xlim=(0, max(r["data"]["depth_mm"]) * 1.3))
            a.legend()
            save(fig, f"published_{os.path.splitext(os.path.basename(path))[0]}.png")
        summary["validation_published"].append(entry)


def pinn_cases(summary, steps):
    from hybridht import pinn

    D, beta, c0, cp, t_end = 1.5e-11, 2e-7, 0.2, 1.0, 4 * HOUR
    prob = pinn.Problem(D=D, beta=beta, t_end=t_end, Lc=3e-3, c0=c0, cp=cp)
    times = [0.5, 1, 2, 4]
    x = np.linspace(0, 3e-3, 301)

    # Forward: PINN vs analytical vs FreeFEM.
    fwd = pinn.train(prob, mode="forward", steps=steps)
    fe = ff.run("diffusion_1d.edp", L=5e-3, nx=400, dt=10, tend=t_end, c0=c0, cs=cp, beta=beta,
                dmodel="const", D=D, nout=8)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    errs = {}
    for th, col in zip(times, (C1, C2, C3, "#6250d6")):
        ref = an.diffusion_robin(x, th * HOUR, D, beta, c0, cp)
        cp_ = pinn.predict(fwd, prob, x, th * HOUR)
        m = np.isclose(fe["t"], th * HOUR)
        ax[0].plot(x * 1e3, cp_, color=col, label=f"PINN {th} h")
        ax[0].plot(x * 1e3, ref, "--", color=REF, lw=1.2)
        ax[0].plot(fe["x"][m][::12] * 1e3, fe["c"][m][::12], "o", color=col, ms=4, mfc="none")
        errs[th] = dict(pinn_vs_analytical=float(np.max(np.abs(cp_ - ref)) / (cp - c0)),
                        freefem_vs_analytical=float(np.max(np.abs(
                            fe["c"][m] - an.diffusion_robin(fe["x"][m], th * HOUR, D, beta, c0, cp))) / (cp - c0)))
    ax[0].plot([], [], "--", color=REF, lw=1.2, label="analytical")
    ax[0].plot([], [], "o", color=REF, ms=4, mfc="none", label="FreeFEM")
    ax[0].set(title="PINN forward (V2 problem)", xlabel="depth [mm]", ylabel="carbon [wt%]", xlim=(0, 2.5))
    ax[0].legend(fontsize=7.5)
    ax[1].bar([str(t) for t in times], [errs[t]["pinn_vs_analytical"] * 100 for t in times], color=C1,
              width=0.5, label="PINN")
    ax[1].bar([str(t) for t in times], [errs[t]["freefem_vs_analytical"] * 100 for t in times], color=C2,
              width=0.25, label="FreeFEM")
    ax[1].set(title="Max error vs analytical", xlabel="time [h]", ylabel="error [% of range]")
    ax[1].legend()
    save(fig, "pinn_forward.png")

    # Inverse: identify D and beta from sparse, noisy SYNTHETIC measurements (FreeFEM + 1 % noise).
    rng = np.random.default_rng(1)
    obs_t = [1, 2, 4]
    xs, ts, cs = [], [], []
    for th in obs_t:
        m = np.isclose(fe["t"], th * HOUR)
        xd = np.linspace(0.05e-3, 1.5e-3, 12)
        cd = np.interp(xd, fe["x"][m], fe["c"][m]) + 0.01 * (cp - c0) * rng.standard_normal(xd.size)
        xs.append(xd); ts.append(np.full_like(xd, th * HOUR)); cs.append(cd)
    data = (np.concatenate(xs), np.concatenate(ts), np.concatenate(cs))
    D0, b0 = 0.5 * D, 3.0 * beta
    inv = pinn.train(prob, mode="inverse", data=data, steps=steps, D_guess=D0, beta_guess=b0)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for th, col in zip(obs_t, (C1, C2, C3)):
        m = data[1] == th * HOUR
        ax[0].plot(data[0][m] * 1e3, data[2][m], "o", color=col, ms=5, mfc="none")
        ax[0].plot(x * 1e3, pinn.predict(inv, prob, x, th * HOUR), color=col, label=f"PINN fit {th} h")
    ax[0].plot([], [], "o", color=REF, ms=5, mfc="none", label="synthetic data (1 % noise)")
    ax[0].set(title="PINN inverse: calibrate D and β", xlabel="depth [mm]", ylabel="carbon [wt%]",
              xlim=(0, 2.5))
    ax[0].legend(fontsize=7.5)
    hst = inv["history"]
    st = [h["step"] for h in hst]
    ax[1].plot(st, [h["D"] / D for h in hst], color=C1, label="D / D_true")
    ax[1].plot(st, [h["beta"] / beta for h in hst], color=C2, label="β / β_true")
    ax[1].axhline(1, color=REF, ls="--", lw=1.2)
    ax[1].set(title="Parameter convergence", xlabel="training step", ylabel="identified / true")
    ax[1].legend()
    save(fig, "pinn_inverse.png")

    summary["pinn_forward"] = dict(errors=errs, train_s=fwd["train_s"], steps=steps,
                                   max_error=max(e["pinn_vs_analytical"] for e in errs.values()),
                                   tol=TOL["pinn_fwd"])
    summary["pinn_forward"]["passed"] = bool(summary["pinn_forward"]["max_error"] <= TOL["pinn_fwd"])
    rel = dict(D=abs(inv["D"] - D) / D, beta=abs(inv["beta"] - beta) / beta)
    summary["pinn_inverse"] = dict(D_true=D, beta_true=beta, D_initial=D0, beta_initial=b0,
                                   D_identified=inv["D"], beta_identified=inv["beta"], rel_error=rel,
                                   train_s=inv["train_s"], steps=steps, tol=TOL["pinn_inv"],
                                   data="synthetic: FreeFEM + 1 % Gaussian noise, 3 times x 12 depths",
                                   passed=bool(max(rel.values()) <= TOL["pinn_inv"]))
    # Inference cost: what a factory-side surrogate call costs once trained.
    t0 = time.time()
    for _ in range(100):
        pinn.predict(fwd, prob, x, t_end)
    summary["pinn_forward"]["inference_ms_301_points"] = (time.time() - t0) * 10


def report(summary):
    L = ["# Verification & validation report", "",
         "Generated by `scripts/run_all.py`. Numbers below are copied from `summary.json`.", "",
         "## Verification against analytical solutions", "",
         "| Case | Error | Tolerance | Result |", "|---|---|---|---|"]
    for k, v in summary["verification"].items():
        e = v["error"] if not isinstance(v["error"], list) else f"observed order {v['orders'][-1]:.2f}"
        e = f"{e:.2e}" if isinstance(e, float) else e
        L.append(f"| {k} | {e} | {v['tol']} | {'PASS' if v['passed'] else 'FAIL'} |")
    L += ["", "![profiles](verification_profiles.png)", "", "![convergence](verification_convergence.png)", "",
          "## Validation against published data", "",
          "### P1: case depth vs the Harris (1943) correlation", "",
          "Harris' correlation was fitted to measured carburized case depths. Model: Tibbetts (1980)",
          "carbon diffusivity D(C,T), surface mass transfer β (assumed; see sensitivity), 927 °C,",
          "carbon potential 1.0 wt%, core 0.2 wt%. Total case depth taken at c₀ + 0.05 wt%.", "",
          "| β [m/s] | t [h] | surface C [wt%] | total case [mm] | depth at 0.4 %C [mm] | Harris [mm] | diff |",
          "|---|---|---|---|---|---|---|"]
    for k, v in summary["validation_harris"].items():
        if not k.startswith("beta"):
            continue
        for r in v["rows"]:
            L.append(f"| {k[5:]} | {r['t_h']} | {r['surface_c']:.3f} | {r['depth_total_mm']:.3f} | "
                     f"{r['depth_0p4_mm']:.3f} | {r['harris_mm']:.3f} | {100*r['rel_diff']:+.1f} % |")
    vh = summary["validation_harris"]
    L += ["", f"Time exponent of case depth (β = 2e-7): {vh['beta=2e-07']['sqrt_t_exponent']:.3f} "
              "(ideal diffusion: 0.5). "
              f"Acceptance: within ±{100*vh['tol']:.0f} % at β = 2e-7 → **{'PASS' if vh['passed'] else 'FAIL'}**.",
          "", "![harris](validation_harris.png)", "", "### P2: digitized published profiles", ""]
    if not summary["validation_published"]:
        L += ["No datasets in `data/published/` yet. See `data/published/README.md` for the format and",
              "candidate open-access sources."]
    for e in summary["validation_published"]:
        if "skipped" in e:
            L.append(f"- `{e['file']}`: skipped ({e['skipped']})")
        else:
            L.append(f"- `{e['file']}` ({e['source']}, doi:{e['doi']}): RMSE {e['rmse']:.3f} wt%, bias "
                     f"{e['bias']:+.3f}, max {e['max_abs']:.3f}; β = {e['beta']:.1e}"
                     f"{' (assumed)' if e['beta_assumed'] else ''}")
    if "pinn_forward" in summary:
        pf, pi = summary["pinn_forward"], summary["pinn_inverse"]
        L += ["", "## PINN (hybrid layer)", "",
              f"Forward: max error vs analytical {100*pf['max_error']:.2f} % of range "
              f"(tolerance {100*pf['tol']:.0f} %) → **{'PASS' if pf['passed'] else 'FAIL'}**. "
              f"Training {pf['train_s']:.0f} s on CPU ({pf['steps']} steps); inference "
              f"{pf['inference_ms_301_points']:.2f} ms for a 301-point profile.", "",
              "| t [h] | PINN vs analytical | FreeFEM vs analytical |", "|---|---|---|"]
        for t, e in pf["errors"].items():
            L.append(f"| {t} | {100*e['pinn_vs_analytical']:.2f} % | {100*e['freefem_vs_analytical']:.3f} % |")
        L += ["", "![pinn forward](pinn_forward.png)", "",
              f"Inverse ({pi['data']}): starting from D = {pi['D_initial']:.2e}, β = {pi['beta_initial']:.2e}, "
              f"identified D = {pi['D_identified']:.3e} m²/s (true {pi['D_true']:.2e}, error "
              f"{100*pi['rel_error']['D']:.1f} %), β = {pi['beta_identified']:.3e} m/s (true "
              f"{pi['beta_true']:.2e}, error {100*pi['rel_error']['beta']:.1f} %). Tolerance "
              f"{100*pi['tol']:.0f} % → **{'PASS' if pi['passed'] else 'FAIL'}**.", "",
              "![pinn inverse](pinn_inverse.png)"]
    with open(os.path.join(OUT, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pinn", action="store_true")
    ap.add_argument("--pinn-steps", type=int, default=20000)
    args = ap.parse_args()
    if not ff.available():
        sys.exit("FreeFem++ not found (set FREEFEM=/path/to/FreeFem++)")
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for name, fn in [("verification", verification), ("validation (Harris)", validation_harris),
                     ("validation (published)", validation_published)]:
        t0 = time.time()
        fn(summary)
        print(f"{name}: {time.time() - t0:.0f} s", flush=True)
    if not args.skip_pinn:
        t0 = time.time()
        pinn_cases(summary, args.pinn_steps)
        print(f"PINN: {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    report(summary)
    print(open(os.path.join(OUT, "REPORT.md")).read())


if __name__ == "__main__":
    main()
