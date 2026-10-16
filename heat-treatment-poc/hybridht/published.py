"""Ingest digitized published measurements and compare them with the FreeFEM model.

Dataset format (one CSV per measured profile, in data/published/):

    # source: Author A., Author B., Title, Journal Vol (Year) pages
    # doi: 10.xxxx/xxxxx
    # figure_or_table: Fig. 7a
    # digitization: WebPlotDigitizer 4.7, 3 repeats, mean reported   (or "table values")
    # synthetic: no
    # steel: 20MnCr5
    # quantity: carbon_wt_pct
    # T_C: 930
    # carbon_potential: 1.0
    # c0: 0.20
    # t_h: 6
    # beta_m_per_s: 2.0e-7          (optional; marked "assumed" in the report if absent)
    depth_mm,value,uncertainty
    0.05,0.95,0.03
    ...

Only single-stage carburizing profiles (quantity = carbon_wt_pct) are simulated for now;
boost/diffuse cycles and hardness profiles need the multi-stage and phase-transformation
models planned next.
"""
import glob
import os

import numpy as np

from . import freefem as ff

REQUIRED = ["source", "doi", "figure_or_table", "digitization", "synthetic", "steel",
            "quantity", "T_C", "carbon_potential", "c0", "t_h"]
DEFAULT_BETA = 2e-7
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "published")


def load(path):
    meta, body = {}, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                k, _, v = line[1:].partition(":")
                meta[k.strip()] = v.strip()
            elif line.strip():
                body.append(line)
    missing = [k for k in REQUIRED if k not in meta]
    if missing:
        raise ValueError(f"{path}: missing metadata {missing}")
    data = np.genfromtxt(body, delimiter=",", names=True)
    return meta, data


def simulate(meta, nx=300, dt=60.0):
    beta = float(meta.get("beta_m_per_s", DEFAULT_BETA))
    prm = dict(L=6e-3, nx=nx, dt=dt, tend=float(meta["t_h"]) * 3600.0, c0=float(meta["c0"]),
               cs=float(meta["carbon_potential"]), beta=beta, dmodel="tibbetts",
               T=float(meta["T_C"]) + 273.15, picard=2)
    x, c = ff.final_profile(ff.run("diffusion_1d.edp", **prm))
    return x * 1e3, c, beta


def compare(path, **kw):
    meta, data = load(path)
    if meta["quantity"] != "carbon_wt_pct":
        return dict(path=path, meta=meta, skipped=f"quantity {meta['quantity']} not simulated yet")
    x_mm, c, beta = simulate(meta, **kw)
    pred = np.interp(data["depth_mm"], x_mm, c)
    res = pred - data["value"]
    return dict(path=path, meta=meta, data=data, x_mm=x_mm, c=c, pred=pred, beta=beta,
                beta_assumed="beta_m_per_s" not in meta,
                rmse=float(np.sqrt(np.mean(res**2))), bias=float(np.mean(res)),
                max_abs=float(np.max(np.abs(res))))


def datasets(directory=DATA_DIR):
    return sorted(glob.glob(os.path.join(directory, "*.csv")))
