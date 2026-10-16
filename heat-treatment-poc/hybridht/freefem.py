"""Thin driver around the FreeFEM scripts in ../freefem."""
import os
import shutil
import subprocess
import tempfile

import numpy as np

FF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "freefem")
FF_EXE = os.environ.get("FREEFEM", "FreeFem++")


def available():
    return shutil.which(FF_EXE) is not None


def _read_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    cols = {name: np.atleast_1d(np.asarray(data[name])) for name in data.dtype.names}
    if "x" in cols:
        order = np.lexsort((cols["x"], cols["t"])) if "t" in cols else np.argsort(cols["x"])
    else:
        order = np.argsort(cols["t"], kind="stable")
    return {k: v[order] for k, v in cols.items()}


def run_multi(script, outputs=("out",), files=None, **params):
    """Run a FreeFEM script with -key value arguments.

    outputs: names of CSV-output arguments (e.g. "out", "mass", "energy"); each is pointed at a
             temporary file and returned parsed, as {name: {column: array}}.
    files:   {argument: text} written to temporary files and passed by path (e.g. stage tables).
    """
    with tempfile.TemporaryDirectory() as tmp:
        args = [FF_EXE, "-nw", "-v", "0", os.path.join(FF_DIR, script)]
        for k, v in params.items():
            args += [f"-{k}", str(v)]
        for k, text in (files or {}).items():
            path = os.path.join(tmp, f"{k}.txt")
            with open(path, "w") as fh:
                fh.write(text)
            args += [f"-{k}", path]
        paths = {name: os.path.join(tmp, f"{name}.csv") for name in outputs}
        for name, path in paths.items():
            args += [f"-{name}", path]
        proc = subprocess.run(args, cwd=FF_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FreeFEM failed ({' '.join(args)}):\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
        return {name: _read_csv(path) for name, path in paths.items()}


def run(script, **params):
    """Run a FreeFEM script whose single output is -out; return the CSV as a dict of arrays."""
    return run_multi(script, ("out",), **params)["out"]


def stages_text(stages):
    """Stage table for carbonitriding_1d.edp from dicts with keys
    hours, T_C, C_pot, beta_C, N_pot, beta_N."""
    lines = [str(len(stages))]
    for s in stages:
        lines.append(f"{s['hours'] * 3600.0} {s['T_C'] + 273.15} {s['C_pot']} {s['beta_C']} "
                     f"{s.get('N_pot', 0.0)} {s.get('beta_N', 0.0)}")
    return "\n".join(lines) + "\n"


def final_profile(res):
    """Last snapshot of a diffusion_1d result as (x, c)."""
    m = res["t"] == res["t"].max()
    return res["x"][m], res["c"][m]


def depth_at(x, c, threshold):
    """First depth where a decreasing profile crosses `threshold` (linear interpolation)."""
    below = np.nonzero(c <= threshold)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return 0.0
    return x[i - 1] + (threshold - c[i - 1]) * (x[i] - x[i - 1]) / (c[i] - c[i - 1])
