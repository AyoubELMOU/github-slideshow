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


def run(script, **params):
    """Run a FreeFEM script with -key value arguments; return the output CSV as a dict of arrays."""
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.csv")
        args = [FF_EXE, "-nw", "-v", "0", os.path.join(FF_DIR, script)]
        for k, v in params.items():
            args += [f"-{k}", str(v)]
        args += ["-out", out]
        proc = subprocess.run(args, cwd=FF_DIR, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FreeFEM failed ({' '.join(args)}):\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
        data = np.genfromtxt(out, delimiter=",", names=True)
    cols = {name: np.asarray(data[name]) for name in data.dtype.names}
    order = np.lexsort((cols["x"], cols["t"])) if "t" in cols else np.argsort(cols["x"])
    return {k: v[order] for k, v in cols.items()}


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
