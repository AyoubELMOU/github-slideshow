"""Phase transformations and hardness along computed thermal histories.

Scope of this first version (each simplification is a planned upgrade, not a hidden one):
- Heating: austenite fraction from the peak temperature, lever rule between Ac1 and Ac3
  (equilibrium; rapid-heating shift handled by `ac_shift`).
- Cooling: martensite only, Koistinen-Marburger. Diffusional products (ferrite, pearlite,
  bainite) are NOT modelled: valid when the quench beats the CCT nose, which needs CCT data
  (OpenCalphad / literature) to check. Material that never austenitized keeps its initial state.
- Interstitials: N is treated as carbon-equivalent in Ms and hardness (assumption).
- As-quenched state: no tempering.

Sources:
- Ms (default): S. M. C. van Bohemen, Mater. Sci. Technol. 28 (2012) 487, exponential carbon
  dependence, valid for high-carbon cases where Andrews' linear formula (J. Iron Steel Inst. 203
  (1965) 721, kept for comparison) is far too low.
- Martensite kinetics: D. P. Koistinen, R. E. Marburger, Acta Metall. 7 (1959) 59.
- Martensite hardness: Maynier, Jungmann, Dollet (1978), in Hardenability Concepts with
  Applications to Steel, TMS-AIME:
      HV_M = 127 + 949 C + 27 Si + 11 Mn + 8 Ni + 16 Cr + 21 log10(Vr),  Vr in degC/h at 700 degC.
  Fitted on low-alloy steels; the carbon entering it is capped at `C_CAP_MAYNIER` (assumption) since
  martensite hardness saturates at high carbon.
"""
import numpy as np

KM_ALPHA = 0.011  # 1/K, Koistinen-Marburger
C_CAP_MAYNIER = 0.6  # wt%, assumption (see module docstring)


def ms_andrews(comp, c_interstitial=None):
    """Martensite start [degC], Andrews linear. `c_interstitial` (wt%) overrides comp['C']."""
    g = lambda k: comp.get(k, 0.0)
    C = g("C") if c_interstitial is None else np.asarray(c_interstitial, float)
    return 539.0 - 423.0 * C - 30.4 * g("Mn") - 17.7 * g("Ni") - 12.1 * g("Cr") - 7.5 * g("Mo")


def ms_van_bohemen(comp, c_interstitial=None):
    """Martensite start [degC], van Bohemen (2012):
    Ms = 565 - (31 Mn + 13 Si + 10 Cr + 18 Ni + 12 Mo) - 600 (1 - exp(-0.96 C))."""
    g = lambda k: comp.get(k, 0.0)
    C = g("C") if c_interstitial is None else np.asarray(c_interstitial, float)
    sub = 31 * g("Mn") + 13 * g("Si") + 10 * g("Cr") + 18 * g("Ni") + 12 * g("Mo")
    return 565.0 - sub - 600.0 * (1.0 - np.exp(-0.96 * C))


def hv_martensite_maynier(comp, C, vr_C_per_h):
    g = lambda k: comp.get(k, 0.0)
    vr = np.maximum(np.asarray(vr_C_per_h, float), 1.0)
    C = np.minimum(np.asarray(C, float), C_CAP_MAYNIER)
    return 127 + 949 * C + 27 * g("Si") + 11 * g("Mn") + 8 * g("Ni") + 16 * g("Cr") + 21 * np.log10(vr)


def phase_history(t, T, comp, ac1, ac3, c_interstitial=None, ac_shift=0.0, ms_model=None):
    """Phase fractions along thermal histories.

    t: (nt,) times [s];  T: (nt, nz) temperatures [degC]
    Returns dict of (nt, nz) arrays: aus (austenite), mart (martensite), base (untransformed
    initial microstructure), plus Ms (nz,), the cooling rate at 700 degC (nz,) and peak T (nz,).
    """
    T = np.asarray(T, float)
    nt, nz = T.shape
    cint = np.full(nz, comp["C"]) if c_interstitial is None else np.asarray(c_interstitial, float)
    Ms = (ms_model or ms_van_bohemen)(comp, cint)
    lo, hi = ac1 + ac_shift, ac3 + ac_shift
    Tpeak_run = np.maximum.accumulate(T, axis=0)
    f_aus_max = np.clip((Tpeak_run - lo) / (hi - lo), 0.0, 1.0)
    # Martensite forms only after the peak (on cooling), from the austenite present, driven by
    # the lowest temperature reached since the peak.
    after_peak = T < Tpeak_run - 1e-9
    Tmin_run = np.where(after_peak, T, np.inf)
    Tmin_run = np.minimum.accumulate(Tmin_run, axis=0)
    km = np.where(Tmin_run < Ms, 1.0 - np.exp(-KM_ALPHA * (Ms - np.minimum(Tmin_run, Ms))), 0.0)
    mart = f_aus_max * km
    aus = f_aus_max - mart
    base = 1.0 - f_aus_max
    # Cooling rate at 700 degC (first downward crossing after the peak), for Maynier.
    vr = np.zeros(nz)
    for j in range(nz):
        i_pk = int(np.argmax(T[:, j]))
        seg = T[i_pk:, j]
        k = np.nonzero(seg <= 700.0)[0]
        if T[i_pk, j] > 700.0 and len(k) and k[0] > 0:
            i = i_pk + k[0]
            vr[j] = (T[i - 1, j] - T[i, j]) / (t[i] - t[i - 1]) * 3600.0
    return dict(aus=aus, mart=mart, base=base, Ms=Ms, vr700=vr, Tpeak=T.max(axis=0), c=cint)


def hardness(ph, comp, hv_austenite=250.0, hv_base=300.0):
    """Rule-of-mixtures hardness [HV] of the final state (last time row).

    hv_austenite: retained-austenite hardness (assumed); hv_base: initial microstructure
    (e.g. quenched-and-tempered core). Martensite: Maynier with local C (+ N) and cooling rate.
    """
    hv_m = hv_martensite_maynier(comp, ph["c"], ph["vr700"])
    return ph["mart"][-1] * hv_m + ph["aus"][-1] * hv_austenite + ph["base"][-1] * hv_base


def depth_at_hv(x, hv, level):
    """Depth where a decreasing hardness profile drops below `level` (linear interpolation)."""
    below = np.nonzero(hv < level)[0]
    if len(below) == 0:
        return float(x[-1])
    i = below[0]
    if i == 0:
        return 0.0
    return float(x[i - 1] + (level - hv[i - 1]) * (x[i] - x[i - 1]) / (hv[i] - hv[i - 1]))
