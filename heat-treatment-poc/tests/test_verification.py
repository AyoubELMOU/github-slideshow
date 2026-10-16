"""Fast regression tests: FreeFEM vs analytical solutions, published-data pipeline, material models."""
import os

import numpy as np
import pytest

from hybridht import analytical as an
from hybridht import cases, published
from hybridht import freefem as ff
from hybridht import materials as mat

needs_ff = pytest.mark.skipif(not ff.available(), reason="FreeFem++ not installed")


def test_harris_reduces_to_textbook_form():
    # 0.025 in/sqrt(h) at 1700 F (927 C)
    assert mat.harris_case_depth_mm(1.0, 926.67) == pytest.approx(0.635, rel=5e-3)


def test_tibbetts_order_of_magnitude():
    D = mat.tibbetts_diffusivity(0.8, 1200.0)
    assert 1e-11 < D < 5e-11


def test_robin_tends_to_dirichlet_for_large_beta():
    x = np.linspace(0, 2e-3, 50)
    a = an.diffusion_robin(x, 3600.0, 1e-11, 1e3, 0.2, 1.0)
    b = an.diffusion_dirichlet(x, 3600.0, 1e-11, 0.2, 1.0)
    assert np.max(np.abs(a - b)) < 1e-6


@needs_ff
def test_v1_dirichlet():
    assert cases.v1_diffusion_dirichlet(nx=200, dt=20.0)["err"] < 2e-3


@needs_ff
def test_v2_robin():
    assert cases.v2_diffusion_robin(nx=200, dt=20.0)["err"] < 2e-3


@needs_ff
@pytest.mark.parametrize("regime", ["below_curie", "above_curie"])
def test_v3_eddy_current(regime):
    r = cases.v3_eddy_current(regime, nx=100)
    assert r["err"] < 1e-3 and r["delta_err"] < 1e-3


@needs_ff
def test_v4_quench():
    assert cases.v4_quench_convective(nx=300, dt=0.02)["err"] < 2e-3


@needs_ff
def test_v5_second_order():
    r = cases.v5_convergence(levels=(25, 50, 100), tend=3600.0)
    assert r["orders"][-1] > 1.7


@needs_ff
def test_published_pipeline_on_synthetic_profile(tmp_path):
    """The ingestion/compare path, exercised on a profile the model itself generated."""
    meta = dict(T_C=927, carbon_potential=1.0, c0=0.2, t_h=2, beta_m_per_s=2e-7)
    x_mm, c, _ = published.simulate(meta, nx=150, dt=120.0)
    depth = np.linspace(0.1, 1.2, 8)
    header = "\n".join(f"# {k}: {v}" for k, v in {
        "source": "synthetic test profile", "doi": "none", "figure_or_table": "none",
        "digitization": "none", "synthetic": "yes", "steel": "test", "quantity": "carbon_wt_pct",
        **meta}.items())
    rows = "\n".join(f"{d},{v},0.01" for d, v in zip(depth, np.interp(depth, x_mm, c)))
    path = tmp_path / "synthetic.csv"
    path.write_text(f"{header}\ndepth_mm,value,uncertainty\n{rows}\n")
    r = published.compare(str(path), nx=150, dt=120.0)
    assert r["rmse"] < 1e-6 and not r["beta_assumed"]


def test_published_rejects_missing_provenance(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("# quantity: carbon_wt_pct\ndepth_mm,value\n0.1,0.9\n")
    with pytest.raises(ValueError, match="missing metadata"):
        published.load(str(path))


def test_no_synthetic_data_in_published_folder():
    for path in published.datasets():
        assert published.load(path)[0]["synthetic"].lower() == "no", path


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1 to train the PINN")
def test_pinn_forward_short():
    from hybridht import pinn
    prob = pinn.Problem(D=1.5e-11, beta=2e-7, t_end=4 * 3600.0, Lc=3e-3, c0=0.2, cp=1.0)
    r = pinn.train(prob, steps=3000)
    x = np.linspace(0, 3e-3, 101)
    err = np.max(np.abs(pinn.predict(r, prob, x, 4 * 3600.0)
                        - an.diffusion_robin(x, 4 * 3600.0, 1.5e-11, 2e-7, 0.2, 1.0))) / 0.8
    assert err < 0.08


# ---------------------------------------------------------------- milestone 2

@needs_ff
def test_v6_stage_split_invariance():
    assert cases.v6_stage_split()["err"] < 1e-10


@needs_ff
def test_v8_mass_conservation():
    assert cases.v8_conservation()["err"] < 1e-8


@needs_ff
def test_v9_cross_code():
    assert cases.v9_cross_code()["err"] < 1e-10


@needs_ff
def test_v10_em_surface_power():
    assert cases.v10_em_surface_power()["err"] < 1e-3


@needs_ff
def test_v11_energy_conservation():
    e = cases._ih(theat=0.3, tend=0.5, dt=1e-2)["energy"]
    bal = np.abs(e["Ejoule"] - e["Estored"] - e["Eloss"]) / e["Ejoule"].max()
    assert bal.max() < 1e-8


def test_andrews_42crmo4():
    ac1, ac3 = mat.andrews_ac1_ac3(mat.STEELS["42CrMo4"])
    assert 700 < ac1 < ac3 < 850


# ---------------------------------------------------------------- milestone 3

from hybridht import mechanics as mech  # noqa: E402
from hybridht import metallurgy as metal  # noqa: E402


def test_v13_thermoelastic_plate():
    assert cases.v13_thermoelastic_plate()["err"] < 1e-4


def test_v14_uniform_history_zero_stress():
    assert cases.v14_uniform_fields()["err"] < 1e-10


def test_ms_models():
    c = mat.STEELS["42CrMo4"]
    # Both formulas agree for medium carbon; van Bohemen stays physical at high carbon.
    assert abs(metal.ms_andrews(c) - metal.ms_van_bohemen(c)) < 10
    assert metal.ms_van_bohemen(mat.STEELS["20MnCr5"], 1.0) > 100 > metal.ms_andrews(mat.STEELS["20MnCr5"], 1.0)


def test_koistinen_marburger_and_retained_austenite():
    t = np.linspace(0, 10, 101)
    T = np.linspace(900, 20, 101)[:, None]
    comp = mat.STEELS["42CrMo4"]
    ph = metal.phase_history(t, T, comp, 740.0, 790.0)
    Ms = metal.ms_van_bohemen(comp)
    assert ph["aus"][-1, 0] == pytest.approx(np.exp(-metal.KM_ALPHA * (Ms - 20.0)), rel=1e-9)
    assert ph["mart"][-1, 0] + ph["aus"][-1, 0] == pytest.approx(1.0)


def test_trip_strain_full_transformation():
    """Equi-biaxial Leblond TRIP, d eps = K sigma (1 - f) df, integrates to K sigma / 2 at f = 1
    (the uniaxial form, 3/2 K s Phi'(f) df with s = 2 sigma / 3, gives K sigma)."""
    K, s, n = 5e-5, 100.0, 20000
    f = np.linspace(0, 1, n + 1)
    eps = np.sum(K * s * (1 - f[:-1]) * np.diff(f))
    assert eps == pytest.approx(K * s / 2, rel=1e-3)


@needs_ff
def test_carbonitriding_chain_regression():
    r = cases.chain_carbonitriding(q=dict(cases.CN_QUENCH, nx=100))
    s = cases.summarize_chain(r)
    assert s["equilibrium_residual"] < 1e-10 and s["yield_violation_MPa"] <= 1e-6
    assert s["surface_sigma_MPa"] < 0 < s["max_tensile_MPa"]          # compressive case, tensile core
    assert 10 < s["surface_ra_pct"] < 50
