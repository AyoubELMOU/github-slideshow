#!/usr/bin/env python3
"""Workflow definition for a Ø50 mm steel cylinder: 3 steels x 5 heat treatments.

Writes results/cylinder50_workflow.json (machine-readable inputs/outputs per case) and
results/CYLINDER50_WORKFLOW.md. Steel-dependent numbers (Ac1, Ac3, Ms, idealized Q&P optimum,
skin depths) are computed here from the published correlations in hybridht; process settings
are starting points for the simulation design of experiments, not validated recipes.
Values in [brackets] must be supplied (heat analysis, measurements, equipment limits).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hybridht import analytical as an  # noqa: E402
from hybridht import materials as mat  # noqa: E402
from hybridht import metallurgy as metal  # noqa: E402

GEOMETRY = dict(shape="solid cylinder", diameter_mm=50.0, length_mm="[length; >= 200 mm assumed so mid-length is free of end effects]",
                model="1D radial (generalized plane strain) at mid-length; FreeFEM axisymmetric r-z for end effects")

# Nominal compositions (wt%). C45 and 42CrMo4: mid-range of EN 10083-2/-3. 40NiCrMo16 is not in the
# current EN 10083-3 list; composition assumed by analogy with 36NiCrMo16 (Ni ~4 %, Cr ~1.8 %, Mo ~0.35 %)
# with C = 0.40 from the designation. Replace all three with the heat analysis.
STEELS = {
    "C45": dict(comp=dict(C=0.45, Si=0.25, Mn=0.65), initial="normalized (ferrite-pearlite)",
                composition_source="mid-range EN 10083-2 (nominal)"),
    "42CrMo4": dict(comp=dict(C=0.42, Si=0.25, Mn=0.75, Cr=1.05, Mo=0.22), initial="quenched and tempered",
                    composition_source="mid-range EN 10083-3 (nominal)"),
    "40NiCrMo16": dict(comp=dict(C=0.40, Si=0.25, Mn=0.65, Ni=4.0, Cr=1.8, Mo=0.35), initial="quenched and tempered",
                       composition_source="ASSUMED by analogy with 36NiCrMo16; confirm"),
}


def qp_ideal(comp, T_room=20.0):
    """Speer-type idealized Q&P optimum: full C partitioning to austenite, no carbides,
    no interface migration (an upper bound; real low-Si steels lose C to carbides)."""
    Ms = metal.ms_van_bohemen(comp)
    best = None
    for QT in np.arange(T_room, Ms, 1.0):
        fm = 1 - np.exp(-metal.KM_ALPHA * (Ms - QT))
        fg = 1 - fm
        Cg = comp["C"] / fg
        Ms_g = metal.ms_van_bohemen(comp, Cg)
        ra = fg * np.exp(-metal.KM_ALPHA * max(Ms_g - T_room, 0.0))
        if best is None or ra > best[1]:
            best = (float(QT), float(ra), float(Cg), float(fm))
    return dict(QT_opt_C=best[0], RA_max_ideal=best[1], C_austenite_wt=best[2], primary_martensite=best[3])


def steel_data(name, s):
    c = s["comp"]
    ac1, ac3 = mat.andrews_ac1_ac3(c)
    return dict(name=name, composition_wt_pct=c, composition_source=s["composition_source"], initial_state=s["initial"],
                Ac1_C=round(ac1), Ac3_C=round(ac3), Ms_vanBohemen_C=round(float(metal.ms_van_bohemen(c))),
                Ms_Andrews_C=round(float(metal.ms_andrews(c))),
                HV_martensite_Maynier_water_quench=round(float(metal.hv_martensite_maynier(c, c["C"], 1e5))),
                qp_ideal={k: round(v, 3) for k, v in qp_ideal(c).items()})


COMMON_OUTPUTS = [
    "radial profiles at mid-length: temperature history, phase fractions, hardness, retained austenite",
    "residual stresses: hoop, axial, radial vs radius; maximum tension and its depth",
    "distortion: diameter and length change",
    "process KPIs: cycle time, energy per part",
    "sensitivity of the outputs to each assumed input",
]
VALIDATION = ["hardness traverse (HV1) on a cross-section", "XRD residual stress with layer removal",
              "XRD retained austenite", "metallography", "diameter and length (CMM)"]


def cases(sd):
    ac3 = {k: v["Ac3_C"] for k, v in sd.items()}
    qp = {k: v["qp_ideal"]["QT_opt_C"] for k, v in sd.items()}
    sigma_hot, mur_hot = 1e6, 1.0  # representative hot-steel conductivity (same as the 1D induction model)
    delta = {f: round(an.skin_depth(f, sigma_hot, mur_hot) * 1e3, 1) for f in (10e3, 30e3, 100e3)}
    return {
        "quenching_partitioning": dict(
            suitability={"C45": "not recommended: low hardenability at Ø50 and low Si (carbides consume C)",
                         "42CrMo4": "research case: low Si limits carbon partitioning",
                         "40NiCrMo16": "best of the three (hardenability, low Ms); low Si still a limit"},
            inputs={"austenitize": {k: f"{ac3[k] + 50} °C (Ac3 + 50), until core within 5 °C, + 15 min" for k in sd},
                    "quench to QT": {k: f"QT ≈ {qp[k]:.0f} °C (idealized optimum; refine with OpenCalphad CCE), salt bath or polymer"
                                     for k in sd},
                    "partition": {k: "PT 400 °C, tP 10–1000 s (DoE)" for k in sd},
                    "final cool": {k: "air to 20 °C" for k in sd}},
            specific_outputs=["retained austenite fraction and carbon vs radius", "fresh martensite vs radius",
                              "whether the core reaches QT before diffusional transformation (needs CCT data)"],
            tools=["OpenCalphad: T0, CCE, carbide driving force", "PRISMS-PF: partitioning with carbides",
                   "OpenFOAM: bath HTC", "FreeFEM: heat, KM, partitioning law, stress"],
            poc_status="not implemented"),
        "carburizing_QT": dict(
            suitability={k: "not recommended: medium-carbon grade; carburizing gives little case/core contrast and high retained austenite; modelled for comparison only"
                         for k in sd},
            inputs={"carburize": {k: "900 °C, carbon potential 0.80 % (restricted), 2 h boost + 1 h diffuse at 0.70 %" for k in sd},
                    "quench": {"C45": "water-polymer (low hardenability; crack risk)", "42CrMo4": "oil 60 °C",
                               "40NiCrMo16": "oil 60 °C or gas (high hardenability)"},
                    "temper": {k: "180 °C, 2 h" for k in sd}},
            specific_outputs=["carbon profile", "check of the carbide limit at the surface", "case depth CHD at 550 HV"],
            tools=["OpenCalphad: carbon activity, carbide limit", "OpenFOAM: furnace β, quench HTC",
                   "FreeFEM: C diffusion, heat, phases, tempering, stress"],
            poc_status="1D slab chain verified (no tempering yet); needs radial geometry"),
        "nitriding": dict(
            suitability={"C45": "use nitrocarburizing instead: no nitride formers, shallow and soft diffusion zone",
                         "42CrMo4": "recommended (Cr, Mo nitride formers)",
                         "40NiCrMo16": "recommended (Cr); Ni does not form nitrides"},
            inputs={"pre-treatment": {"C45": "normalized or Q&T", "42CrMo4": "Q&T, tempered ≥ 580 °C",
                                      "40NiCrMo16": "Q&T, tempered ≥ 580 °C"},
                    "treatment": {"C45": "gas nitrocarburizing 570 °C, 2 h, [NH3 + CO2 ratio]",
                                  "42CrMo4": "gas nitriding 520 °C, 30 h, KN from the Lehrer diagram for a thin γ′ layer",
                                  "40NiCrMo16": "gas nitriding 520 °C, 30 h, KN from the Lehrer diagram for a thin γ′ layer"},
                    "cool": {k: "slow, in N2" for k in sd}},
            specific_outputs=["nitrogen profile", "compound-layer thickness and phase (ε / γ′)",
                              "nitriding hardness depth (core + 50 HV)"],
            tools=["OpenCalphad: Fe–N–Cr, Lehrer diagram", "QE/LAMMPS: N trapping energies",
                   "OpenFOAM: NH3 dissociation → KN", "FreeFEM: diffusion–reaction, compound layer, stress"],
            poc_status="not implemented (N diffusion exists in the carbonitriding solver)"),
        "induction": dict(
            suitability={"C45": "recommended (classic induction grade); fast quench needed",
                         "42CrMo4": "recommended",
                         "40NiCrMo16": "possible; high hardenability, watch cracking and retained austenite"},
            inputs={"pre-treatment": {"C45": "normalized or Q&T", "42CrMo4": "Q&T", "40NiCrMo16": "Q&T"},
                    "heating": {k: (f"scan hardening, 10–30 kHz (hot skin depth {delta[10e3]} / {delta[30e3]} mm), "
                                    f"target 2–3 mm, [power], [scan speed], [coil gap]; peak T ≥ Ac3 = {ac3[k]} °C + rapid-heating margin")
                                for k in sd},
                    "quench": {"C45": "spray, water-polymer (higher severity)", "42CrMo4": "spray, polymer",
                               "40NiCrMo16": "spray, mild polymer (crack risk)"},
                    "temper": {k: "180 °C, 1 h" for k in sd}},
            specific_outputs=["surface hardening depth SHD (0.8 × minimum surface HV)", "peak temperature map",
                              "tensile peak below the case"],
            tools=["PRISMS-PF: rapid austenitization → Ac3 shift", "OpenFOAM: spray HTC",
                   "FreeFEM: EM μ(T,H) ⇄ heat → phases → stress"],
            poc_status="1D slab coupled model verified; needs radial geometry and scanning coil"),
        "laser_and_laser_induction": dict(
            suitability={"C45": "possible; self-quench works for thin cases", "42CrMo4": "recommended",
                         "40NiCrMo16": "recommended (high hardenability helps self-quench)"},
            inputs={"laser": {k: "diode laser on the rotating bar, helical track, [power], [spot size], [scan speed], [overlap], absorptivity [measured or calibrated]"
                              for k in sd},
                    "hybrid option": {k: "induction preheat/post-heat zone ahead of the spot, [frequency], [preheat temperature]" for k in sd},
                    "limits": {k: f"surface below solidus; peak T ≥ Ac3 = {ac3[k]} °C + rapid-heating margin" for k in sd}},
            specific_outputs=["track depth and width", "back-tempered overlap bands", "melting margin",
                              "residual stress along and across the track"],
            tools=["QE: absorptivity trends", "OpenCalphad: solidus, Ac1/Ac3", "PRISMS-PF: superheated austenitization",
                   "OpenFOAM: melt pool if melting", "FreeFEM: moving source (+ eddy currents), phases, stress"],
            poc_status="not implemented (eddy-current solver reusable for the hybrid)"),
    }


def main():
    sd = {k: steel_data(k, v) for k, v in STEELS.items()}
    doc = dict(geometry=GEOMETRY, steels=sd, cases=cases(sd), common_outputs=COMMON_OUTPUTS,
               validation_measurements=VALIDATION,
               notes=["Ac1/Ac3: Andrews (1965); Ms: van Bohemen (2012); HV: Maynier (1978) at 1e5 °C/h.",
                      "Q&P optimum: idealized full partitioning (upper bound).",
                      "Process settings are DoE starting points, not validated recipes; [brackets] = to supply."])
    out = os.path.join(ROOT, "results")
    with open(os.path.join(out, "cylinder50_workflow.json"), "w") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    L = ["# Ø50 mm cylinder: workflow inputs and outputs", "", "Generated by `scripts/cylinder50_workflow.py`.", "",
         "## Steels (computed from nominal compositions)", "",
         "| Steel | Composition source | Ac1 [°C] | Ac3 [°C] | Ms [°C] | Martensite HV (water) | Q&P ideal QT [°C] | Q&P ideal RA |",
         "|---|---|---|---|---|---|---|---|"]
    for k, v in sd.items():
        q = v["qp_ideal"]
        L.append(f"| {k} | {v['composition_source']} | {v['Ac1_C']} | {v['Ac3_C']} | {v['Ms_vanBohemen_C']} | "
                 f"{v['HV_martensite_Maynier_water_quench']} | {q['QT_opt_C']:.0f} | {100*q['RA_max_ideal']:.0f} % |")
    for name, c in doc["cases"].items():
        L += ["", f"## {name.replace('_', ' ')}", "", f"PoC status: {c['poc_status']}", "",
              "| Input | " + " | ".join(sd) + " |", "|---|" + "---|" * len(sd)]
        for step, vals in c["inputs"].items():
            L.append(f"| {step} | " + " | ".join(vals[k] for k in sd) + " |")
        L.append("| suitability | " + " | ".join(c["suitability"][k] for k in sd) + " |")
        L += ["", "Outputs: " + "; ".join(c["specific_outputs"] + COMMON_OUTPUTS), "",
              "Tools: " + "; ".join(c["tools"])]
    L += ["", "## Validation measurements", ""] + [f"- {m}" for m in VALIDATION]
    with open(os.path.join(out, "CYLINDER50_WORKFLOW.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(json.dumps(sd, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
