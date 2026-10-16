# Hybrid FEA + PINN heat treatment of gears: proof of concept

The aim is to show two communities that a hybrid finite-element + physics-informed neural network
(PINN) chain built only on open-source codes can predict how a gear responds to heat treatment,
comparing **hybrid induction hardening** with **carbonitriding**:

- **Researchers** need to see that it is possible and rigorous: verified, validated and reproducible.
- **Engineers** need to see that it can run in a factory: fast, calibrated to shop-floor data
  and usable by operators.

## Architecture

```
 offline, high fidelity (hours)                                  online, factory (ms)
 ┌──────────────┐  ┌──────────┐  ┌─────────────┐  ┌──────────┐
 │ Quantum      │→ │ LAMMPS   │→ │ OpenCalphad │→ │PRISMS-PF │─┐
 │ ESPRESSO     │  │ (D, mob.)│  │ (phases, Ms)│  │(kinetics)│ │   ┌──────────────────┐
 └──────────────┘  └──────────┘  └─────────────┘  └──────────┘ ├──→│ FreeFEM macro    │──→ PINN / neural
 ┌──────────────────────────────────────────┐                  │   │ EM + heat + C/N  │    operator surrogate
 │ OpenFOAM: quench HTC, furnace β          │──────────────────┘   │ + phases + mech. │    + inverse calibration
 └──────────────────────────────────────────┘                      └──────────────────┘    from sensor data
```

Only two couplings are two-way: electromagnetics with heat (μ(T), σ(T), Curie point), and
heat with phase transformation and mechanics. Both live inside FreeFEM. Everything else is a
one-way parameter chain.

## What is in this first milestone

| ID | Case | Reference | Code |
|---|---|---|---|
| V1 | Carburizing, constant D, fixed surface C | erfc solution | `freefem/diffusion_1d.edp` |
| V2 | Carburizing with surface mass transfer β | Crank / Carslaw–Jaeger | `freefem/diffusion_1d.edp` |
| V3 | Induction: harmonic eddy currents below/above Curie | sinh solution, skin depth | `freefem/eddy_current_1d.edp` |
| V4 | Quench with convective surface | same Robin solution for T | `freefem/diffusion_1d.edp` |
| V5 | Mesh/time convergence of V2 | observed order of accuracy | `hybridht/cases.py` |
| P1 | Nonlinear carburizing, Tibbetts D(C,T) | Harris (1943) case-depth correlation from measured depths | `hybridht/cases.py` |
| P2 | Digitized published profiles | `data/published/*.csv` with provenance | `hybridht/published.py` |
| N1 | PINN forward (V2 problem) | analytical + FreeFEM | `hybridht/pinn.py` |
| N2 | PINN inverse: identify D and β from sparse noisy data | known true values (synthetic data) | `hybridht/pinn.py` |
| V6 | Multi-stage schedule: splitting a stage | exact invariance | `freefem/carbonitriding_1d.edp` |
| V7 | C + N co-diffusion | analytical (each species) | `freefem/carbonitriding_1d.edp` |
| V8 | Boost/diffuse carbonitriding | mass conservation (surface flux vs profile) | `freefem/carbonitriding_1d.edp` |
| V9 | Carbonitriding solver vs carburizing solver | cross-code identity | both |
| V10 | Induction with imposed surface field, cold and hot | P = H0²ωμδ/4 | `freefem/induction_heating_1d.edp` |
| V11 | Coupled EM–thermal heating + spray quench | energy conservation | `freefem/induction_heating_1d.edp` |
| V12 | Coupled problem, time-step refinement | observed order | `freefem/induction_heating_1d.edp` |
| P3 | Boost/diffuse carbonitriding cycle | model only (case depth at C + N = 0.35 wt%) | `hybridht/cases.py` |
| P4 | Induction hardening of 42CrMo4 at 10/30/100 kHz | model only (peak T ≥ Andrews Ac3) | `hybridht/cases.py` |
| V13 | Thermo-elastic plate, quadratic temperature | analytical stress | `hybridht/mechanics.py` |
| V14 | Uniform temperature + transformation history | zero stress (exact) | `hybridht/mechanics.py` |
| V15–V16 | Full process chains | equilibrium, yield-surface consistency | `hybridht/mechanics.py` |
| V17 | Process outputs at nx and 2 nx | mesh sensitivity < 2 % | `hybridht/cases.py` |
| P5 | Carbonitriding vs induction: hardness, retained austenite, residual stress | model only + sensitivity to assumed inputs | `hybridht/metallurgy.py`, `hybridht/mechanics.py` |

Results, plots and the acceptance criteria are in [`results/REPORT.md`](results/REPORT.md).

## Run it

```bash
sudo apt-get install freefem++        # Ubuntu 24.04: FreeFEM 4.13
pip install -r requirements.txt
make test     # regression tests, ~10 s
make verify   # verification + validation report without the PINN, ~2 min
make all      # including PINN training, mesh and sensitivity studies, ~35 min on a laptop CPU
```

The full toolchain, including OpenFOAM, LAMMPS, Quantum ESPRESSO, OpenCalphad and PRISMS-PF, is
in `environment/Dockerfile`.

## Honesty rules for data

- Experimental comparisons use only numbers taken from a cited source, with the figure/table
  and the digitization method recorded (`data/published/README.md`). Tests fail if a file in
  `data/published/` is marked synthetic.
- The PINN inverse case uses **synthetic** measurements (FreeFEM + 1 % noise), labelled as such
  everywhere. It demonstrates the calibration workflow; real calibration needs real profiles.
- Assumed parameters (for example the mass-transfer coefficient β when a paper does not report
  it) are flagged in the report and covered by a sensitivity sweep.

## Next milestones

1. Digitize the candidate datasets listed in `data/published/README.md`, starting with the
   20MnCr5 β/D study and the 42CrMo4 induction case.
2. ~~Multi-stage boost/diffuse carburizing and nitrogen co-diffusion (carbonitriding).~~ Done (V6–V9, P3);
   still missing: C–N thermodynamic interaction.
3. ~~Coupled induction heating with μ(T), σ(T).~~ Done in 1D (V10–V12, P4); next: axisymmetric, then a
   2D tooth geometry, temperature-dependent k and cp (latent heat), μ(H) saturation.
4. ~~Phase kinetics → hardness; thermo-elasto-plasticity with TRIP → residual stress.~~ Done for the
   plate layer model (V13–V17, P5): martensite only, as quenched. Next: diffusional transformations
   from CCT/JMAK data, tempering, and the FreeFEM 2D tooth model for distortion.
5. OpenFOAM quench HTC map and OpenCalphad Ms/phase data, once a Fe-C-N-Cr-Mo-Ni TDB is secured.
6. Neural-operator surrogate over process parameters; factory-side calibration and dashboard.
