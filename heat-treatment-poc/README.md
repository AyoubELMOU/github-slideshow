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

Results, plots and the acceptance criteria are in [`results/REPORT.md`](results/REPORT.md).

## Run it

```bash
sudo apt-get install freefem++        # Ubuntu 24.04: FreeFEM 4.13
pip install -r requirements.txt
make test     # regression tests, ~10 s
make verify   # verification + validation report without the PINN, ~2 min
make all      # including PINN training, ~10 min on a laptop CPU
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
2. Multi-stage boost/diffuse carburizing and nitrogen co-diffusion (carbonitriding).
3. Coupled induction heating: eddy currents + heat with μ(T), σ(T), on an axisymmetric then a 2D
   tooth geometry.
4. Phase kinetics (Leblond/JMAK, Koistinen–Marburger with Ms(C, N)) → hardness, then
   thermo-elasto-plasticity with TRIP → residual stress and distortion.
5. OpenFOAM quench HTC map and OpenCalphad Ms/phase data, once a Fe-C-N-Cr-Mo-Ni TDB is secured.
6. Neural-operator surrogate over process parameters; factory-side calibration and dashboard.
