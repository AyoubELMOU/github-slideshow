# Reproducible toolchain for the hybrid FEA + PINN heat-treatment proof of concept.
# Ubuntu 24.04 packages provide FreeFEM, OpenFOAM, LAMMPS, Quantum ESPRESSO and deal.II;
# OpenCalphad and PRISMS-PF are built from source.
#
# Status: the FreeFEM + Python layer is exercised in CI (.github/workflows/heat-treatment-poc.yml).
# The OpenCalphad / PRISMS-PF build stages have not been run in CI yet.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake gfortran git ca-certificates \
        python3 python3-pip python3-venv \
        freefem++ \
        openfoam \
        lammps \
        quantum-espresso \
        libdeal.ii-dev \
    && rm -rf /var/lib/apt/lists/*

# OpenCalphad (thermodynamics). Needs a TDB database for Fe-C-N-Cr-Mo-Ni: not bundled.
RUN git clone --depth 1 https://github.com/sundmanbo/opencalphad.git /opt/opencalphad \
    && cd /opt/opencalphad && make -f Makefile 2>/dev/null || echo "OpenCalphad: build manually, see /opt/opencalphad/README"

# PRISMS-PF (phase field, on deal.II).
RUN git clone --depth 1 https://github.com/prisms-center/phaseField.git /opt/prisms-pf \
    && cd /opt/prisms-pf && cmake . && make -j"$(nproc)" || echo "PRISMS-PF: build manually"

WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
ENV PATH=/opt/venv/bin:$PATH
