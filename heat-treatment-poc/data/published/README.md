# Published experimental data

Put one CSV per measured profile here, in the format documented in
[`hybridht/published.py`](../../hybridht/published.py). `scripts/run_all.py` picks up
every file automatically, simulates the stated process with FreeFEM and reports RMSE,
bias and max error against the measurement.

Rules:

- **Never** add numbers that were not taken from the cited source. Record the figure/table
  and the digitization method (tool, number of repeats) so the uncertainty is traceable.
- Synthetic or test profiles do not belong here; tests generate their own.
- If the paper does not report the mass-transfer coefficient, omit `beta_m_per_s`; the
  report flags the comparison as using an assumed value.

## Candidate open-access sources to digitize first

Found during set-up. They have not been read in full yet: the session's network policy
blocked the publisher sites. Check each one's licence and data before using it.

| Use | Source |
|---|---|
| β and D_C for 20MnCr5 and other steels at 930 °C (gas carburizing) | *Elevating carbon diffusion: deciphering the interplay of alloy composition and carburizing treatment in low carbon steels*, Cogent Engineering (2024), doi:10.1080/23311916.2024.2345847 |
| Measured vs simulated carburized layer, 20MnCr5 | *Verification of the Simulated Carburizing Process in Different Bore Sizes*, Coatings 13(6) 1019 (2023), https://www.mdpi.com/2079-6412/13/6/1019 |
| Carburizing + quench distortion, measured vs predicted | *Predictions and Experiments on the Distortion of the 20Cr2Ni4A C-ring during Carburizing and Quenching Process*, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9230426/ |
| Induction hardening of 42CrMo4 cylinders, hardened case vs prediction (31 kHz, 441 A, 8 s, water quench) | *Predicting the induction hardened case in 42CrMo4 cylinders*, https://www.diva-portal.org/smash/get/diva2:1657790/FULLTEXT01.pdf |
| Hardness profiles of induction-hardened 4140 rods (0.75–3 mm cases) | Iowa State thesis, *Assessment of depth of case-hardening in steel rods by electromagnetic methods*, https://dr.lib.iastate.edu/server/api/core/bitstreams/44c00b06-4216-4525-bcab-d06e89f17d11/content |
