# Drakos+2020 N-body reference data

Source: Drakos, Taylor & Benson 2020 — *Mass-loss in tidally stripped systems:
the energy-based truncation method* (MNRAS 494, 378; arXiv:2003.09452). Digitized
reference for comparing a tidal stripping model against their GADGET-2 N-body
mass-loss curves. The followup Drakos+2022 (arXiv:2207.14803) extends the model
to Hernquist/Einasto/King satellites; its NFW fast/slow orbits are Sim 3/4 here,
so this NFW set already covers the Drakos+2022 NFW case.

There is no public raw N-body release ("data available on reasonable request"),
so the bound-mass curves are recovered from the vector figure in the arXiv
e-print. Fig. 8 (`Plots/MassLossPredictionEta.pdf`) plots `M/Msat` vs `t/torb`
for the ten NFW satellite orbits; the simulation points are dense open circles
(one colour per panel, ~51 per panel) and the black lines are the three `eta`
mass-loss models. Each circle is a vector path, so its centre is recovered
essentially exactly as a path-vertex centroid — this is not raster digitization.
`scripts/drakos20_nbody_massloss.py` regenerates the CSVs from the e-print.

## The satellites and host

All ten orbits use the *same* satellite: an NFW halo with concentration `c = 10`,
truncated at `rcut = rvir = 10 rs` (ICICLE initial conditions, unbound particles
removed — close to a lowered NFW with tidal energy `ET ≈ 0.28`), `N ≈ 1.29e6`
particles. Units are the satellite's own: `Msat`, `rs`, `vunit = sqrt(G Msat/rs)`,
`tunit = sqrt(rs^3 / G Msat)`.

The host is a fixed NFW potential with `c_host = 10` and the same mean density
within its virial radius as the satellite (a merger at fixed redshift), so
`Rvir_host/rs = 10 (Mhost/Msat)^(1/3)`. Only ratios enter the dynamics, so the
absolute mass/overdensity are free when instantiating the halos; matching
`c = 10` for both, the mass ratio, and the orbit reproduces the runs.

## Files

- `mbound_S1.csv` … `mbound_S10.csv` — bound mass vs time for orbits S1…S10.
  Columns: `t_over_torb, mbound_over_msat, sigma_rel`. `t_over_torb` is time in
  radial orbital periods (`torb` from `orbits.csv`); `mbound_over_msat` is the
  self-bound mass over the initial satellite mass, anchored to 1 at `t = 0`.
  `sigma_rel` is the Poisson bound-count shot noise `1/sqrt(N0 · m/Msat)` with
  `N0 = 1.29e6`, floored at 2% for the definitional ambiguity in the bound mass
  (temporarily-bound outer material, Drakos+2020 Sec. 6.2).

- `orbits.csv` — Drakos+2020 Table 1: per-orbit host + orbit parameters.
  Columns: `sim, Mhost_over_Msat, Rvir_over_rs, ra_over_rs, rp_over_rs,
  va_over_vunit, torb_over_tunit, circularity, eta_rel, Rc_over_Rvir`.
  Sim 3 is the paper's "Fast" simulation, Sim 4 the "Slow" one. Orbits are
  set at apocenter (`r = ra`, `v_r = 0`, `v_phi = va`). S9/S10 have mass ratio
  10 (large satellite): the distant-tide approximation breaks and Drakos's own
  model fails there too — kept as stress tests.

## Regenerating

```
python scripts/drakos20_nbody_massloss.py
```

Downloads the arXiv:2003.09452 e-print into `.cache/` (not committed), extracts
Fig. 8, and rewrites the CSVs. Needs `pymupdf` (`pip install pymupdf`); only
required to regenerate, not to read the committed CSVs.
