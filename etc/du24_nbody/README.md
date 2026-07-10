# Du+24 N-body reference data

Source: Du, Benson, Treu et al. 2024 — *Tidal evolution of cored and cuspy dark
matter halos* (arXiv:2403.09597). These are the authors' raw idealized-N-body
outputs, shipped with the Galacticus source tree at
`testSuite/data/idealizedSubhaloSimulations/`. Only the two gamma_subhalo = 1
(NFW: alpha=1, beta=3, gamma=1) orbits are copied here.

## Raw data

- `simulationData_xc_0.7_ratio_0.05_alpha_1.0_beta_3.0_gamma_1.0.hdf5` — R_p/R_a = 1/20
- `simulationData_xc_0.7_ratio_0.2_alpha_1.0_beta_3.0_gamma_1.0.hdf5` — R_p/R_a = 1/5

Each holds a 381-sample time series (`time`, `boundMass`, `velocityMaximum`,
`radiusVelocityMaximum` and their `*Error` estimates), the apo/peri-centre
snapshot indices, and the initial NFW halo/host structural scalars. Subhalo
M_vir = 1e9, host M_vir = 1e12, subhalo concentration c ≈ 20.6.

## Derived CSVs

Regenerate with `python scripts/du24_nbody_reference.py`. All are sampled at
apocenters, where the subhalo is relaxed — V_max/R_max oscillate between peri
and apo, so the clean tidal-track relation and the mass-loss plateaus both live
at the apocenters (Du+24 Fig. 12 / Fig. 10). Errors are the N-body's own
measurement uncertainties, propagated to the ratios.

- `track_1_5.csv`, `track_1_20.csv` — V_max/V_max,0 and R_max/R_max,0 vs
  M_bound/M_bound,0. Columns: `x, V_over_V0, R_over_R0, sigma_V_rel, sigma_R_rel`.
- `mbound_t_1_20.csv` — M_bound/M_bound,0 at apocenters, R_p/R_a = 1/20, vs
  **orbit number** k = t/T_r (apocenter k is the k-th orbit). Columns:
  `orbit, mbound_over_m0, sigma_rel`. The period-normalized axis lets the model's
  k-th apocenter be compared to the N-body's, immune to the SatGen-vs-N-body radial-
  period offset and the N-body's orbital decay. The physical apocenter times in Gyr
  live in the HDF5.
- `mbound_t_1_5.csv` — same for R_p/R_a = 1/5. Held out of the calibration
  likelihood as a generalization test (`load_du24_test_observables`).

The earlier hand-digitized CSVs were within ~3% (tidal tracks, from the Du+24
eq. 19 fit) but off by up to a factor ~1.6 point-to-point on the 1/20 mass-loss
curve; these raw curves replace them everywhere.
