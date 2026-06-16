# Du+24 N-body reference data

Source: Du, Benson, Treu et al. 2024 — *Tidal evolution of cored and cuspy dark
matter halos* (arXiv:2403.09597). Digitized reference for comparing a tidal
stripping/heating model against Du+24's NFW-subhalo N-body tracks.

All data is for the gamma_subhalo = 1 (NFW) inner-slope row (alpha=1, beta=3,
gamma=1). The 2/5 pericenter ratio is *not* in Du+24 for gamma=1 — only 1/5
and 1/20.

## Files

- `track_1_5.csv`, `track_1_20.csv` — V_max/V_max,0 and R_max/R_max,0 as
  functions of M_bound/M_bound,0 for each orbit. Columns:
  `x, V_over_V0, R_over_R0, sigma_V_rel, sigma_R_rel`. The V/V0 and R/R0
  values come from Du+24 eq. 19 (their published NFW-subhalo tidal-track fit)
  evaluated at typical Fig. 12 row 3 x samples; that fit goes cleanly through
  all N-body points in Fig. 12 row 3, so it's a faithful proxy for the average
  N-body behavior. Sigmas are 1/sqrt(N_bound) with N0 = 1e7 initial particles
  (Du+24 quote 1e4 bound particles at the M_bound/M0 = 1e-3 resolution floor).

- `mbound_t_1_20.csv` — M_bound(t) at apocenter plateaus for R_p/R_a = 1/20.
  Columns: `t_Gyr, mbound_over_m0, sigma_rel`. Hand-digitized from the green
  curve in Du+24 Fig. 10 by reading off the M_bound value at each visible
  plateau between peri passages (~12 apocenters over 37 Gyr). Sigmas as above.

## Replacing with WebPlotDigitizer output

Both files are pragmatic v1 estimates. To swap in raw N-body digitization from
Du+24 Fig. 12 row 3 and Fig. 10:
1. Run WebPlotDigitizer against the Du+24 paper figures (arXiv:2403.09597).
2. Extract per-orbit point lists with the same column structure.
3. Overwrite the CSV — downstream code reads only the column names.
