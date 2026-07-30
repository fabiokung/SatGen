# Aguirre-Santaella+25 N-body reference data

Source: Aguirre-Santaella, Sanchez-Conde & Ogiya 2025 — *New insights on
low-mass dark matter subhalo tidal tracks via numerical simulations*
(arXiv:2506.01152; MNRAS 545, 2). Idealized, high-resolution N-body runs of a
single live NFW subhalo orbiting a Milky-Way host (NFW halo + stellar disc +
gas disc + bulge, time-evolving potential), run with an upgraded GPU version of
the DASH code. Complements the DASH library (Ogiya+19), which uses a pure-NFW
spherical host: here the host carries the full MW baryonic disc, and the runs
reach down to f_b ~ 1e-3 with N = 2^25 particles.

24 runs spanning concentration c in {5,10,15,20,30}, orbital energy x_c in
{0.8,1.0,1.2,1.4,1.6}, circularity eta in {0.1..0.8}, accretion redshift z_acc
in {1,1.5,2,2.5,3,4}, and orbit inclination theta in {0,45,90} deg. All
m_sub = 1e6 Msun. All inner slope gamma = 1 (NFW); the paper's prompt-cusp
(gamma = 1.5) runs are not in this release.

## Download

Not committed (41 MB extracted). Pull the archive from the DAMASCO group page:

    curl -sL -o etc/aguirre25_nbody/tidal_tracks_subhaloes.zip \
      https://saco.csic.es/s/EP7Y3YJRYS7DTRo/download

`scripts/aguirre25_nbody_tracks.py` reads the runs straight out of the zip; no
need to extract.

## Run directories

Folder name encodes the initial conditions, `p` being the decimal point, e.g.
`fulldc10x0p8e0p1N25z1m1e6i90...`:

- `c10`  concentration c = 10
- `x0p8` orbital energy x_c = 0.8
- `e0p1` circularity eta = 0.1
- `N25`  particle number 2^25
- `z1`   accretion redshift z_acc = 1
- `m1e6` subhalo mass 1e6 Msun
- `i90`  orbit inclination theta = 90 deg

The `full[d|s]` prefix is an internal batch tag, not a physical distinction:
both batches are NFW subhaloes in the same disc+bulge MW host (confirmed by
inner density slope ~ -1 and identical disc-driven orbital-plane precession).

## Files per run

- `radprof_NNN.txt` — radial profile at snapshot NNN (101 snapshots, 000..100).
  Columns: `r, rho, M(<r)` (r in units of the initial subhalo virial radius;
  further columns are velocity-dispersion / anisotropy diagnostics we don't
  use). V_max/r_max come from V_circ(r) = sqrt(M(<r)/r).
- `subhalo_evolution.txt` — 101-row time series aligned to the snapshots.
  Columns: `t_lookback, z, X[3]/Rhost, V[3]/sqrt(Mhost/Rhost), f_b, R_h/R_sub`.
- `host_params.txt`, `energy.txt` — host evolution and energy bookkeeping
  (unused here).

## Processed tracks

`python scripts/aguirre25_nbody_tracks.py` builds V_max/r_max tidal tracks and
writes `etc/calibration_runs/aguirre25_nbody_tracks.pkl`, keyed by run id, each
a dict(params, t_lookback, z, fb, V=V_max/V_max,0, R=r_max/r_max,0). Tracks are
unreliable below f_b ~ 10^-3.5 (~3000 particles in r_max); the floor is stored
as `fb_floor` in the pickle.
