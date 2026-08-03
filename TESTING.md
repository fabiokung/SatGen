# SatGen: Testing & Validation

Living document — update as checks are added or reference values refined.

The full cluster-scale pipeline (TreeGen + SatEvo at log M ~ 14) takes hours. The three
tiers below keep iteration fast.

## Tier 1 — unit tests (~3 min, no tree files)

```bash
python -m pytest --ignore=test_evolve.py    # whole suite, 240 tests
python -m pytest test_evolve_unit.py -v     # just evolve.py
```

`test_evolve.py` is the Tier-3 plotting script below, not a pytest module — it imports
a Qt backend, so a bare `pytest` fails at collection. Always ignore it.

`test_evolve_unit.py` covers `g_P10`, `g_EPW18`, `ltidal`, `msub`, `Dekel2`. The rest of
the suite covers the truncations and the heating/stripping engines
(`test_stripping.py`, `test_stripping_truncation.py`, `test_subhalo_functions.py`,
`test_dash_setup.py`).

## Tier 2 — integration, MW-scale trees (~5-10 min)

```bash
python SatEvo.py --datadir test_data/ --outdir test_data/sat_out/
python scripts/check_output.py test_data/sat_out/tree*_lgM12.*.npz
```

Fixtures `test_data/tree{0..3}_lgM12.0?.npz` are committed; regenerate only when the
tree format changes:

```bash
python TreeGen.py --ntree 4 --lgM0_lo 12.0 --lgM0_hi 12.1 --outdir test_data
```

## Tier 3 — single-satellite tidal track (~10 sec)

```bash
python test_evolve.py    # requires display / Qt5Agg; edit mpl.use('Agg') for headless
```

## Validation suite

Runs on any SatEvo output:

```bash
python scripts/check_output.py    <sat_output.npz> [...]   # self-consistency
python scripts/plot_tidal_tracks.py                        # compare to paper fits
python scripts/plot_shmf.py       <sat_output.npz> [...]   # SHMF power law
python scripts/plot_size_mass.py  <sat_output.npz> [...]   # size-mass relation
```

### check_output.py — self-consistency

| Check | Pass criterion |
|-------|---------------|
| No negative masses | All `mass[order>0] > 0` |
| Monotonic mass loss post-infall | `mass[id,iz] <= mass[id,iz+1]` on all evolved branches |
| Subhalo mass fraction | `fsub = sum m_sub(m>1e-4 M_host) / M_host` in [0.01, 0.20] for a MW host |
| No satellite above host mass | `max(mass_sub) < mass_host` at every snapshot |
| StellarMass <= DarkMatterMass | `StellarMass[id,iz] <= mass[id,iz]` everywhere |

### plot_tidal_tracks.py — calibration against N-body fits

`g_P10` and `g_EPW18` vs bound mass fraction, overlaid on the original papers'
reference values (Penarrubia+10 Fig. 5 for NFW alpha=1 and cuspy alpha=1.5;
Errani, Penarrubia & Walker 2018 Fig. 3).

**Pass:** within +/-10% of paper values across x in [1e-3, 1].

### plot_shmf.py — subhalo mass function

`dN/d ln(m/M_host)` at z=0 over surviving subhalos, power-law fit.

**Pass:** slope in [-2.0, -1.8] over at least 2 decades in mass ratio above the
resolution limit. References: Springel+08 (Aquarius), Garrison-Kimmel+14 (ELVIS).

### plot_size_mass.py — size-mass relation

`R_eff` vs `M_star` at z=0 for satellites with `M_star > 0`.

**Pass:** median relation within +/-0.5 dex of McConnachie+12 (VizieR J/AJ/144/4) over
`M_star` in [1e5, 1e10] M_sun.

**Known limitation at low masses.** The Jiang+19 size formula
(`R_eff = 0.02 (c/10)^-0.7 R_vir`, eq. 6) overestimates sizes by ~0.5-1 dex for
ultra-faint dwarfs (log M_star < 7); it was calibrated for log M_halo >~ 11. The
low-M_star objects in SatEvo output are also stripped remnants of larger halos whose
R_eff was set at infall (~0.2-0.5 kpc) and barely decreases during stripping (EPW18
tidal puffing), so the comparison against intrinsically-small observed dwarfs is
structurally unfair at that end.

## Additional validation (requires cluster-scale runs, done manually)

| Check | Reference | Criterion |
|-------|-----------|-----------|
| Radial distribution of subhalos | Springel+08 Fig 7 | `n_sub(r)` flatter than DM inside `r_vir/4` |
| Satellite luminosity function | McConnachie+12 + Drlica-Wagner+20 (arXiv:1912.03302) | N(M_V < -8) ~ 10-50 within 300 kpc, completeness-corrected |
| V_max function | Klypin+11, Garrison-Kimmel+14 (arXiv:1310.6746) | `N(>V_max)` ~ `V_max^-3` |
