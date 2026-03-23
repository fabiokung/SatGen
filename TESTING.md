# SatGen: Testing & Validation

> Living document — update as new checks are added or reference values are refined.

---

## Fast dev/test cycle

The full cluster-scale pipeline (TreeGen + SatEvo at log M ≈ 14) can take hours. Use the tiers below to keep iteration fast.

### Tier 1 — Unit tests (~5 sec)

No tree files needed. Tests `evolve.py` functions directly.

```bash
source .venv/bin/activate
python -m pytest test_evolve_unit.py -v
```

Covers: `g_P10`, `g_EPW18`, `ltidal`, `msub`, `Dekel2`.

### Tier 2 — Integration test, MW-scale trees (~5–10 min total)

Generate the test fixture once (only needed when tree format changes):

```bash
source .venv/bin/activate
python TreeGen.py --ntree 4 --lgM0_lo 12.0 --lgM0_hi 12.1 --outdir test_data
```

This writes `test_data/tree{0..3}_lgM12.0?.npz`. Run SatEvo on all 4 (~5 min):

```bash
python SatEvo.py --datadir test_data/ --outdir test_data/sat_out/
```

Then validate output:

```bash
python scripts/check_output.py test_data/sat_out/tree*_lgM12.*.npz
```

### Tier 3 — Single-satellite tidal track plot (~10 sec)

```bash
python test_evolve.py          # existing script, requires display / Qt5Agg
```

For headless environments, save plots to file (edit `mpl.use('Agg')` at top of script).

---

## Validation suite

Run all diagnostics on any SatEvo output file:

```bash
source .venv/bin/activate
python scripts/check_output.py     <sat_output.npz> [...]   # self-consistency (one or more files)
python scripts/plot_tidal_tracks.py                          # compare to paper fits
python scripts/plot_shmf.py        <sat_output.npz> [...]   # SHMF power law
python scripts/plot_size_mass.py   <sat_output.npz> [...]   # size-mass relation
```

---

## Check descriptions and pass criteria

### check_output.py — self-consistency

| Check | Pass criterion |
|-------|---------------|
| No negative masses | All `mass[order>0] > 0` |
| Monotonic mass loss post-infall | `mass[id,iz] ≤ mass[id,iz+1]` for all evolved branches |
| Subhalo mass fraction | `fsub = Σm_sub(m>1e-4 M_host) / M_host ∈ [0.01, 0.20]` for MW-scale host |
| No surviving satellites above host mass | `max(mass_sub) < mass_host` at every snapshot |
| StellarMass ≤ DarkMatterMass | `StellarMass[id,iz] ≤ mass[id,iz]` everywhere |

### plot_tidal_tracks.py — calibration against N-body fits

Plots `g_P10` and `g_EPW18` as functions of bound mass fraction and overlays the reference values from the original papers.

**Pass:** Implementation curves within ±10% of paper values across the full range x ∈ [10⁻³, 1].

References:
- Penarrubia+10, Fig 5 — `g_P10` for NFW (α=1) and cuspy (α=1.5)
- Errani, Penarrubia & Walker 2018, Fig 3 — `g_EPW18`

### plot_shmf.py — subhalo mass function

Plots `dN/d ln(m/M_host)` at z=0 for all surviving subhalos. Fits a power law.

**Pass:** Slope ∈ [−2.0, −1.8] over at least 2 decades in mass ratio above the resolution limit.

References: Springel+08 (Aquarius), Garrison-Kimmel+14 (ELVIS)

### plot_size_mass.py — size-mass relation

Plots `R_eff` vs `M_star` at z=0 for all satellites with `M_star > 0`.

**Pass:** Median relation within ±0.5 dex of McConnachie+12 observed Local Group dwarfs across `M_star ∈ [10⁵, 10¹⁰] M_sun`.

**Known limitation at low masses:** The Jiang+19 size formula (`R_eff = 0.02 (c/10)^-0.7 R_vir`, eq. 6) overestimates sizes by ~0.5–1 dex for ultra-faint dwarfs (log M_star < 7). It was calibrated for log M_halo ≳ 11 and is not expected to match MC12 ultra-faints. Additionally, the low-M_star objects in SatEvo output are stripped remnants of larger halos whose R_eff was set at infall (~0.2–0.5 kpc) and barely decreases during stripping (EPW18 tidal puffing), making the comparison with intrinsically-small observed dwarfs structurally unfair at the low-mass end.

Reference: McConnachie+12, VizieR catalog J/AJ/144/4

---

## Additional validation (requires larger runs)

These require the full cluster-scale output and are run manually:

| Check | Reference | Criterion |
|-------|-----------|-----------|
| Radial distribution of subhalos | Springel+08 Fig 7 | `n_sub(r)` flatter than DM inside `r_vir/4` |
| Satellite luminosity function | McConnachie+12 + Drlica-Wagner+20 | N(M_V < −8) ~ 10–50 within 300 kpc for MW host (completeness-corrected) |
| V_max function | Klypin+11, Garrison-Kimmel+14 | `N(>V_max) ∝ V_max^{−3}` approximately |

---

## Reference data sources

| Dataset | Location | Use |
|---------|----------|-----|
| Penarrubia+10 tidal tracks | Table 1 / Fig 5 of paper | Verify `g_P10` in evolve.py |
| Errani+18 tidal tracks | Fig 3 of paper | Verify `g_EPW18` in evolve.py |
| McConnachie+12 | VizieR J/AJ/144/4 | Size-mass, luminosity function |
| Aquarius (Springel+08) | VizieR / paper tables | SHMF slope, radial distribution |
| ELVIS (Garrison-Kimmel+14) | arXiv:1310.6746 supplemental | SHMF, V_max function |
| Drlica-Wagner+20 (DES Y3) | arXiv:1912.03302 | Completeness-corrected satellite LF |
