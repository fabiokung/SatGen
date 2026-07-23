"""
Integration checks for tidal stripping on Du+24's radial 1/20 orbit.

The King62 rate strips on the subhalo dynamical time T_dyn,sub(l_t) while the
tidal-tensor cumulant accumulates on the host orbital time (Du+24 eqs. 35,
38-39). A subhalo denser than the tidal field out to its edge has its tidal
radius beyond the profile (l_t = rh) and is not stripped, so a dense core on the
radial orbit strips down to a bound remnant rather than disrupting -- under the
hard cut and both tail prescriptions.

Slower than the Tier-1 unit tests (~30 s) -- runs short evolve_heating
integrations. Run with:
    source .venv/bin/activate
    python -m pytest test_stripping_truncation.py -v
"""

import numpy as np
import pytest
from scipy.integrate import quad

import config as cfg
from profiles import NFW
from subhalo_functions import NumericProfile, truncate_kazantzidis
from orbit import orbit
import stripping_common as sc


@pytest.fixture
def du24_setup():
    """Du+24 Section II.A setup on the radial Rp/Ra = 1/20 orbit."""
    original = cfg.Mres
    cfg.Mres = 1e3
    hNFW = NFW(1.0e12, 263.2 / 23.69)
    sat = NFW(1.0e9, 26.32 / 1.279)
    rvals = np.logspace(np.log10(cfg.Rres), np.log10(sat.rh), 200)
    M_sub = sat.M(rvals)
    xv0_20, _ = sc.make_orbit(hNFW, R0=0.7 * hNFW.rh, z0=0., eta=0.131)
    yield hNFW, rvals, M_sub, xv0_20
    cfg.Mres = original


def _run(setup, truncation, second_order=False, engine='uniform'):
    """Run one short evolution on the 1/20 orbit through either engine. The uniform
    grid takes Nstep; the revirialization scheme takes step_frac (dt <= step_frac *
    min(t_orb, T_strip/alpha)) and reshapes the profile once per apocenter."""
    hNFW, rvals, M_sub, xv0_20 = setup
    sub = NumericProfile(rvals, M_sub)
    if engine == 'revirial':
        return sc.evolve_heating_revirial(
            hNFW, sub, xv0_20, tmax=12., step_frac=0.02,
            epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
            second_order=second_order, truncation=truncation)
    return sc.evolve_heating(
        hNFW, sub, xv0_20, tmax=12., Nstep=12000,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        second_order=second_order, truncation=truncation)


# both engines evolve the same physics; the survival/track behaviour is asserted
# against both. Thresholds bracket both engines' converged values -- the corrected
# per-apocenter heating of the revirialization scheme strips a dense core somewhat
# less deeply than the uniform grid, so a shared ratio bound is looser than either
# engine alone (dt-robustness is pinned per-engine in the dedicated tests below).
def test_dens_hist_conserves_residence_and_localizes(du24_setup):
    """The dt-weighted residence histogram matches the bound-mass time integral,
    sits only at radii the track visits, and doesn't perturb the evolution."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    sub = NumericProfile(rvals, M_sub)
    redges = np.logspace(-4, np.log10(500.), 60)
    tedges = np.linspace(0., 12., 6)                     # 5 time chunks
    res = sc.evolve_heating_revirial(
        hNFW, sub, xv0_20, tmax=12., step_frac=0.02, epsh=0.0741, gamma=0.,
        beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt', truncation='hard',
        dens_redges=redges, dens_tedges=tedges)
    bare = sc.evolve_heating_revirial(
        hNFW, NumericProfile(rvals, M_sub), xv0_20, tmax=12., step_frac=0.02,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        truncation='hard')

    H = res.dens_hist
    assert H is not None and H.shape == (len(redges) - 1, len(tedges) - 1)
    assert (H >= 0.).all()
    # the accumulator must not change the evolution
    assert np.array_equal(res.t, bare.t) and np.array_equal(res.m, bare.m)
    assert np.array_equal(res.r, bare.r)
    # total residence = int m dt over the run (pre-strip m per fine step); compare
    # to the trapezoid of the recorded bound-mass track (from t=0, m=Mh)
    t_ref = np.concatenate([[0.], res.t])
    m_ref = np.concatenate([[float(sub.Mh)], res.m])
    assert H.sum() == pytest.approx(np.trapezoid(m_ref, t_ref), rel=0.05)
    # every occupied radial bin is one the recorded orbit actually passed through
    visited = np.zeros(len(redges) - 1, bool)
    visited[np.clip(np.searchsorted(redges, res.r) - 1, 0, len(redges) - 2)] = True
    assert np.all(~(H.sum(1) > 0) | visited)


def test_dens_kernels_weighted_residence(du24_setup):
    """dens_kernels adds one kernel-weighted histogram per callable w(r, v): the unit
    kernel reproduces dens_hist bin-for-bin, a speed kernel occupies the same bins with
    the dt-weighted speed average, and the accumulators don't perturb the evolution."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    redges = np.logspace(-4, np.log10(500.), 60)
    tedges = np.linspace(0., 12., 6)
    kw = dict(tmax=12., step_frac=0.02, epsh=0.0741, gamma=0., beta_h=0.278,
              alpha=3.93, t_dyn_mode='sub_lt', truncation='hard',
              dens_redges=redges, dens_tedges=tedges)
    res = sc.evolve_heating_revirial(
        hNFW, NumericProfile(rvals, M_sub), xv0_20,
        dens_kernels=[lambda r, v: 1., lambda r, v: v], **kw)
    bare = sc.evolve_heating_revirial(hNFW, NumericProfile(rvals, M_sub), xv0_20, **kw)

    Hk = res.dens_hist_k
    assert Hk is not None and Hk.shape == (2, len(redges) - 1, len(tedges) - 1)
    assert np.array_equal(Hk[0], res.dens_hist)          # unit kernel == dens_hist
    assert (Hk[1] >= 0.).all()
    assert np.array_equal((Hk[1] > 0), (res.dens_hist > 0))
    # ratio = dt m-weighted mean orbital speed; bounded by the central escape speed
    vbar = Hk[1].sum() / res.dens_hist.sum()
    assert 0. < vbar < np.sqrt(-2. * hNFW.Phi(redges[0]))
    # the kernel accumulators must not change the evolution
    assert np.array_equal(res.t, bare.t) and np.array_equal(res.m, bare.m)
    assert bare.dens_hist_k is None


@pytest.mark.parametrize('engine', ['uniform', 'revirial'])
def test_hard_cut_survives_on_radial_orbit(du24_setup, engine):
    """A dense NFW subhalo on the radial orbit strips to a bound remnant, not to the
    mass floor: once the remnant is denser than the tidal field out to its edge its
    tidal radius exceeds the profile (l_t = rh) and stripping halts. Guards the
    tidal-radius logic against over-stripping dense cores, under the hard cut at the
    Du+24 Table IV rate (alpha=3.93, the most aggressive stripping we run)."""
    res = _run(du24_setup, 'hard', engine=engine)
    m = res.m[np.isfinite(res.m)]
    assert np.all(m > cfg.Mres), "a dense core must not be stripped to the floor"
    assert m[-1] / m[0] < 0.15         # heavily stripped but bound
    assert m[-1] / cfg.Mres > 100      # remnant stays well above the mass floor


@pytest.mark.parametrize('engine', ['uniform', 'revirial'])
def test_kazantzidis_tail_survives(du24_setup, engine):
    """The Kazantzidis l_t tail with the second-order term keeps the subhalo bound,
    with a finite Vmax, out to late times on the radial orbit."""
    res = _run(du24_setup, 'kazantzidis', second_order=True, engine=engine)
    m = res.m[np.isfinite(res.m)]
    assert np.all(m > cfg.Mres), "kazantzidis run should not hit the mass floor"
    late = np.isfinite(res.t) & (res.t > 9.)
    assert np.sum(late) > 0
    assert np.all(np.isfinite(res.vmax[late])), \
        "the Kazantzidis remnant keeps a resolved Vmax peak at late times"


@pytest.mark.parametrize('engine', ['uniform', 'revirial'])
def test_powerlaw_tail_survives_at_calibrated_params(du24_setup, engine):
    """The slope-deficit power-law tail keeps the subhalo bound with a finite
    Vmax through 12 Gyr on the 1/20 orbit at the MCMC-calibrated (p50)
    stripping rate -- the regime the model operates in.

    Not run at the Du+24 Table IV rate (alpha=3.93) like the Kazantzidis
    test: that rate over-strips, the King62 budget then sits below the tail's
    minimum mass 4 pi rho_t lt^3/(n-3) (an r^-n envelope cannot carry less)
    on ~90% of steps, and the fallback hard cut disrupts the subhalo just as
    truncation='hard' does."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    sub = NumericProfile(rvals, M_sub)
    if engine == 'revirial':
        res = sc.evolve_heating_revirial(
            hNFW, sub, xv0_20, tmax=12., step_frac=0.02,
            epsh=0.0708, gamma=0.0094, beta_h=0.906, alpha=0.171, f2=0.188,
            t_dyn_mode='sub_lt', second_order=True, truncation='powerlaw')
    else:
        res = sc.evolve_heating(
            hNFW, sub, xv0_20, tmax=12., Nstep=12000,
            epsh=0.0708, gamma=0.0094, beta_h=0.906, alpha=0.171, f2=0.188,
            t_dyn_mode='sub_lt', second_order=True, truncation='powerlaw')
    m = res.m[np.isfinite(res.m)]
    assert np.all(m > cfg.Mres), "powerlaw run should not hit the mass floor"
    assert m[-1] / m[0] < 0.15, "the 1/20 orbit strips deeply, not frozen"
    late = np.isfinite(res.t) & (res.t > 9.)
    assert np.sum(late) > 0
    assert np.all(np.isfinite(res.vmax[late])), \
        "the powerlaw profile keeps a Vmax peak through deep stripping"


def test_clamp_track_aligned_to_evolution(du24_setup):
    """evolve_heating records per-step shell-clamp counts (res.clamp) and worst
    overshoot % (res.clamp_worst) on the result, aligned to the mass track, so
    clamping can be binned by stage / bound fraction post-hoc."""
    res = _run(du24_setup, 'powerlaw', second_order=True)
    assert res.clamp is not None and res.clamp_worst is not None
    assert res.clamp.shape == res.m.shape == res.clamp_worst.shape
    live = np.isfinite(res.m)
    # clamp telemetry is written exactly where the track is live, NaN elsewhere
    assert np.all(np.isfinite(res.clamp[live]))
    assert np.all(np.isnan(res.clamp[~live]))
    assert np.all(res.clamp[live] >= 0.)            # shells clamped: a count
    assert np.nansum(res.clamp) > 0                 # the 1/20 orbit does clamp


def test_revirial_clamp_telemetry_populated(du24_setup):
    """evolve_heating_revirial records shell-clamp telemetry per re-virialization
    (res.clamp shells, clamp_worst %, clamp_worst_r radius), aligned to the mass
    track: finite on the apocenter re-virialization steps, NaN elsewhere."""
    res = _run(du24_setup, 'kazantzidis', second_order=True, engine='revirial')
    for arr in (res.clamp, res.clamp_total, res.clamp_worst, res.clamp_worst_r):
        assert arr is not None and arr.shape == res.m.shape
    revir = np.isfinite(res.clamp)
    assert revir.sum() > 0                          # some re-virializations happened
    assert np.all(res.clamp[revir] >= 0.)           # shells clamped: a count
    assert np.nansum(res.clamp) > 0                 # kazantzidis tail clamps on 1/20
    # shells heated is the fraction denominator: >= shells clamped, and > 0 wherever
    # a reshape ran (the profile always heats its whole grid)
    assert np.all(res.clamp_total[revir] >= res.clamp[revir])
    assert np.all(res.clamp_total[revir] > 0.)
    fired = revir & (res.clamp > 0)                 # steps where a shell actually clamped
    assert np.all(np.isfinite(res.clamp_worst_r[fired]))
    assert np.all(res.clamp_worst_r[fired] > 0.)


def test_revirial_expand_clamp_discriminable_from_revir(du24_setup):
    """The per-step _ExpandedProfile clamp telemetry (expand_clamp*) is recorded
    separately from the apocenter reshape telemetry (clamp*), so the two clamp
    regimes can be told apart: the reshape telemetry is finite only on the
    (few) re-virialization steps, the expansion telemetry on (almost) every step.
    On the cuspy NFW subhalo neither actually clamps -- perturb rises monotonically
    outward, so heating only unbinds a contiguous outer block."""
    res = _run(du24_setup, 'hard', engine='revirial')
    for arr in (res.expand_clamp, res.expand_clamp_total,
                res.expand_clamp_worst, res.expand_clamp_worst_r):
        assert arr is not None and arr.shape == res.m.shape
    revir = np.isfinite(res.clamp)              # apocenter reshape steps
    expand = np.isfinite(res.expand_clamp)      # per-step expansion steps
    # the expansion runs far more often than the reshape (per step vs per orbit)
    assert expand.sum() > 5 * revir.sum()
    assert np.all(res.expand_clamp[expand] >= 0.)
    assert np.all(res.expand_clamp_total[expand] >= res.expand_clamp[expand])
    # cuspy subhalo: no shell crossing anywhere, only outer-block unbinding
    assert np.nansum(res.clamp) == 0
    assert np.nansum(res.expand_clamp) == 0


def test_expanded_profile_raises_when_heating_unbinds():
    """_ExpandedProfile raises HeatingUnbindsError when the accumulated heating
    leaves <=2 bound shells, mirroring heat_profile's catastrophic-heating guard."""
    from subhalo_functions import HeatingUnbindsError
    h = NFW(1e9, 11.68)
    ri = np.logspace(np.log10(1e-3 * h.rh), np.log10(h.rh), 200)
    prof = NumericProfile(ri, np.asarray(h.M(ri), float))
    r_ref = prof.ri
    M_ref = np.asarray(prof.M(r_ref), float)
    sig2 = np.zeros_like(r_ref)
    sc._ExpandedProfile(r_ref, M_ref, sig2, Q=1e-20, c2=0.)   # gentle: intact
    Q_huge = 1e10 * cfg.G * prof.Mh / prof.rh                 # unbinds the halo
    with pytest.raises(HeatingUnbindsError):
        sc._ExpandedProfile(r_ref, M_ref, sig2, Q=Q_huge, c2=0.)


def test_revirial_heating_unbinds_disrupts_gracefully(du24_setup):
    """Strong heating on a cored subhalo unbinds the analytic profile per step
    (_ExpandedProfile raises HeatingUnbindsError); evolve_heating_revirial catches
    it, drives the bound mass to the floor, and terminates without a crash, with
    the telemetry arrays still aligned to the mass track. Cored because a cusp
    keeps perturb -> 0 in the core, so only the outer block ever unbinds."""
    hNFW, rvals, _M_sub, xv0_20 = du24_setup
    sat = NFW(1.0e9, 26.32 / 1.279)
    rc = 0.5 * sat.rh                       # flatten the inner cusp into a core
    rho0 = sat.M(rc) / (4. / 3. * np.pi * rc ** 3)
    M_core = np.where(rvals < rc, 4. / 3. * np.pi * rvals ** 3 * rho0,
                      np.asarray(sat.M(rvals), float))
    sub = NumericProfile(rvals, M_core)
    res = sc.evolve_heating_revirial(
        hNFW, sub, xv0_20, tmax=12., step_frac=0.02, epsh=6., gamma=0.,
        beta_h=1., alpha=1., t_dyn_mode='sub_lt', truncation='hard',
        lt_choice='Tormen98')
    m = res.m[np.isfinite(res.m)]
    assert len(m) > 0
    assert m[-1] <= cfg.Mres * (1. + 1e-9)          # disrupted to the floor
    for arr in (res.clamp, res.expand_clamp, res.expand_clamp_worst_r):
        assert arr.shape == res.m.shape             # telemetry stays aligned


def _final_mass_ratio(setup, nstep, alpha=3.93):
    hNFW, rvals, M_sub, xv0_20 = setup
    res = sc.evolve_heating(
        hNFW, NumericProfile(rvals, M_sub), xv0_20,
        tmax=12., Nstep=nstep, epsh=0.0741, gamma=0., beta_h=0.278,
        alpha=alpha, t_dyn_mode='sub_lt', second_order=True,
        truncation='kazantzidis')
    m = res.m[np.isfinite(res.m)]
    return m.min() / m[0]


def test_king62_step_is_dt_robust(du24_setup):
    """The King62 step relaxes the bound mass toward M(<lt) and is integrated
    exactly (exp), so the deep-stripping asymptote is insensitive to dt:
    halving the step leaves it unchanged. A forward-Euler step overshoots the
    M(<lt) floor at the coarse dt -- the failure that froze stripping at a
    spurious high mass for isolated (orbit, dt)."""
    coarse = _final_mass_ratio(du24_setup, 6000)
    fine = _final_mass_ratio(du24_setup, 24000)
    assert coarse == pytest.approx(fine, rel=0.1), \
        f"deep-stripping mass must be dt-robust: {coarse:.4f} vs {fine:.4f}"
    assert fine < 0.01, "the 1/20 orbit strips deeply, not frozen near unity"


def test_strip_cfl_guard_raises_on_coarse_step(du24_setup):
    """alpha*dt/T_strip above STRIP_CFL_MAX raises -- past it exp(-x) sinks the
    stripped excess into floating-point noise, silently turning the King62 rate into
    a hard cut. A coarse step (Nstep=2000) at a high rate (alpha=50) crosses the
    threshold while the subhalo is actively stripping."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    with pytest.raises(ValueError, match=r"alpha\*dt/T_strip"):
        sc.evolve_heating(
            hNFW, NumericProfile(rvals, M_sub), xv0_20,
            tmax=12., Nstep=2000, epsh=0.0741, gamma=0., beta_h=0.278,
            alpha=50., t_dyn_mode='sub_lt', second_order=True,
            truncation='hard')


def test_r_stop_parks_and_early_terminates(du24_setup):
    """r_stop halts the integration once the orbit dips inside it: the last
    recorded slot sits below r_stop with the bound mass preserved (not floored),
    and the remaining slots stay NaN."""
    hNFW, rvals, M_sub, _ = du24_setup
    xv0, _ = sc.make_orbit(hNFW, R0=0.5 * hNFW.rh, z0=0., eta=0.3)  # eccentric
    r0 = np.hypot(xv0[0], xv0[2])
    rstop = 0.6 * r0                       # pericentre < r_stop < r0
    res = sc.evolve_heating(
        hNFW, NumericProfile(rvals, M_sub), xv0, tmax=12., Nstep=2000,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        truncation='hard', dynamical_friction=False, r_stop=rstop)
    fin = np.isfinite(res.r)
    assert fin.sum() < 2000                    # stopped before filling every slot
    assert res.r[fin][-1] < rstop              # terminal record is inside r_stop
    assert res.m[fin][-1] > cfg.Mres           # bound mass preserved, not floored
    assert not np.isfinite(res.r[fin.sum()])   # slots past the break stay NaN


def _cb_run(du24_setup, **cb):
    """One revirialization evolution on the du24 radial orbit with circuit-breaker kwargs.
    The lenient-floor tests pass freeze_strip near 1 so ordinary stripping never resets the
    streak, isolating the freeze mechanism from the (separate) not-stripping condition. This
    orbit is radial, so freeze_rspan defaults to 1.0 here -- disabling the settled-radius gate
    so the other criteria can be tested in isolation; the radial-span gate has its own test."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    cb.setdefault('freeze_rspan', 1.0)
    return sc.evolve_heating_revirial(
        hNFW, NumericProfile(rvals, M_sub), xv0_20, tmax=12., step_frac=0.02,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        truncation='hard', **cb)


def test_circuit_breaker_radial_orbit_not_frozen(du24_setup):
    """A radial orbit (deep pericentre passages) must NOT freeze even with the mass stable and
    the orbit tight: its wide min/max radial span fails the settled test, so it keeps
    integrating to register the rare central passages. Only a permissive freeze_rspan (>= its
    span) lets it freeze."""
    radial = _cb_run(du24_setup, freeze_strip=0.99, freeze_orbits=2, freeze_tdyn=1e9,
                     freeze_rspan=0.3)
    assert radial.frozen is False
    loose_span = _cb_run(du24_setup, freeze_strip=0.99, freeze_orbits=2, freeze_tdyn=1e9,
                         freeze_rspan=1.0)
    assert loose_span.frozen is True


def test_circuit_breaker_off_by_default(du24_setup):
    """Without freeze_strip the run is never frozen -- existing behaviour unchanged."""
    assert _cb_run(du24_setup).frozen is False


def test_circuit_breaker_freezes_settled_orbit(du24_setup):
    """A lenient strip floor never resets the streak, and a large freeze_tdyn makes every
    orbit count as tight, so the breaker fires after freeze_orbits re-virializations and
    stops the run well before tmax."""
    full = _cb_run(du24_setup)
    frz = _cb_run(du24_setup, freeze_strip=0.99, freeze_orbits=2, freeze_tdyn=1e9)
    assert frz.frozen is True
    assert frz.t[-1] < full.t[-1]                       # stopped early
    assert len(frz.t) < len(full.t)                     # fewer steps
    assert int(np.count_nonzero(frz.apo)) >= 2          # froze after >= freeze_orbits revirs


def test_circuit_breaker_strip_resets_streak(du24_setup):
    """A strict strip floor on a stripping orbit keeps resetting the streak, so the run
    evolves far longer than the lenient floor that freezes at freeze_orbits."""
    lenient = _cb_run(du24_setup, freeze_strip=0.99, freeze_orbits=2, freeze_tdyn=1e9)
    strict = _cb_run(du24_setup, freeze_strip=1e-9, freeze_orbits=2, freeze_tdyn=1e9)
    assert lenient.frozen is True
    assert strict.t[-1] > lenient.t[-1]


def test_circuit_breaker_tight_gate(du24_setup):
    """The mass streak alone does not freeze -- the orbit must be tight (apocenter t_dyn <
    freeze_tdyn). A tiny freeze_tdyn (never tight, no wall-clock backstop) does not freeze;
    a large one does."""
    loose = _cb_run(du24_setup, freeze_strip=0.99, freeze_orbits=2, freeze_tdyn=1e-9)
    tight = _cb_run(du24_setup, freeze_strip=0.99, freeze_orbits=2, freeze_tdyn=1e9)
    assert loose.frozen is False
    assert tight.frozen is True


def test_circuit_breaker_walltime_backstop(du24_setup):
    """With no tight test, the wall-clock backstop still freezes once the mass streak is
    met: any elapsed time exceeds a zero-second budget."""
    res = _cb_run(du24_setup, freeze_strip=0.99, freeze_orbits=2,
                  freeze_tdyn=None, freeze_walltime=0.)
    assert res.frozen is True


def test_circuit_breaker_needs_freeze_orbits(du24_setup):
    """Requiring more consecutive settled orbits than the run completes never freezes."""
    res = _cb_run(du24_setup, freeze_strip=0.99, freeze_orbits=100000, freeze_tdyn=1e9)
    assert res.frozen is False


def test_mass_floor_freeze(du24_setup):
    """freeze_mfrac freezes a still-stripping run once m falls below the floor -- no
    not-stripping streak needed, independent of the breaker -- and frozen_mfloor tells
    it apart from a plain breaker freeze."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    m0 = M_sub[-1]
    full = _cb_run(du24_setup)
    frz = _cb_run(du24_setup, freeze_mfrac=0.5)
    assert frz.frozen is True and frz.frozen_mfloor is True
    assert frz.m[-1] < 0.5 * m0
    assert frz.t[-1] < full.t[-1]                       # stopped early
    plain = _cb_run(du24_setup, freeze_strip=0.99, freeze_orbits=2, freeze_tdyn=1e9)
    assert plain.frozen is True and plain.frozen_mfloor is False
    assert full.frozen_mfloor is False


def test_dynamical_friction_disabled_by_none():
    """orbit.integrate with m omitted (None) disables dynamical friction: a
    heavy satellite on a circular orbit holds its radius, while the same orbit
    with m provided sinks to the centre. This is the path evolve_heating takes
    for dynamical_friction=False -- it passes m=None to the integrator, which
    skips the fDF term in ftot rather than evaluating it at zero mass."""
    host = NFW(1.0e12, 263.2 / 23.69)
    R0 = 50.
    xv0, _ = sc.make_orbit(host, R0=R0, z0=0., eta=1.0)  # circular
    tgrid = np.linspace(0., 8., 400)[1:]
    o_off = orbit(xv0.copy())
    o_off.integrate(tgrid, host)                 # m=None -> DF off
    o_on = orbit(xv0.copy())
    o_on.integrate(tgrid, host, 1.0e11)          # m provided -> DF on
    r_off = np.hypot(o_off.xvArray[:, 0], o_off.xvArray[:, 2])
    r_on = np.hypot(o_on.xvArray[:, 0], o_on.xvArray[:, 2])
    assert r_off.max() / r_off.min() < 1.001, "DF-off circular orbit must hold its radius"
    assert r_on[-1] < 0.1 * R0, "DF-on heavy satellite must sink to the centre"


def test_evolve_heating_df_off_matches_collisionless(du24_setup):
    """dynamical_friction=False makes evolve_heating omit m from the orbit
    integrator, so the trajectory is exactly the collisionless one -- the
    subhalo's (shrinking) bound mass never feeds back into the orbit. Turning
    DF on deepens the pericentre."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    ns = 12000
    timesteps = np.linspace(0., 12., ns + 1)[1:]
    res_off = sc.evolve_heating(
        hNFW, NumericProfile(rvals, M_sub), xv0_20, tmax=12., Nstep=ns,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        second_order=True, truncation='kazantzidis', dynamical_friction=False)
    o_bare = orbit(xv0_20.copy())
    o_bare.integrate(timesteps, hNFW)
    r_bare = np.hypot(o_bare.xvArray[:, 0], o_bare.xvArray[:, 2])
    fin = np.isfinite(res_off.r)
    assert np.allclose(res_off.r[fin], r_bare[fin], rtol=1e-9, atol=0.), \
        "DF-off orbit must match the bare collisionless integration"
    res_on = sc.evolve_heating(
        hNFW, NumericProfile(rvals, M_sub), xv0_20, tmax=12., Nstep=ns,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        second_order=True, truncation='kazantzidis', dynamical_friction=True)
    fon = np.isfinite(res_on.r)
    assert res_on.r[fon].min() < res_off.r[fin].min(), \
        "DF on deepens the pericentre vs DF off"


def _revir(setup, **kw):
    hNFW, rvals, M_sub, xv0_20 = setup
    return sc.evolve_heating_revirial(
        hNFW, NumericProfile(rvals, M_sub), xv0_20, tmax=12.,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        second_order=True, truncation='hard', **kw)


def _n_revir(res):
    """Re-virialization count = step-updates of the (piecewise-constant between
    apocenters) Vmax track."""
    v = res.vmax[np.isfinite(res.vmax)]
    return int(np.sum(np.abs(np.diff(v)) > 0.))


def test_revirial_apocenter_count_is_dt_independent(du24_setup):
    """The re-virialization count is set by the orbit (one per apocenter), not by
    the timestep: refining step_frac leaves both the number of re-virializations and
    the final bound mass unchanged. This is what makes the heating dt-convergent --
    heat_profile is applied a fixed number of times regardless of dt."""
    coarse = _revir(du24_setup, step_frac=0.05)
    fine = _revir(du24_setup, step_frac=0.02)
    assert _n_revir(coarse) == _n_revir(fine)
    assert _n_revir(coarse) > 5    # the 1/20 re-virializes at every apocenter
    rc = coarse.m[np.isfinite(coarse.m)]
    rf = fine.m[np.isfinite(fine.m)]
    assert rc[-1] / rc[0] == pytest.approx(rf[-1] / rf[0], rel=0.05), \
        "final bound mass is dt-convergent (no plateau-then-jump)"


def test_revirial_profile_frozen_between_apocenters(du24_setup):
    """Between re-virializations the reference profile is held fixed: Vmax is
    piecewise-constant, stepping only at apocenters. Many steps share each level."""
    res = _revir(du24_setup, step_frac=0.05)
    v = res.vmax[np.isfinite(res.vmax)]
    n_levels = len(np.unique(np.round(v, 8)))
    assert n_levels >= 5                  # multiple re-virialization events
    assert n_levels < len(v) / 5          # but far fewer than steps (frozen between)


def test_revirial_fallback_revirializes_without_apocenter(du24_setup):
    """Re-virialization normally fires at each apocenter (r local maximum). When the
    orbit has no local maximum over the integration -- a window shorter than half a
    radial period (a monotone-ish plunge from apocenter, as here), a perfectly
    circular orbit, or an orbit distorted by a future time-evolving host -- the
    revir_fallback timer re-virializes after revir_fallback * t_orb instead, so the
    accumulated heating does not grow unbounded. The 1/20 starts at apocenter and
    plunges, so a 1.5 Gyr window registers no apocenter: without the fallback the
    only re-virialization is the terminal flush; with it, heating feeds back into the
    profile repeatedly during the plunge."""
    def run(fb):
        hNFW, rvals, M_sub, xv0_20 = du24_setup
        return sc.evolve_heating_revirial(
            hNFW, NumericProfile(rvals, M_sub), xv0_20, tmax=1.5, step_frac=0.02,
            revir_fallback=fb, epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93,
            t_dyn_mode='sub_lt', second_order=True, truncation='hard',
            dynamical_friction=False)
    no_fb = run(1e9)
    with_fb = run(0.3)
    # no apocenter in a sub-radial-period plunge: the only re-virialization is the
    # end-of-evolution flush (one step-update of Vmax, at the final step)
    assert _n_revir(no_fb) == 1
    # the fallback re-virializes repeatedly mid-plunge, well before the end
    assert _n_revir(with_fb) > 1


def test_revirial_apo_only_at_apocenter_on_eccentric(du24_setup):
    """The fallback is timed against the apocenter dynamical time t_dyn(r_apo), not
    t_dyn(current r). The latter collapses toward pericentre and used to fire a
    spurious second re-virialization mid-plunge each orbit. On the eccentric 1/20
    orbit every in-loop re-virialization must now sit at an apocenter (r near the
    orbital-radius maximum), not at the ~0.2 r_max mid-plunge point. The terminal
    flush (the last recorded step) settles the remnant wherever the run ended and is
    excluded from the apocenter check."""
    res = _revir(du24_setup)
    rmax_orb = res.r[np.isfinite(res.r)].max()
    last = len(res.apo) - 1                        # the end-of-run flush step
    mid_apo = np.where(res.apo)[0]
    mid_apo = mid_apo[mid_apo != last]
    assert mid_apo.size > 2
    assert np.all(res.r[mid_apo] > 0.8 * rmax_orb)


def test_revirial_apo_flag_marks_postrevir_steps(du24_setup):
    """res.apo (bool) marks the re-virialization steps -- where Vmax/rmax step to the
    post-reshape equilibrium. Vmax is carried (constant) between, so every Vmax change
    coincides with an apo-flagged step, and the apo steps are few among many."""
    res = _revir(du24_setup)
    assert res.apo.dtype == bool
    vchange = set((np.where(np.abs(np.diff(res.vmax)) > 0.)[0] + 1).tolist())
    apo = set(np.where(res.apo)[0].tolist())
    assert vchange.issubset(apo)                  # every structural update is a flagged apocenter
    assert res.apo.sum() < len(res.apo) / 5       # frozen between (few apo among many steps)


def test_revirial_disrupted_run_last_step_not_apo(du24_setup):
    """A run that disrupts (m -> Mres under early_terminate) skips the terminal flush,
    so its last recorded step is NOT force-marked apo -- the downstream apocenter
    extraction then won't treat the floored final state as an equilibrium apocenter."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    sub0 = NumericProfile(rvals, M_sub)
    cfg.Mres = 0.3 * sub0.Mh              # high floor: the deep 1/20 disrupts
    res = sc.evolve_heating_revirial(
        hNFW, sub0, xv0_20, tmax=37., step_frac=0.02,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        second_order=True, truncation='hard', early_terminate=True)
    assert res.m[np.isfinite(res.m)][-1] <= cfg.Mres * (1. + 1e-9)   # ended at the floor
    assert not bool(res.apo[-1])          # disrupted -> no flush -> last step not an apocenter


def test_revirial_r_stop_parks(du24_setup):
    """r_stop halts the integration once the orbit dips inside it: the terminal
    record sits below r_stop with the bound mass preserved (not floored)."""
    hNFW, rvals, M_sub, _ = du24_setup
    xv0, _ = sc.make_orbit(hNFW, R0=0.5 * hNFW.rh, z0=0., eta=0.3)
    r0 = np.hypot(xv0[0], xv0[2])
    rstop = 0.6 * r0
    res = sc.evolve_heating_revirial(
        hNFW, NumericProfile(rvals, M_sub), xv0, tmax=12., step_frac=0.05,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        truncation='hard', dynamical_friction=False, r_stop=rstop)
    r = res.r[np.isfinite(res.r)]
    m = res.m[np.isfinite(res.m)]
    assert r[-1] < rstop               # terminal record is inside r_stop
    assert m[-1] > cfg.Mres            # bound mass preserved, not floored


def test_revirial_raises_pericentre_unresolved_over_budget(du24_setup):
    """Exceeding the step budget raises PericentreUnresolvedError -- a subclass of
    OverstripError, so an `except OverstripError` handler catches it while the
    distinct type flags an unresolvable orbit."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    with pytest.raises(sc.OverstripError) as ei:
        sc.evolve_heating_revirial(
            hNFW, NumericProfile(rvals, M_sub), xv0_20, tmax=12., step_frac=0.02,
            max_steps=50, epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93,
            t_dyn_mode='sub_lt', second_order=True, truncation='hard')
    assert issubclass(sc.PericentreUnresolvedError, sc.OverstripError)
    # the raise carries the terminal bound mass alongside dens_hist/t_last/r_last
    m_last = ei.value.m_last
    assert np.isfinite(m_last) and cfg.Mres < m_last <= NumericProfile(rvals, M_sub).Mh


def test_revirial_disruption_terminates_without_revirializing(du24_setup):
    """A subhalo stripped to the mass floor ends via early_terminate at m = Mres,
    with no final re-virialization -- a floored profile has no bound structure to
    settle, and re-virializing one is meaningless/fragile. A high mass floor forces
    the otherwise-surviving dense 1/20 to disrupt."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    sub0 = NumericProfile(rvals, M_sub)
    cfg.Mres = 0.3 * sub0.Mh              # high floor: the deep 1/20 crosses it
    res = sc.evolve_heating_revirial(
        hNFW, sub0, xv0_20, tmax=37., step_frac=0.02,
        epsh=0.0741, gamma=0., beta_h=0.278, alpha=3.93, t_dyn_mode='sub_lt',
        second_order=True, truncation='hard', early_terminate=True)
    fin = np.isfinite(res.m)
    assert res.m[fin][-1] <= cfg.Mres * (1. + 1e-9)     # ended at the floor
    assert res.t[fin][-1] < 37. * 0.999                 # stopped before tmax
    assert np.all(np.isfinite(res.vmax[fin]))           # no floored-reshape garbage


def test_truncate_powerlaw_monotone_at_large_budget():
    """Regression: a tiny tail budget on a large M_t let the analytic tail's
    outer grid points edge ~1e-8 over m_total in float64, so the exact m_total
    override at r_out dipped M(<r) below its neighbour and NumericProfile
    rejected it -- crashing the powerlaw MCMC mid-DE-search at alpha_s ~ 0
    (m_total ~ Mh). The outer tail is now clamped to m_total. Fixture is the
    captured failing evolution step (433-point heated profile)."""
    import os
    from subhalo_functions import truncate_powerlaw
    d = np.load(os.path.join(os.path.dirname(__file__), 'test_data',
                             'truncate_powerlaw_nonmono.npz'))
    prof = NumericProfile(d['ri'], d['Mr'])
    res = truncate_powerlaw(prof, float(d['r_t']), n=float(d['n']),
                            m_total=float(d['m_total']))
    assert np.all(np.diff(res.Mr) >= 0.)                     # non-decreasing
    assert res.Mr[-1] == pytest.approx(float(d['m_total']))   # mass conserved


def _du24_sub_M_analytic(sat, r):
    """Enclosed mass of the Du+24 subhalo (eqs. 8-10) at radius r: NFW within
    r_vir, exponential truncation (r_decay = 0.1 r_vir, kappa from eq. 10)
    beyond it -- the profile du24_nfw_setup should reproduce."""
    rvir, rdecay = sat.rh, 0.1 * sat.rh
    kappa = rvir / rdecay - sat.s(rvir)          # eq. 10 (sat.s = -dln rho/dln r)
    rho_t = float(sat.rho(rvir))
    if r <= rvir:
        return float(sat.M(r))
    tail = quad(lambda s: 4. * np.pi * s**2 * rho_t * (s / rvir)**kappa
                * np.exp(-(s - rvir) / rdecay), rvir, r)[0]
    return float(sat.Mh + tail)


def test_du24_subhalo_matches_analytic_profile():
    """du24_nfw_setup builds the full Du+24 subhalo (eqs. 8-10), not a bare NFW
    cut at r_vir. The enclosed mass tracks the analytic profile at all radii."""
    _, sat, rvals, M_sub = sc.du24_nfw_setup()
    sub = NumericProfile(rvals, M_sub)
    for x in (0.05, 0.3, 1.0, 1.5, 2.5):
        r = x * sat.rh
        assert float(sub.M(r)) == pytest.approx(_du24_sub_M_analytic(sat, r),
                                                rel=2e-3), f"M(<{x} rvir) off"


def test_du24_subhalo_total_mass_and_virial_mass():
    """The eq. 9-10 tail carries ~16% of M_vir for the gamma=1 NFW, so the IC
    total is ~1.16 M_vir (Du+24 quote 1.02-1.2). The mass within r_vir is still
    exactly M_vir -- the tail is added beyond it, the NFW body is untouched."""
    _, sat, rvals, M_sub = sc.du24_nfw_setup()
    sub = NumericProfile(rvals, M_sub)
    assert float(sub.M(sat.rh)) == pytest.approx(sat.Mh, rel=1e-3)
    assert 1.10 < sub.Mh / sat.Mh < 1.22


def test_truncate_kazantzidis_rho_t_override():
    """rho_t makes the join density exact when r_t sits at the profile's outer
    knot, where np.gradient density is unreliable. The tail mass then matches
    the analytic incomplete-gamma integral at that density; the finite-difference
    join (no override) is off, so the override is load-bearing for an edge join."""
    nfw = NFW(1e9, 20.0)
    rvir, rdecay = nfw.rh, 0.1 * nfw.rh
    r = np.logspace(np.log10(cfg.Rres), np.log10(rvir), 200)
    num = NumericProfile(r, nfw.M(r))
    slope = -nfw.s(rvir)
    kappa = rvir / rdecay + slope
    rho_t = float(nfw.rho(rvir))
    tail = quad(lambda s: 4. * np.pi * s**2 * rho_t * (s / rvir)**kappa
                * np.exp(-(s - rvir) / rdecay), rvir, rvir + 60. * rdecay)[0]
    over = truncate_kazantzidis(num, r_t=rvir, r_decay=rdecay, slope=slope,
                                rho_t=rho_t)
    assert over.Mh == pytest.approx(float(nfw.M(rvir)) + tail, rel=1e-4)
    fd = truncate_kazantzidis(num, r_t=rvir, r_decay=rdecay, slope=slope)
    assert abs(fd.Mh - over.Mh) / over.Mh > 1e-3


def test_cumulant_step_matches_closed_form():
    # composing exact steps with constant g, t_orb reproduces the analytic
    # solution of Du+24 eq. 39, G(t) = (t_orb g / beta_h)(1 - e^{-beta_h t/t_orb}).
    g = np.array([[1.0, 0.3], [0.3, -0.5]])
    beta_h, t_orb, dt, N = 3.0, 1.7, 0.01, 500
    G = np.zeros_like(g)
    for _ in range(N):
        G = sc._cumulant_step(G, g, beta_h, t_orb, dt)
    t = N * dt
    G_analytic = (t_orb * g / beta_h) * (1. - np.exp(-beta_h * t / t_orb))
    assert np.allclose(G, G_analytic, rtol=1e-12, atol=0.)


def test_cumulant_step_positive_where_euler_flips():
    # for b = beta_h dt/t_orb > 1 forward Euler's 1 - b factor goes negative and
    # drives a decaying cumulant (g=0) below zero; the exact step keeps it in
    # (0, G_0].
    G0 = np.full((3, 3), 2.0)
    g = np.zeros((3, 3))
    Gx = sc._cumulant_step(G0, g, beta_h=2.0, t_orb=1.0, dt=1.0)
    assert np.all(Gx > 0.) and np.all(Gx < G0)
    euler = G0 + (g - 2.0 * G0 / 1.0) * 1.0
    assert np.all(euler < 0.)


def test_cumulant_step_zero_beta_is_riemann_sum():
    # beta_h -> 0 has no decay, so the cumulant is the plain time integral of g.
    g = np.array([[0.5, -0.2], [-0.2, 0.9]])
    t_orb, dt, N = 1.0, 0.01, 300
    G = np.zeros_like(g)
    for _ in range(N):
        G = sc._cumulant_step(G, g, 0.0, t_orb, dt)
    assert np.allclose(G, g * (N * dt), rtol=1e-13)


def test_cumulant_step_beta_to_zero_limit():
    # the t_orb g / beta_h term is a 0/0 as beta_h -> 0; the step must approach
    # the plain integral G + g dt continuously rather than blow up.
    G0 = np.full((3, 3), 0.7)
    g = np.array([[1.0, 0.3, 0.0], [0.3, -0.5, 0.2], [0.0, 0.2, 0.4]])
    t_orb, dt = 1.7, 0.05
    riemann = G0 + g * dt
    assert np.array_equal(sc._cumulant_step(G0, g, 0.0, t_orb, dt), riemann)
    # error is O(beta_h) above the roundoff floor of (t_orb g/beta_h)(1-e^{-b}):
    # each 100x drop in beta_h shrinks the gap to the beta_h=0 integral ~100x
    errs = [np.abs(sc._cumulant_step(G0, g, b, t_orb, dt) - riemann).max()
            for b in (1e-1, 1e-3, 1e-6)]
    assert errs[0] < 1e-2
    assert errs[1] < 0.05 * errs[0] and errs[2] < 0.05 * errs[1]
    # expm1 keeps the small-beta_h forcing weight accurate: no 0/0 blow-up and
    # the step still matches the beta_h=0 integral to ~machine precision
    G_tiny = sc._cumulant_step(G0, g, 1e-12, t_orb, dt)
    assert np.all(np.isfinite(G_tiny)) and np.abs(G_tiny - riemann).max() < 1e-12


def test_cumulant_step_dt_to_zero_matches_euler():
    # both schemes share the leading term G + (g - beta_h G/T) dt, so their
    # difference is O(dt^2): halving dt cuts the exact-vs-Euler gap ~4x.
    G0 = np.full((3, 3), 0.7)
    g = np.full((3, 3), 1.3)
    beta_h, t_orb = 6.0, 1.1

    def gap(dt):
        ex = sc._cumulant_step(G0, g, beta_h, t_orb, dt)
        eu = G0 + (g - beta_h * G0 / t_orb) * dt
        return np.abs(ex - eu).max()

    d1, d2 = gap(1e-2), gap(5e-3)
    assert d1 > 0.
    assert 3.5 < d1 / d2 < 4.5


def test_cumulant_step_more_accurate_than_euler():
    # for a large step (b = beta_h dt/t_orb ~ 1) the exact step reproduces the
    # analytic single-step solution to machine precision while forward Euler's
    # first-order truncation is off by ~b^2/2 |G_eq - G_0|.
    G0 = np.full((3, 3), 0.7)
    g = np.full((3, 3), 1.3)
    beta_h, t_orb, dt = 6.0, 1.1, 0.2  # b = 6*0.2/1.1 ~ 1.09
    b = beta_h * dt / t_orb
    G_eq = t_orb * g / beta_h
    analytic = G0 * np.exp(-b) + G_eq * (1. - np.exp(-b))
    err_exact = np.abs(sc._cumulant_step(G0, g, beta_h, t_orb, dt) - analytic).max()
    euler = G0 + (g - beta_h * G0 / t_orb) * dt
    err_euler = np.abs(euler - analytic).max()
    assert err_exact < 1e-14
    assert err_euler > 1e-2
    assert err_euler > 1e6 * max(err_exact, 1e-16)


def test_heating_stepper_trapezoidal_increment():
    # first-order path: the per-step energy amplitude eps_r(r)/r^2 is the
    # trapezoidal time-integral of the heating rate, (hr_prev + hr)/2 * dt. The
    # first step has no left sample and falls back to the rectangle hr*dt.
    st = sc._HeatingStepper(None, second_order=False)
    dt = 0.1
    hrs = [2.0, 4.0, 6.0, 3.0]
    amp = [st.step(dt, hr, r=1.0, t_now=0., t_orb=1.0)[0](1.0) for hr in hrs]
    expected = [2.0 * dt, 0.5 * (2 + 4) * dt, 0.5 * (4 + 6) * dt,
                0.5 * (6 + 3) * dt]
    assert np.allclose(amp, expected)


def test_heating_stepper_trapezoidal_H_spans_reset():
    # second-order H accumulates the same trapezoidal increment, and the
    # per-orbit reset zeroes the accumulator but not the quadrature's left
    # sample, so the increment straddling a reset still uses the boundary hr.
    class _Stub:
        rh = 1.0
        _sig2 = staticmethod(lambda r_: np.zeros_like(np.asarray(r_, float)))

    st = sc._HeatingStepper(_Stub(), second_order=True)
    dt = 0.1
    st.step(dt, 2.0, r=3.0, t_now=0.0, t_orb=1.0)   # rectangle: H = 2*dt
    st.step(dt, 4.0, r=2.0, t_now=0.1, t_orb=1.0)   # trap: H += (2+4)/2*dt
    assert st.H == pytest.approx((2.0 + 0.5 * (2 + 4)) * dt)
    st.reset(_Stub(), t_now=0.2)                    # H -> 0, hr_prev kept at 4
    st.step(dt, 6.0, r=2.5, t_now=0.2, t_orb=1.0)   # trap across reset: (4+6)/2*dt
    assert st.H == pytest.approx(0.5 * (4 + 6) * dt)


def test_heating_stepper_trapezoidal_is_second_order():
    # integrating a smooth heating rate f(t) over the run, the stepper's
    # trapezoidal H converges as O(dt^2) while the rectangle sum is O(dt): the
    # trapezoid is far closer at the same resolution and improves ~4x when dt
    # halves (vs ~2x for the rectangle).
    import math

    class _Stub:
        rh = 1.0
        _sig2 = staticmethod(lambda r_: np.zeros_like(np.asarray(r_, float)))

    f = lambda t: math.sin(t) + 0.5 * t        # smooth, nonlinear, non-periodic
    F = lambda t: -math.cos(t) + 0.25 * t**2   # antiderivative
    T = 2.0

    def errs(N):
        dt = T / N
        st = sc._HeatingStepper(_Stub(), second_order=True)
        rect = 0.
        for t in np.linspace(0., T, N + 1)[1:]:
            st.step(dt, f(t), r=1.0, t_now=t, t_orb=1e9)  # t_orb huge: no reset
            rect += f(t) * dt
        analytic = F(T) - F(0.)
        return abs(st.H - analytic), abs(rect - analytic)

    et1, er1 = errs(64)
    et2, er2 = errs(128)
    assert et1 < 0.1 * er1              # trapezoid much closer at same N
    assert 3.5 < et1 / et2 < 4.5        # O(dt^2)
    assert 1.7 < er1 / er2 < 2.3        # rectangle only O(dt)


def _green_final(setup, step_frac):
    """Final bound fraction and adaptive step count from the Green/DASH engine on
    the 1/20 orbit at the given step_frac."""
    hNFW, _rvals, _M_sub, xv0_20 = setup
    res = sc.evolve_satgen_green(hNFW, 1.0e9, 26.32 / 1.279, xv0_20, tmax=12.,
                                 alpha='conc', step_frac=step_frac)
    m = res.m[np.isfinite(res.m)]
    return m[-1] / 1.0e9, len(res.t)


def test_green_adaptive_step_frac_convergence(du24_setup):
    """Refining step_frac leaves the deep-stripping bound mass unchanged: the King62
    strip is integrated exactly (exp relaxation toward M(<lt)), so it is dt-robust --
    a forward-Euler step would drift with dt. The finer run also takes more steps."""
    coarse, n_coarse = _green_final(du24_setup, step_frac=0.04)
    fine, n_fine = _green_final(du24_setup, step_frac=0.005)
    assert coarse == pytest.approx(fine, rel=0.03), \
        f"step_frac refinement must converge: {coarse:.4f} vs {fine:.4f}"
    assert n_fine > n_coarse                      # finer step_frac -> more steps
    assert fine < 0.1, "the 1/20 orbit strips deeply, not frozen near unity"
