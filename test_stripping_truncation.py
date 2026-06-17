"""
Integration regression: the Kazantzidis l_t tail (with the second-order term)
prevents the disruption that the hard cut produces on Du+24's radial 1/20
orbit.

The King62 rate strips on the subhalo dynamical time T_dyn,sub(l_t) while the
tidal-tensor cumulant accumulates on the host orbital time (Du+24 eqs. 35,
38-39). Under that split the hard cut still collapses the radial orbit to a
bare stub, but at ~6-8 Gyr rather than the ~4 Gyr of the older
all-subhalo-time rate.

Slower than the Tier-1 unit tests (~30 s) -- runs two short evolve_heating
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


def _run(setup, truncation, second_order=False):
    hNFW, rvals, M_sub, xv0_20 = setup
    return sc.evolve_heating(
        hNFW, NumericProfile(rvals, M_sub), xv0_20,
        tmax=12., Nstep=12000, epsh=0.0741, gamma=0., beta_h=0.278,
        alpha=3.93, t_dyn_mode='sub_lt', second_order=second_order,
        truncation=truncation)


def test_hard_cut_disrupts_on_radial_orbit(du24_setup):
    """The hard cut collapses the subhalo to the mass floor on the 1/20
    orbit. The split timescale slows this to ~6-8 Gyr; the older
    all-subhalo-time rate disrupted at ~4 Gyr."""
    res = _run(du24_setup, 'hard')
    m = res.m[np.isfinite(res.m)]
    t = res.t[np.isfinite(res.m)]
    floored = np.where(m <= cfg.Mres)[0]
    assert len(floored) > 0, "hard cut should disrupt on the 1/20 orbit"
    assert 5. < t[floored[0]] < 10.  # driver: ~6.5 Gyr (first order)


def test_kazantzidis_tail_survives(du24_setup):
    """The Kazantzidis l_t tail with the second-order term keeps the subhalo
    bound, with a finite Vmax, past the time the hard cut disrupts it."""
    res = _run(du24_setup, 'kazantzidis', second_order=True)
    m = res.m[np.isfinite(res.m)]
    assert np.all(m > cfg.Mres), "kazantzidis run should not hit the mass floor"
    late = np.isfinite(res.t) & (res.t > 9.)
    assert np.sum(late) > 0
    assert np.all(np.isfinite(res.vmax[late])), \
        "the Kazantzidis profile keeps a Vmax peak past the hard-cut disruption"


def test_powerlaw_tail_survives_at_calibrated_params(du24_setup):
    """The slope-deficit power-law tail keeps the subhalo bound with a finite
    Vmax through 12 Gyr on the 1/20 orbit at the MCMC-calibrated (p50)
    stripping rate -- the regime the model operates in.

    Not run at the Du+24 Table IV rate (alpha=3.93) like the Kazantzidis
    test: that rate over-strips, the King62 budget then sits below the tail's
    minimum mass 4 pi rho_t lt^3/(n-3) (an r^-n envelope cannot carry less)
    on ~90% of steps, and the per-step fallback hard cut disrupts the
    subhalo just as truncation='hard' does."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    res = sc.evolve_heating(
        hNFW, NumericProfile(rvals, M_sub), xv0_20,
        tmax=12., Nstep=12000, epsh=0.0708, gamma=0.0094, beta_h=0.906,
        alpha=0.171, f2=0.188, t_dyn_mode='sub_lt', second_order=True,
        truncation='powerlaw')
    m = res.m[np.isfinite(res.m)]
    assert np.all(m > cfg.Mres), "powerlaw run should not hit the mass floor"
    assert m[-1] / m[0] < 0.05, "the 1/20 orbit strips deeply, not frozen"
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
    stripped excess into floating-point noise (a silent hard cut)."""
    hNFW, rvals, M_sub, xv0_20 = du24_setup
    with pytest.raises(ValueError, match=r"alpha\*dt/T_strip"):
        sc.evolve_heating(
            hNFW, NumericProfile(rvals, M_sub), xv0_20,
            tmax=12., Nstep=12000, epsh=0.0741, gamma=0., beta_h=0.278,
            alpha=20., t_dyn_mode='sub_lt', second_order=True,
            truncation='kazantzidis')


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
