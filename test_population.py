import numpy as np
import pytest

import config as cfg
import stripping_common as sc
from population import clump_field, evolve_population
from subhalo_functions import NumericProfile


@pytest.fixture(scope='module')
def pop_setup():
    cfg.Mres, cfg.Rres = 1e4, 1e-4
    host, sat, rv, Msub = sc.du24_nfw_setup()
    num0 = NumericProfile(rv, Msub)
    etas = (0.65, 0.45, 0.3)
    xvs = [sc.make_orbit(host, R0=0.5 * host.rh, eta=e)[0] for e in etas]
    return host, num0, xvs


ENGINE = dict(step_frac=0.02, second_order=True, dynamical_friction=True)
TMAX = 3.


def _solo(host, num0, xv0, **kw):
    return sc.evolve_heating_revirial(host, num0, xv0, tmax=TMAX, **ENGINE, **kw)


def test_uncoupled_population_matches_separate_runs(pop_setup):
    """The driver interleaves subhalos on a due-time queue; with nothing coupling them
    that must reproduce N separate runs exactly, not just closely. A subhalo's step
    sequence may not depend on when its neighbours happen to be stepped."""
    host, num0, xvs = pop_setup
    solo = [_solo(host, num0, xv) for xv in xvs]
    out = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                            tmax=TMAX, tsnap=np.linspace(0., TMAX, 7),
                            keep_results=True, **ENGINE)
    for a, b in zip(solo, out['results']):
        for f in ('t', 'r', 'm', 'vmax', 'rmax', 'lt', 'xv', 'apo'):
            x, y = getattr(a, f), getattr(b, f)
            assert x.shape == y.shape, f
            assert ((x == y) | (np.isnan(x) & np.isnan(y))).all(), f


def test_population_residence_is_the_sum_over_subhalos(pop_setup):
    """The shared histogram is the population total: every subhalo writes into one array,
    so it must equal the sum of what they accumulate alone. Not bit-exact -- the shared
    array takes the same terms interleaved in time where the reference sums them one
    subhalo at a time, so they differ at the ulp."""
    host, num0, xvs = pop_setup
    redges = np.logspace(np.log10(1e-3), np.log10(host.rh), 25)
    tedges = np.linspace(0., TMAX, 7)
    kw = dict(dens_redges=redges, dens_tedges=tedges)
    solo = [_solo(host, num0, xv, **kw) for xv in xvs]
    out = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                            tmax=TMAX, **kw, **ENGINE)
    assert np.allclose(out['dens_hist'], sum(r.dens_hist for r in solo),
                       rtol=1e-12, atol=0.)


def test_total_work_is_the_sum_not_n_times_the_slowest(pop_setup):
    """The point of the due-time queue: a subhalo on a short timestep must not drag the
    others onto it. Total accepted steps equal the sum of the solo counts -- a global
    minimum timestep would instead cost about N times the largest."""
    host, num0, xvs = pop_setup
    solo = [_solo(host, num0, xv) for xv in xvs]
    out = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                            tmax=TMAX, tsnap=np.linspace(0., TMAX, 7), **ENGINE)
    assert list(out['n_step']) == [r.t.size for r in solo]
    n_lockstep = len(xvs) * max(r.t.size for r in solo)
    assert out['n_step'].sum() < n_lockstep


def test_one_subhalo_raising_does_not_kill_the_population(pop_setup):
    """A subhalo the engine cannot resolve terminates itself, not the run: the others
    still evolve to tmax and the failed one is labelled and keeps the residence it
    accumulated."""
    host, num0, xvs = pop_setup
    redges = np.logspace(np.log10(1e-3), np.log10(host.rh), 25)
    tedges = np.linspace(0., TMAX, 7)
    subs = [dict(prof=num0, xv0=xvs[0]),
            dict(prof=num0, xv0=xvs[1], max_steps=12),   # cannot finish
            dict(prof=num0, xv0=xvs[2])]
    out = evolve_population(host, subs, tmax=TMAX, dens_redges=redges,
                            dens_tedges=tedges, **ENGINE)
    assert out['status'][1] == 'unresolved'
    assert set(out['status'][[0, 2]]) == {'completed'}
    assert out['t_last'][0] > 0.9 * TMAX and out['t_last'][2] > 0.9 * TMAX
    assert out['dens_hist'].sum() > 0.


def test_held_subhalo_keeps_depositing_to_tmax(pop_setup):
    """A frozen subhalo has stopped being integrated but has not left the host, so it
    must keep contributing residence at its settled radius. Without the hold its
    residence would stop at the freeze and the population's field would lose it."""
    host, num0, xvs = pop_setup
    # long enough for the streak to complete: this orbit re-virializes about every
    # 2 Gyr, and the breaker needs freeze_orbits of them inside the loop
    tmax = 6.
    redges = np.logspace(np.log10(1e-3), np.log10(host.rh), 25)
    tedges = np.linspace(0., tmax, 7)
    freeze = dict(freeze_strip=1.0, freeze_orbits=2, freeze_rspan=10.,
                  freeze_tdyn=1e9)
    subs = [dict(prof=num0, xv0=xvs[2], **freeze)]
    kw = dict(dens_redges=redges, dens_tedges=tedges, **ENGINE)
    on = evolve_population(host, subs, tmax=tmax, hold_terminal=True, **kw)
    off = evolve_population(host, subs, tmax=tmax, hold_terminal=False, **kw)
    assert on['status'][0] == 'frozen'
    assert on['t_last'][0] < tmax          # froze early, so there is a tail to hold
    assert on['dens_hist'].sum() > off['dens_hist'].sum()
    # the held mass lands at the settled radius, in the last time bins the run never
    # reached under its own integration
    assert on['dens_hist'][:, -1].sum() > 0.
    assert off['dens_hist'][:, -1].sum() == 0.


def test_clump_field_conserves_mass_and_tracks_the_population(pop_setup):
    """The field is the realization's own: enclosed clump mass must be monotonic in
    radius and account for every bound subhalo out to the last edge."""
    host, num0, xvs = pop_setup
    redges = np.logspace(np.log10(1e-3), np.log10(3. * host.rh), 25)
    out = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                            tmax=TMAX, tsnap=np.linspace(0., TMAX, 7), **ENGINE)
    f = clump_field(out, redges, host=host)
    assert (np.diff(f['M_enc'], axis=0) >= 0.).all()
    assert np.allclose(f['M_enc'][-1], out['m_snaps'].sum(axis=0))
    assert f['M_enc'][-1, 0] == pytest.approx(out['m_init'].sum())
    assert (f['rho'] >= 0.).all() and (f['sigma'] >= 0.).all()
    assert f['frac'].shape == f['M_enc'].shape
