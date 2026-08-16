import numpy as np
import pytest

import config as cfg
import stripping_common as sc
import tidal_engine as te
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
    return te.evolve_heating_revirial(host, num0, xv0, tmax=TMAX, **ENGINE, **kw)


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
    n_global_dt = len(xvs) * max(r.t.size for r in solo)
    assert out['n_step'].sum() < n_global_dt


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


def test_engine_advances_a_time_dependent_host(pop_setup):
    """A host with set_time must be advanced on the subhalo's clock: the engine calls
    set_time(t) once per step, so a factor that turns on at t > 0 changes the orbit.
    Both halves matter -- the hook being called at all, and the potential actually
    differing from the static run."""
    host, num0, xvs = pop_setup
    from profiles import NFW
    from subhalo_functions import CorrectedHost

    times = []

    class Recording(CorrectedHost):
        def set_time(self, t):
            times.append(t)
            super().set_time(t)

    rk = np.geomspace(1e-2 * host.rh, host.rh, 15)
    g = np.vstack([np.ones(len(rk)), np.full(len(rk), 1.3)])  # 30% heavier after 1 Gyr
    th = Recording(host, rk, [0., 1., TMAX], g)
    res_t = _solo(th, num0, xvs[0])
    assert len(times) > 100, "set_time was never called on the engine hot path"
    assert times == sorted(times)
    res_s = _solo(host, num0, xvs[0])
    r_t = np.interp(2.5, res_t.t, res_t.r)
    r_s = np.interp(2.5, res_s.t, res_s.r)
    assert abs(r_t - r_s) > 1e-3 * r_s, "time-dependent factor left the orbit unchanged"


def test_noop_barriers_leave_the_run_identical(pop_setup):
    """Barriers synchronize the schedule, not the physics: steps are never truncated at
    a barrier, so with a do-nothing callback every subhalo's trajectory and residence
    must match the unbarriered run exactly."""
    host, num0, xvs = pop_setup
    redges = np.logspace(np.log10(1e-3), np.log10(host.rh), 25)
    tedges = np.linspace(0., TMAX, 7)
    kw = dict(dens_redges=redges, dens_tedges=tedges, keep_results=True)
    ref = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                            tmax=TMAX, **kw, **ENGINE)
    seen = []
    out = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                            tmax=TMAX, barriers=tedges[1:-1],
                            on_barrier=lambda t, runs: seen.append(t),
                            **kw, **ENGINE)
    assert seen == list(tedges[1:-1])
    for a, b in zip(ref['results'], out['results']):
        for f in ('t', 'r', 'm'):
            x, y = getattr(a, f), getattr(b, f)
            assert x.shape == y.shape and (x == y).all(), f
    assert (ref['dens_hist'] == out['dens_hist']).all()


def test_per_run_hists_sum_to_the_shared_one(pop_setup):
    """share_hist=False keeps each subhalo's residence on its own run (needed when the
    members carry different abundance weights); their sum must reproduce the shared
    total, held tails included."""
    host, num0, xvs = pop_setup
    redges = np.logspace(np.log10(1e-3), np.log10(host.rh), 25)
    tedges = np.linspace(0., TMAX, 7)
    kw = dict(dens_redges=redges, dens_tedges=tedges)
    shared = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                               tmax=TMAX, **kw, **ENGINE)
    per = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                            tmax=TMAX, share_hist=False, keep_results=True,
                            **kw, **ENGINE)
    assert np.allclose(per['dens_hist'], shared['dens_hist'], rtol=1e-12, atol=0.)
    assert np.allclose(sum(r.dens_hist for r in per['results']),
                       shared['dens_hist'], rtol=1e-12, atol=0.)


def test_checkpoint_resume_reproduces_the_uninterrupted_run(pop_setup):
    """A checkpoint taken at a barrier and passed back as resume_state must continue
    to the exact terminal state of the uninterrupted run -- the queue is rebuilt from
    the run clocks at every segment boundary, so resuming there adds nothing. The
    state round-trips through pickle, which is what a production checkpoint does."""
    import pickle
    import profiles as pr
    host, num0, xvs = pop_setup
    heavy = pr.NFW(2. * host.Mh, 10., Delta=200.)
    redges = np.logspace(np.log10(1e-3), np.log10(host.rh), 25)
    tedges = np.linspace(0., TMAX, 7)

    def swap(t_b, runs):
        for r in runs:
            r.potential = heavy

    kw = dict(dens_redges=redges, dens_tedges=tedges, share_hist=False,
              keep_results=True, barriers=tedges[1:-1], on_barrier=swap)
    ref = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                            tmax=TMAX, **kw, **ENGINE)

    blob = {}

    class Interrupt(Exception):
        pass

    def ckpt(t_b, state):
        blob['state'] = pickle.dumps(state)
        if t_b >= tedges[2]:
            raise Interrupt

    with pytest.raises(Interrupt):
        evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                          tmax=TMAX, checkpoint_fn=ckpt, **kw, **ENGINE)
    out = evolve_population(host, [dict(prof=num0, xv0=xv) for xv in xvs],
                            tmax=TMAX, resume_state=pickle.loads(blob['state']),
                            **kw, **ENGINE)
    for a, b in zip(ref['results'], out['results']):
        for f in ('t', 'r', 'm', 'xv'):
            x, y = getattr(a, f), getattr(b, f)
            assert x.shape == y.shape and (x == y).all(), f
    assert (ref['dens_hist'] == out['dens_hist']).all()
    assert list(ref['n_step']) == list(out['n_step'])
    assert list(ref['status']) == list(out['status'])


def test_barrier_potential_swap_changes_the_evolution(pop_setup):
    """The coupling contract: a callback that swaps run.potential must be honored by
    the next segment. A doubled-mass host from the first barrier on has to leave a
    different terminal state than the static run."""
    import profiles as pr
    host, num0, xvs = pop_setup
    heavy = pr.NFW(2. * host.Mh, 10., Delta=200.)

    def swap(t_b, runs):
        for r in runs:
            r.potential = heavy
    static = evolve_population(host, [dict(prof=num0, xv0=xvs[0])], tmax=TMAX,
                               **ENGINE)
    moved = evolve_population(host, [dict(prof=num0, xv0=xvs[0])], tmax=TMAX,
                              barriers=[TMAX / 3.], on_barrier=swap, **ENGINE)
    assert moved['m_final'][0] != static['m_final'][0]
