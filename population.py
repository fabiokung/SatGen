# joint evolution of a whole subhalo population in one host

import heapq

import numpy as np

import config as cfg
from subhalo_functions import HeatingUnbindsError
from tidal_engine import PericentreUnresolvedError, _dens_accumulate, _SubhaloRun


def _snap_track(tsnap, t, y, y0, y_tail):
    """A per-step track sampled onto the snapshot times: held at y0 before the first
    accepted step and at y_tail after the last. Interpolating orbital radius, bound mass
    and speed is safe (all single-valued in t); the cylindrical angle is not, and is not
    carried here."""
    ok = np.isfinite(t) & np.isfinite(y)
    if not ok.any():
        return np.full(len(tsnap), y0, float)
    return np.interp(tsnap, np.concatenate(([0.], t[ok])),
                     np.concatenate(([y0], y[ok])), left=y0, right=y_tail)


def _hold_residence(hist, redges, tedges, r, m, t_from, hist_k=None, kernels=None,
                    v=None):
    """Deposit a held subhalo's residence: mass m sitting at radius r from t_from to the
    end of the time grid, split across time bins by overlap. A subhalo that froze or
    parked stopped being integrated but has not left the host, so it keeps contributing
    to the population's field and residence for the rest of the run."""
    i = min(max(int(np.searchsorted(redges, r)) - 1, 0), hist.shape[0] - 1)
    for j in range(len(tedges) - 1):
        dt = tedges[j + 1] - max(tedges[j], t_from)
        if dt > 0.:
            hist[i, j] += m * dt
            if hist_k is not None and kernels is not None:
                for k, ker in enumerate(kernels):
                    hist_k[k, i, j] += ker(r, v) * m * dt


def _status(run, res, exc):
    """What ended this subhalo: 'unresolved' (the step budget could not resolve the
    pericentre), 'heat_unbound' (heating unbound the profile), 'disrupted' (bound mass at
    the floor), 'frozen' (circuit breaker or mass floor), 'parked' (halted at r_stop), or
    'completed' (ran to tmax)."""
    if isinstance(exc, PericentreUnresolvedError):
        return 'unresolved'
    if isinstance(exc, HeatingUnbindsError):
        return 'heat_unbound'
    if res.m.size and res.m[-1] <= cfg.Mres:
        return 'disrupted'
    if res.frozen:
        return 'frozen'
    if run.r_stop is not None and res.r.size and res.r[-1] < run.r_stop:
        return 'parked'
    return 'completed'


def evolve_population(host, subhalos, tmax=10., tsnap=None,
                      dens_redges=None, dens_tedges=None, dens_kernels=None,
                      hold_terminal=True, keep_results=False, **engine_kw):
    """Evolve a whole population of subhalos together in one host, each on its own
    adaptive timestep.

    subhalos: a sequence of dicts, one per subhalo, each carrying 'prof' (the initial
    NumericProfile) and 'xv0' (cylindrical [R, phi, z, VR, Vphi, Vz]). Any further key is
    a per-subhalo override of engine_kw -- r_stop and freeze_tdyn differ per subhalo once
    the population is not monochromatic, so they cannot be passed once for the run.

    Scheduling is a due-time queue: every subhalo carries its own clock, and the one
    furthest behind takes the next step. Total work is then the sum of the per-subhalo
    step counts, the same as evolving them separately -- a global minimum timestep would
    instead cost N times the most demanding subhalo's count. Nothing couples the subhalos
    yet, so this is numerically identical to N separate evolve_heating_revirial runs; the
    queue exists so that a coupling term has one place to read the population's state
    from, and so that the realization is the unit of work and of output.

    An exception from one subhalo terminates that subhalo, not the run: the residence and
    terminal state it carries are kept and it leaves the queue. A subhalo that froze or
    parked keeps depositing residence at its settled radius until tmax (hold_terminal),
    since it has stopped being integrated but has not left the host.

    Returns a dict: 'dens_hist'/'dens_hist_k' are the residence summed over the whole
    population (the realization is the population, so no weights); 'tsnap' with
    'r_snaps'/'m_snaps'/'v_snaps' (n_sub, n_snap) is the field over time; 'status',
    'n_step', 't_last', 'r_last', 'm_final', 'vmax', 'rmax' are per subhalo. With
    keep_results, 'results' carries every subhalo's full EvolutionResult -- affordable
    for a few hundred subhalos, not for a large population.
    """
    subhalos = list(subhalos)
    n = len(subhalos)
    tsnap = np.asarray(dens_tedges if tsnap is None else tsnap, float)
    dens_on = dens_redges is not None and dens_tedges is not None

    runs = []
    for sub in subhalos:
        kw = dict(engine_kw)
        kw.update({k: v for k, v in sub.items() if k not in ('prof', 'xv0')})
        runs.append(_SubhaloRun(
            host, sub['prof'], sub['xv0'], tmax,
            kw.pop('step_frac', 0.01), kw.pop('dt_growth_max', 3.0),
            kw.pop('strip_number_max', 1.0), kw.pop('floor_orbit_frac', 1e-3),
            kw.pop('dt_abs_frac', 1e-6), kw.pop('max_steps', 200000),
            kw.pop('revir_fallback', 4.0), kw.pop('epsh', 3.), kw.pop('gamma', 2.5),
            kw.pop('alpha', 1.), kw.pop('beta_h', 1.),
            kw.pop('second_order', False), kw.pop('f2', 0.406),
            kw.pop('chi_v', -0.333), kw.pop('t_dyn_mode', 'sub_lt'),
            kw.pop('truncation', 'hard'), kw.pop('tail_n', 5.),
            kw.pop('tail_xi', 0.), kw.pop('lt_choice', 'King62'),
            kw.pop('label', None), kw.pop('early_terminate', False),
            kw.pop('dynamical_friction', True), kw.pop('r_stop', None),
            dens_redges, dens_tedges, dens_kernels,
            kw.pop('freeze_strip', None), kw.pop('freeze_orbits', 20),
            kw.pop('freeze_tdyn', None), kw.pop('freeze_nstep', None),
            kw.pop('freeze_rspan', 0.5), kw.pop('freeze_mfrac', None),
            kw.pop('freeze_tsink', None), kw.pop('log_every', None)))
        if kw:
            raise TypeError(f"unknown engine parameters: {sorted(kw)}")

    # One shared residence histogram, written in place by every subhalo, so the
    # realization total costs one array instead of n. Per-subhalo residence is not
    # recoverable from it -- that is the output contract, not an oversight.
    if dens_on:
        pop_hist = np.zeros((len(np.asarray(dens_redges)) - 1,
                             len(np.asarray(dens_tedges)) - 1))
        pop_hist_k = (np.zeros((len(dens_kernels),) + pop_hist.shape)
                      if dens_kernels else None)
        for run in runs:
            run.dens_hist = pop_hist
            run.dens_hist_k = pop_hist_k
    else:
        pop_hist = pop_hist_k = None

    # due-time queue: pop the subhalo furthest behind, advance it by its own step, push
    # it back at its new time. One heap entry per live subhalo, never a stale one.
    heap = [(run.t, i) for i, run in enumerate(runs)
            if run.t < run.tmax - run.dt_abs]
    heapq.heapify(heap)
    excs = [None] * n
    while heap:
        _, i = heapq.heappop(heap)
        run = runs[i]
        try:
            alive = run.step()
        except (PericentreUnresolvedError, HeatingUnbindsError) as exc:
            # this subhalo is done, the population is not: keep what it accumulated and
            # drop it from the queue
            excs[i] = exc
            continue
        if alive and run.t < run.tmax - run.dt_abs:
            heapq.heappush(heap, (run.t, i))

    out = dict(tsnap=tsnap, n_sub=n,
               status=np.empty(n, dtype=object), n_step=np.zeros(n, int),
               t_last=np.zeros(n), r_last=np.zeros(n), m_final=np.zeros(n),
               m_init=np.zeros(n), vmax=np.zeros(n), rmax=np.zeros(n),
               r_snaps=np.zeros((n, tsnap.size)), m_snaps=np.zeros((n, tsnap.size)),
               v_snaps=np.zeros((n, tsnap.size)),
               dens_hist=pop_hist, dens_hist_k=pop_hist_k)
    results = [] if keep_results else None

    for i, run in enumerate(runs):
        exc = excs[i]
        res = run.finish()
        if results is not None:
            results.append(res)
        r0 = float(np.sqrt(subhalos[i]['xv0'][0]**2 + subhalos[i]['xv0'][2]**2))
        m0 = float(run.m0)
        if res.t.size:
            t_last, r_last, m_last = float(res.t[-1]), float(res.r[-1]), float(res.m[-1])
        else:
            t_last, r_last, m_last = 0., r0, m0
        if exc is not None:
            # the engine carries the state it reached on the exception; prefer it over
            # the flushed result, which stops at the last accepted step
            t_last = float(getattr(exc, 't_last', t_last))
            r_last = float(getattr(exc, 'r_last', r_last))
            m_last = float(getattr(exc, 'm_last', m_last))
        v = res.xv[:, 3:] if res.xv is not None and res.xv.size else np.zeros((0, 3))
        speed = np.sqrt((v**2).sum(axis=1)) if v.size else np.zeros(0)
        v_last = float(speed[-1]) if speed.size else 0.
        status = _status(run, res, exc)
        held = status in ('frozen', 'parked')

        out['status'][i] = status
        out['n_step'][i] = int(res.t.size)
        out['t_last'][i], out['r_last'][i] = t_last, r_last
        out['m_final'][i], out['m_init'][i] = m_last, m0
        out['vmax'][i] = float(res.vmax[-1]) if res.vmax.size else run.vmax0
        out['rmax'][i] = float(res.rmax[-1]) if res.rmax.size else run.rmax0
        out['r_snaps'][i] = _snap_track(tsnap, res.t, res.r, r0, r_last)
        out['m_snaps'][i] = _snap_track(tsnap, res.t, res.m, m0, m_last)
        xv0 = np.asarray(subhalos[i]['xv0'], float)
        v0 = float(np.sqrt(xv0[3]**2 + xv0[4]**2 + xv0[5]**2))
        out['v_snaps'][i] = _snap_track(tsnap, res.t[:speed.size], speed, v0,
                                        0. if held else v_last)
        # a held subhalo stopped being integrated but is still in the host: keep it
        # depositing at its settled radius for the rest of the run
        if dens_on and hold_terminal and (held or exc is not None) and t_last < tsnap[-1]:
            _hold_residence(pop_hist, np.asarray(dens_redges, float), tsnap,
                            r_last, m_last, t_last, hist_k=pop_hist_k,
                            kernels=dens_kernels, v=v_last)

    if results is not None:
        out['results'] = results
    return out


def clump_field(out, redges, host=None):
    """The population's own field on the snapshot grid, from an evolve_population result:
    enclosed clump mass M_c(<r, t), clump mass density rho_c(r, t) in the shells of
    redges, and the mass-weighted rms speed sigma_c(r, t) in each shell. Shapes
    (n_rbin + 1, n_snap) for the enclosed mass (evaluated on the edges) and
    (n_rbin, n_snap) for the other two.

    This is the quantity a realization has and an orbit-averaged model does not: with
    host given, 'frac' carries M_c(<r)/M_host(<r), whose excursion above 1 is what a
    kinematic mass-segregation bound tests.
    """
    redges = np.asarray(redges, float)
    r, m = out['r_snaps'], out['m_snaps']
    v = out['v_snaps']
    nr, nt = redges.size - 1, out['tsnap'].size
    M_enc = np.zeros((nr + 1, nt))
    rho = np.zeros((nr, nt))
    sig = np.zeros((nr, nt))
    shell = cfg.FourPi / 3. * (redges[1:]**3 - redges[:-1]**3)
    for j in range(nt):
        idx = np.searchsorted(redges, r[:, j]) - 1
        for k in range(nr):
            sel = idx == k
            mk = m[sel, j]
            rho[k, j] = mk.sum() / shell[k]
            if mk.sum() > 0.:
                sig[k, j] = np.sqrt((mk * v[sel, j]**2).sum() / mk.sum())
        # enclosed on the edges: everything inside redges[e], end bins inclusive
        M_enc[:, j] = [m[r[:, j] < e, j].sum() for e in redges]
    field = dict(M_enc=M_enc, rho=rho, sigma=sig, redges=redges, tsnap=out['tsnap'])
    if host is not None:
        with np.errstate(divide='ignore', invalid='ignore'):
            field['frac'] = M_enc / np.asarray(host.M(redges), float)[:, None]
    return field
