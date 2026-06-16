"""Content-addressed cache of tidal-stripping truncation runs.

Each evolution is keyed by (model, orbit, order, Nstep, calib, tail) and cached
as one pickle under etc/calibration_runs/truncation_cache/. `load_runs(specs)`
computes only the keys not already on disk (in parallel) and returns
{label: result}. This replaces the monolithic truncation_experiments.pkl:
changing one calibration recomputes only the runs at that calibration, and runs
shared across notebooks (e.g. galacticus at Du+24 Table IV, which never moves)
are computed once and reused.

A notebook pins its calibration explicitly:

    import truncation_runs as tr
    p50 = dict(alpha_s=..., eps_h=..., beta_h=..., f_2=..., gamma_h=...)
    specs = tr.standard_specs(p50, tail=dict(tail_n=..., tail_xi=...))
    res, clamps = tr.load_runs(specs)
"""
import hashlib
import json
import os
import pickle
import re
import time
import warnings
from multiprocessing import Pool

import numpy as np

import model_params as mp

CACHE_VERSION = 1                               # bump to invalidate on algorithm changes
NSTEP, TMAX = 37000, 37.                        # converged (notebook figures)
MODELS = ('hard', 'kazantzidis', 'powerlaw', 'galacticus')
ORBITS = ('1/5', '1/20')
DEFAULT_TAIL = dict(tail_n=5.0, tail_xi=0.0)     # evolve_heating powerlaw defaults

CACHEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'etc', 'calibration_runs', 'truncation_cache')


def label(spec):
    tag = '1st+2nd' if spec['second_order'] else '1st'
    return f"{spec['orbit']} {spec['model']} ({tag})"


def _canon(spec):
    """Everything that affects the result, rounded so float noise doesn't bust
    the cache."""
    def rnd(d):
        return {k: (round(v, 8) if isinstance(v, float) else v)
                for k, v in sorted(d.items())}
    return json.dumps(dict(v=CACHE_VERSION, model=spec['model'],
                           orbit=spec['orbit'], order=spec['second_order'],
                           nstep=spec['nstep'], tmax=spec['tmax'],
                           calib=rnd(spec['calib']), tail=rnd(spec['tail'])),
                      sort_keys=True)


def key(spec):
    return hashlib.sha1(_canon(spec).encode()).hexdigest()[:16]


def _cachefile(spec):
    safe = label(spec).replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    return os.path.join(CACHEDIR, f"{safe}_{key(spec)}.pkl")


def make_spec(model, orbit, second_order, calib, tail=None,
              nstep=NSTEP, tmax=TMAX):
    return dict(model=model, orbit=orbit, second_order=second_order,
                calib=dict(calib), tail=dict(tail or {}),
                nstep=nstep, tmax=tmax)


def standard_specs(calib, tail=None, calib_galacticus=None,
                   models=MODELS, orders=(False, True), nstep=NSTEP, tmax=TMAX):
    """The model x orbit x order grid for the notebooks.

    calib: the 5 heating params (alpha_s, eps_h, beta_h, f_2, gamma_h) for the
        profile-stripping models. tail (tail_n, tail_xi) attaches to powerlaw.
    calib_galacticus: heating params for the galacticus model (Du+24 Table IV by
        default -- Mode B runs at Du+24's own calibration). Ignored if galacticus
        is not in `models`.
    """
    cg = calib_galacticus or mp.DU24_TABLE_IV
    tail = tail or DEFAULT_TAIL
    specs = []
    for orbit in ORBITS:
        for model in models:
            c = _evolve_calib(cg if model == 'galacticus' else calib)
            t = tail if model == 'powerlaw' else None
            for order in orders:
                specs.append(make_spec(model, orbit, order, c, t, nstep, tmax))
    return specs


def _evolve_calib(p):
    """p50-style dict (alpha_s, eps_h, ...) -> evolve_heating kwargs."""
    return dict(epsh=p['eps_h'], gamma=p['gamma_h'], alpha=p['alpha_s'],
                beta_h=p['beta_h'], f2=p['f_2'], chi_v=mp.CHI_V,
                t_dyn_mode=mp.T_DYN_MODE)


def run_one(spec):
    """One evolution at converged Nstep, with shell-clamp telemetry. Module-level
    so the Pool can pickle it."""
    import config as cfg
    import stripping_common as sc
    import stripping_galacticus as sg
    from subhalo_functions import (NumericProfile, ShellClampWarning,
                                   truncate_kazantzidis)

    cfg.Mres = 1e3
    hNFW, sat, rvals, M_sub = sc.du24_nfw_setup()
    rvir = sat.rh
    xv0, _ = sc.make_orbit(hNFW, R0=0.7 * hNFW.rh, z0=0.,
                           eta=sc.DU24_ETA[spec['orbit']])
    kw = dict(tmax=spec['tmax'], Nstep=spec['nstep'],
              second_order=spec['second_order'], label=label(spec),
              **spec['calib'])

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ShellClampWarning)
        if spec['model'] == 'galacticus':
            s_rvir = -1. - 2. * sc.DU24_C_SUB / (1. + sc.DU24_C_SUB)
            infall = truncate_kazantzidis(NumericProfile(rvals, M_sub), rvir,
                                          r_decay=rvir, slope=s_rvir)
            res = sg.evolve_galacticus(hNFW, infall, xv0, rvir, **kw)
        else:
            res = sc.evolve_heating(hNFW, NumericProfile(rvals, M_sub), xv0,
                                    truncation=spec['model'], **kw, **spec['tail'])
    clamped = [w for w in caught if issubclass(w.category, ShellClampWarning)]
    shells = sum(int(re.search(r'clamped (\d+) shell', str(w.message)).group(1))  # type: ignore[union-attr]
                 for w in clamped)
    clamps = dict(steps=len(clamped), shells=shells)
    print(f"  {label(spec)}: {time.time() - t0:.0f} s, "
          f"clamped {clamps['steps']} steps / {clamps['shells']} shells", flush=True)
    return dict(spec=spec, result=res, clamps=clamps, key=key(spec))


def ensure_runs(specs, nproc=None):
    """Compute and cache any specs not already on disk."""
    os.makedirs(CACHEDIR, exist_ok=True)
    missing = [s for s in specs if not os.path.exists(_cachefile(s))]
    if not missing:
        return
    print(f"computing {len(missing)}/{len(specs)} uncached runs ...", flush=True)
    nproc = nproc or min(len(missing), os.cpu_count() or 4)
    with Pool(nproc) as pool:
        for rec in pool.map(run_one, missing):
            with open(_cachefile(rec['spec']), 'wb') as fh:
                pickle.dump(rec, fh)


def load_runs(specs, nproc=None):
    """{label: result}, {label: clamps} for specs, computing missing first."""
    ensure_runs(specs, nproc=nproc)
    results, clamps = {}, {}
    for s in specs:
        with open(_cachefile(s), 'rb') as fh:
            rec = pickle.load(fh)
        results[label(s)] = rec['result']
        clamps[label(s)] = rec['clamps']
    return results, clamps


def seed_from_pkl(pkl_path, calib, tail=None, calib_galacticus=None):
    """Backfill the cache from a legacy monolithic pkl so its 16 runs aren't
    recomputed. calib/tail/calib_galacticus must be the values that pkl was
    generated with, so the reconstructed keys match."""
    data = pickle.load(open(pkl_path, 'rb'))
    res, clamps = data['results'], data['clamp_counts']
    os.makedirs(CACHEDIR, exist_ok=True)
    n = 0
    for s in standard_specs(calib, tail=tail, calib_galacticus=calib_galacticus):
        lab = label(s)
        if lab not in res:
            continue
        f = _cachefile(s)
        if os.path.exists(f):
            continue
        with open(f, 'wb') as fh:
            pickle.dump(dict(spec=s, result=res[lab], clamps=clamps[lab],
                             key=key(s)), fh)
        n += 1
    print(f"seeded {n} runs from {os.path.basename(pkl_path)}")
    return n
