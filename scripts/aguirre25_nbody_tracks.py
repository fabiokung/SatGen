"""Extract Aguirre-Santaella+25 N-body V_max/r_max tidal tracks.

Idealized N-body of a single NFW subhalo orbiting a Milky-Way host (NFW halo +
stellar/gas disc + bulge), from Aguirre-Santaella, Sanchez-Conde & Ogiya 2025
(arXiv:2506.01152). Each run dir holds per-snapshot enclosed-mass profiles
(radprof_NNN.txt: r, rho, M(<r)) and a subhalo_evolution.txt time series
(t_lookback, z, x/Rhost, v, f_b, R_h/R_sub). V_max/r_max come from the
enclosed-mass profiles: V_circ(r) = sqrt(M(<r)/r), peak interpolated -- same
construction as the DASH tracks.

    python scripts/aguirre25_nbody_tracks.py

Writes etc/calibration_runs/aguirre25_nbody_tracks.pkl keyed by run id, each a
dict(params, t_lookback, z, fb, V=V_max/V_max0, R=r_max/r_max0).
"""
import os
import pickle
import re
import sys
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'etc',
                                       'aguirre25_nbody'))
ZIP = os.path.join(DATADIR, 'tidal_tracks_subhaloes.zip')
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'etc',
                                   'calibration_runs', 'aguirre25_nbody_tracks.pkl'))

# Paper's convergence floor: tracks unreliable below ~3000 particles in r_max,
# i.e. f_b ~ 10^-3.5 for the N=2^25 runs (Aguirre-Santaella+25 sec. 2).
FB_FLOOR = 10**-3.5

# fulld c10 x0p8 e0p1 N25 z1 m1e6 i90 paral16mar14aGH; 'p' is the decimal point.
# The d/s prefix is an internal batch tag, not a physical distinction -- all
# runs are NFW subhaloes in the same disc+bulge MW host.
NAME = re.compile(
    r'^full(?P<batch>[ds])c(?P<c>[0-9p]+)x(?P<xc>[0-9p]+)e(?P<eta>[0-9p]+)'
    r'N(?P<N>\d+)z(?P<z>[0-9p]+)m(?P<m>[0-9pe]+)i(?P<incl>\d+)')


def parse(name):
    m = NAME.match(name)
    if m is None:
        return None
    f = lambda s: float(s.replace('p', '.'))
    return dict(batch=m['batch'], c=f(m['c']), xc=f(m['xc']), eta=f(m['eta']),
                logN=int(m['N']), zacc=f(m['z']), msub=f(m['m']),
                incl=f(m['incl']))


def vmax_rmax(r, M):
    """Peak of V_circ = sqrt(M(<r)/r), parabola-interpolated in log r."""
    good = (r > 0.) & (M > 0.)
    r, M = r[good], M[good]
    if len(r) < 4:
        return np.nan, np.nan
    vc = np.sqrt(M / r)
    k = int(np.argmax(vc))
    if 0 < k < len(r) - 1:
        x, y = np.log(r[k - 1:k + 2]), vc[k - 1:k + 2]
        den = (x[0] - x[1]) * (x[0] - x[2]) * (x[1] - x[2])
        a = (x[2] * (y[1] - y[0]) + x[1] * (y[0] - y[2]) + x[0] * (y[2] - y[1])) / den
        b = (x[2]**2 * (y[0] - y[1]) + x[1]**2 * (y[2] - y[0])
             + x[0]**2 * (y[1] - y[2])) / den
        if a < 0.:
            xv = -b / (2. * a)
            c = y[0] - a * x[0]**2 - b * x[0]
            return float(a * xv**2 + b * xv + c), float(np.exp(xv))
    return float(vc[k]), float(r[k])


def track(zf, run, members):
    """V_max/r_max per snapshot, paired with f_b(t) from subhalo_evolution.txt."""
    ev = np.genfromtxt(zf.open(members['evo']))
    t, z, fb = ev[:, 0], ev[:, 1], ev[:, 8]

    prof = members['prof']  # snapshot index -> radprof member name
    vr = np.full((len(prof), 2), np.nan)
    for i in sorted(prof):
        d = np.genfromtxt(zf.open(prof[i]))
        vr[i] = vmax_rmax(d[:, 0], d[:, 2])

    n = min(len(t), len(vr))
    v0, r0 = vr[0]
    return dict(params=parse(run), t_lookback=t[:n], z=z[:n], fb=fb[:n],
                V=vr[:n, 0] / v0, R=vr[:n, 1] / r0)


def main():
    runs = {}
    with zipfile.ZipFile(ZIP) as zf:
        for name in zf.namelist():
            parts = name.split('/')
            if '__MACOSX' in parts or len(parts) < 3:
                continue
            run = parts[-2]
            if not run.startswith('full'):
                continue
            leaf = parts[-1]
            members = runs.setdefault(run, {'prof': {}})
            if leaf == 'subhalo_evolution.txt':
                members['evo'] = name
            else:
                m = re.match(r'radprof_(\d+)\.txt$', leaf)
                if m:
                    members['prof'][int(m[1])] = name

        tracks = {}
        for run, members in sorted(runs.items()):
            if 'evo' not in members or not members['prof']:
                continue
            tracks[run] = track(zf, run, members)
            p = tracks[run]['params']
            print(f"  {run}: {len(tracks[run]['fb'])} snaps, "
                  f"c={p['c']:g} xc={p['xc']:g} eta={p['eta']:g} "
                  f"fb_end={tracks[run]['fb'][-1]:.2e}", flush=True)

    assert tracks, f"no runs found in {ZIP}"
    pickle.dump(dict(tracks=tracks, fb_floor=FB_FLOOR), open(OUT, 'wb'))
    print(f"wrote {len(tracks)} tracks -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
