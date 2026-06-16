"""Extract DASH N-body V_max/r_max tidal tracks.

The DASH subhalo_evo files carry only f_b and the half-mass radius, not the
structural parameters. V_max/r_max come from the enclosed-mass profiles
(radprof_m.txt): V_circ(r) = sqrt(M(<r)/r) in (V_vir,s, r_vir,s) units, peak
interpolated. Same construction GvdB+19 use for their Fig. 5/6.

    python scripts/dash_nbody_tracks.py

Writes etc/calibration_runs/dash_nbody_tracks.pkl keyed by (orbit_label, c_s),
each a dict(snap, V=V_max/V_max0, R=r_max/r_max0).
"""
import os
import pickle
import sys
import tarfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

DASHDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'etc', 'dash'))
TAR = os.path.join(DASHDIR, 'DASH_19.05.09.tar.gz')
IDX = os.path.join(DASHDIR, 'dash_index.pkl')
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'etc',
                                   'calibration_runs', 'dash_nbody_tracks.pkl'))

CH_NODE = 10.0
ORBITS = [('radial', 1.15, 0.2), ('intermediate', 1.0, 0.3), ('wide', 0.87, 0.5)]
CS_NODES = [12.5, 15.8, 19.9, 25.0, 31.5]


def lookup(idx, cs, xc, eta, ch=CH_NODE):
    for r in idx:
        if (abs(r['ch'] - ch) < 1e-6 and abs(r['cs'] - cs) < 1e-6
                and abs(r['xc'] - xc) < 1e-6 and abs(r['eta'] - eta) < 1e-6):
            return r
    return None


def vmax_rmax(rbins, Mrow):
    """Peak of V_circ = sqrt(M(<r)/r), parabola-interpolated in log r."""
    good = (Mrow > 0.) & (Mrow < 1e50)
    r, m = rbins[good], Mrow[good]
    if len(r) < 4:
        return np.nan, np.nan
    vc = np.sqrt(m / r)
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


def track(member_file):
    d = np.genfromtxt(member_file, comments='#')
    rbins, M, snap = d[0, 1:], d[1:, 1:], d[1:, 0]
    vr = np.array([vmax_rmax(rbins, M[i]) for i in range(len(M))])
    v0, r0 = vr[0]
    return dict(snap=snap, V=vr[:, 0] / v0, R=vr[:, 1] / r0)


def main():
    idx = pickle.load(open(IDX, 'rb'))
    # tar member name (DASH_19.05.09/.../radprof_m.txt) -> (label, c_s)
    want = {}
    for label, xc, eta in ORBITS:
        for cs in CS_NODES:
            rec = lookup(idx, cs, xc, eta)
            assert rec is not None, f"no DASH cell for {label} c_s={cs}"
            member = rec['file'].split('etc/dash/')[1].replace(
                'subhalo_evo.txt', 'radprof_m.txt')
            want[member] = (label, cs)

    tracks = {}
    with tarfile.open(TAR, 'r:gz') as tf:
        for m in tf:
            if m.name in want:
                tracks[want[m.name]] = track(tf.extractfile(m))
                print(f"  {want[m.name]}: {len(tracks[want[m.name]]['snap'])} snapshots",
                      flush=True)
    assert len(tracks) == len(want), f"got {len(tracks)} of {len(want)} cells"

    pickle.dump(dict(tracks=tracks, ch_node=CH_NODE, cs_nodes=CS_NODES),
                open(OUT, 'wb'))
    print(f"wrote {len(tracks)} tracks -> {OUT}", flush=True)


if __name__ == '__main__':
    main()
