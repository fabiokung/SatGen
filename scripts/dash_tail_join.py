"""Where does the tidal-tail join sit in N-body, in units of l_t? (DASH)

For each stripped DASH snapshot we have the bound mass profile M(<r) and the
subhalo's orbital phase-space. The model's join knob is tail_xi = log10(r_join /
l_t), with l_t the King62 tidal radius computed at the current orbital position
(exactly what evolve_heating does each step). So we measure both from N-body:

  l_t   -- ev.ltidal(NumericProfile, [host], xv, 'King62'), same call the
           stripping loop makes, on the actual bound-mass profile. The host is
           the run's own NFW (its c_h from the DASH index), M_vir,h/M_vir,s=1000.
  r_te  -- the NFW-departure scale from Green & van den Bosch (2019), their
           concentration-dependent transfer function calibrated on DASH itself:
           H(r|f_b,c) = f_te/[1+(r(r_vir-r_te)/(r_vir r_te))^delta], with r_te,
           f_te, delta all functions of (f_b, c_s). profiles.Green.rte() returns
           r_te in physical units. This is the point where rho peels off the
           bound NFW onto the steeper tail -- the C1 join the model's tail_xi
           anchors on. Because Green+19 is fit to DASH, feeding it the DASH-index
           c_s and the DASH bound fraction f_b = M_bound/M_vir,s (subhalo_evo
           col. 8) is self-consistent with DASH's own Delta=200, M_vir,s=M(<r_vir,s)
           conventions (Ogiya+19 sec. 2.1.1).

xi = log10(r_te / l_t). DASH is scale-free (M_vir,h/M_vir,s = 1000, Delta=200
wrt critical); we anchor an arbitrary host mass M_vir,h = 1e12 -- xi is a length
ratio, the scale drops. The subhalo Green profile's r_vir,s equals bins*RVIR_S
by construction (same Delta, mass ratio), so r_te and l_t share one length unit.

The l_t equation is selectable: 'King62' (yc=1, centrifugal term, default) or
'Tormen98' (yc=0, no centrifugal term -- the form Du+24 use). Non-default choices
write a suffixed pickle (dash_tail_join_<choice>.pkl).

    python scripts/dash_tail_join.py [lt_choice]
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import config as cfg
import evolve as ev
from profiles import NFW, Green
from subhalo_functions import NumericProfile

MV_H, C_H, RATIO = 1e12, 10.0, 1000.0          # DASH: M_vir,h/M_vir,s = 1000
RVIR_H = NFW(MV_H, C_H).rh                      # Delta=200: independent of c_h
VVIR_H = NFW(MV_H, C_H).Vcirc(RVIR_H)           # sqrt(G M_vir,h/r_vir,h), c-independent
RVIR_S = RVIR_H * (1. / RATIO)**(1. / 3.)
MV_S = MV_H / RATIO

FB_LO, FB_HI = 0.02, 0.5                        # stripped but resolved


def xv_cyl(pos, vel):
    """DASH com (pos in r_vir,h, vel in v_vir,h) -> SatGen [R,phi,z,VR,Vphi,Vz]."""
    x, y, z = pos * RVIR_H
    vx, vy, vz = vel * VVIR_H
    R = np.hypot(x, y)
    return np.array([R, np.arctan2(y, x), z,
                     (x * vx + y * vy) / R, (x * vy - y * vx) / R, vz])


def tail_slope(r, rho, r_lo, r_hi):
    """Power-law index d ln rho/d ln r of the resolved outer tail in [r_lo, r_hi].

    The asymptote of the truncated power-law envelope (= -tail_n). Fit a line to
    ln rho vs ln r over the tail beyond the join, where it's a near-power-law
    before resolution noise takes over. nan if too few points span the range.
    """
    m = (r >= r_lo) & (r <= r_hi) & (rho > 0.)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(r[m]), np.log(rho[m]), 1)[0])


def one_snapshot(bins, m_row, rho_row, pos, vel, green, fb, host, lt_choice='King62'):
    """xi_rte for one snapshot, or None if unusable.

    M(<r) (m_row) builds the NumericProfile for l_t; r_te is Green+19's
    concentration-dependent NFW-departure scale at this snapshot's (f_b, c_s);
    the binned density (rho_row) gives the outer tail slope.
    """
    M_phys = m_row * MV_S
    r = bins * RVIR_S
    grow = np.where(np.diff(M_phys) > 1e-6 * M_phys[-1])[0]
    if grow.size < 6:
        return None
    hi = grow[-1] + 2                           # keep one flat point past the edge
    sub = NumericProfile(r[:hi], M_phys[:hi])
    lt = ev.ltidal(sub, [host], xv_cyl(pos, vel), lt_choice)
    if not np.isfinite(lt) or lt <= cfg.Rres:
        return None
    green.update_mass(fb * MV_S)
    rte = green.rte()
    rho = rho_row * MV_S / RVIR_S**3            # to Msun/kpc^3 (rho in M_vir,s/r_vir,s^3)
    out = dict(lt=lt, r_orb=np.linalg.norm(pos) * RVIR_H, rte=rte,
               xi_rte=np.log10(rte / lt) if np.isfinite(rte) and rte > 0. else np.nan)
    r_edge = r[hi - 2]                          # last bin where M is still growing
    out['tail_slope'] = (tail_slope(r, rho, rte, r_edge)
                         if np.isfinite(rte) else np.nan)
    return out


def main():
    import glob
    lt_choice = sys.argv[1] if len(sys.argv) > 1 else 'King62'
    suffix = '' if lt_choice == 'King62' else '_' + lt_choice.lower()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dashroot = os.path.join(os.path.dirname(__file__), '..', 'etc', 'dash',
                            'DASH_19.05.09')
    idx = pickle.load(open(os.path.join(os.path.dirname(__file__), '..',
                                        'etc', 'dash', 'dash_index.pkl'), 'rb'))
    by_dir = {os.path.dirname(e['file']): e for e in idx}
    # only the runs whose radial profiles were extracted from the tarball
    runs = sorted(os.path.dirname(p)
                  for p in glob.glob(os.path.join(dashroot, '*/*/*/*/radprof_rho.txt')))
    print(f"host: r_vir,h={RVIR_H:.1f} kpc, v_vir,h={VVIR_H:.1f} kpc/Gyr; "
          f"r_vir,s={RVIR_S:.2f} kpc", flush=True)
    print(f"scanning {len(runs)} DASH runs with profiles, "
          f"f_b in [{FB_LO},{FB_HI}], l_t={lt_choice}, r_te=Green+19", flush=True)

    recs = []
    for k, d in enumerate(runs):
        try:
            M = np.loadtxt(os.path.join(d, 'radprof_m.txt'))
            rho = np.loadtxt(os.path.join(d, 'radprof_rho.txt'))
            evo = np.loadtxt(os.path.join(d, 'subhalo_evo.txt'))
        except (OSError, ValueError):
            continue
        meta = by_dir.get(os.path.relpath(os.path.abspath(d), repo_root), {})
        cs, ch = meta.get('cs', np.nan), meta.get('ch', np.nan)
        if not (np.isfinite(cs) and cs > 0. and np.isfinite(ch) and ch > 0.):
            continue
        host = NFW(MV_H, ch)                     # this run's own host concentration
        green = Green(MV_S, cs)                  # Delta=200 == DASH; rh == RVIR_S
        bins = M[0, 1:]
        fb = evo[:, 7]
        sel = np.where((fb > FB_LO) & (fb < FB_HI))[0]
        for sid in sel[::3]:                    # thin: every 3rd qualifying snapshot
            if 1 + sid >= M.shape[0] or 1 + sid >= rho.shape[0]:
                continue
            r = one_snapshot(bins, M[1 + sid, 1:], rho[1 + sid, 1:],
                             evo[sid, 1:4], evo[sid, 4:7], green, float(fb[sid]),
                             host, lt_choice)
            if r is None:
                continue
            r.update(c_s=cs, c_h=ch, eta=meta.get('eta', np.nan),
                     xc=meta.get('xc', np.nan), rapo=meta.get('rapo', np.nan),
                     fb=float(fb[sid]))
            recs.append(r)
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(runs)} runs, {len(recs)} snapshots", flush=True)

    out = os.path.join(os.path.dirname(__file__), '..', 'etc', 'calibration_runs',
                       f'dash_tail_join{suffix}.pkl')
    pickle.dump(recs, open(out, 'wb'))
    series = (('xi_rte (join)', np.array([r['xi_rte'] for r in recs])),
              ('tail slope', np.array([r['tail_slope'] for r in recs])))
    for name, xi in series:
        x = xi[np.isfinite(xi)]
        print(f"{name:14s}: N={x.size}  median={np.median(x):+.2f}  "
              f"[16,84]=[{np.percentile(x,16):+.2f},{np.percentile(x,84):+.2f}]  "
              f"[5,95]=[{np.percentile(x,5):+.2f},{np.percentile(x,95):+.2f}]",
              flush=True)
    print(f"-> {out}", flush=True)


if __name__ == '__main__':
    main()
