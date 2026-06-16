"""Where does the tidal-tail join sit in N-body, in units of l_t? (DASH)

For each stripped DASH snapshot we have the bound mass profile M(<r) and the
subhalo's orbital phase-space. The model's join knob is tail_xi = log10(r_join /
l_t), with l_t the King62 tidal radius computed at the current orbital position
(exactly what evolve_heating does each step). So we measure both from N-body:

  l_t       -- ev.ltidal(NumericProfile, [host], xv, 'King62'), same call the
               stripping loop makes.
  r_te      -- the NFW-departure scale, fit the same way Du+24 define it (and
               scripts/du24_tail_join.py reads it for N-GADGET): the stripped
               density divided by the unstripped NFW is a transfer function
               H = f_t/(1+(r/r_te)^delta), and r_te is its turnover. This is the
               point where rho peels off the bound NFW onto the steeper tail --
               i.e. the C1 join the model's tail_xi anchors on. The local slope
               there is the NFW slope (~ -2.5..-3), NOT -tail_n.

xi = log10(r_te / l_t). A fixed-slope r_break (at -4, -5) sits DOWNSTREAM of the
join -- it's the tail asymptote, not the join -- so we keep it only as a
diagnostic for comparison. The unstripped subhalo is NFW(M_vir,s, c_s) with c_s
from the DASH index. DASH is scale-free; we anchor an arbitrary host (M_vir,h =
1e12, c_h = 10, the DASH value) -- xi is a ratio, scale drops.

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
from scipy.optimize import least_squares

import config as cfg
import evolve as ev
from profiles import NFW
from subhalo_functions import NumericProfile

MV_H, C_H, RATIO = 1e12, 10.0, 1000.0          # DASH: M_vir,h/M_vir,s = 1000, c_h = 10
HOST = NFW(MV_H, C_H)
RVIR_H = HOST.rh
VVIR_H = HOST.Vcirc(RVIR_H)                     # kpc/Gyr
RVIR_S = RVIR_H * (1. / RATIO)**(1. / 3.)
MV_S = MV_H / RATIO

FB_LO, FB_HI = 0.02, 0.5                        # stripped but resolved
SLOPES = (-4., -5.)
DELTA = 3.                                      # Du+24 cuspy transfer-fn index


def xv_cyl(pos, vel):
    """DASH com (pos in r_vir,h, vel in v_vir,h) -> SatGen [R,phi,z,VR,Vphi,Vz]."""
    x, y, z = pos * RVIR_H
    vx, vy, vz = vel * VVIR_H
    R = np.hypot(x, y)
    return np.array([R, np.arctan2(y, x), z,
                     (x * vx + y * vy) / R, (x * vy - y * vx) / R, vz])


def break_radius(rmid, rho, slope):
    """Outermost-going first crossing of d ln rho/d ln r = slope, interpolated."""
    ok = rho > 0
    rmid, rho = rmid[ok], rho[ok]
    if rmid.size < 5:
        return np.nan
    lnr = np.log(rmid)
    sl = np.gradient(np.log(rho), lnr)
    below = np.where(sl <= slope)[0]
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return rmid[0]
    f = (slope - sl[i - 1]) / (sl[i] - sl[i - 1])
    return float(np.exp(lnr[i - 1] + f * (lnr[i] - lnr[i - 1])))


def transfer_rte(r, rho, rho_nfw):
    """NFW-departure scale r_te from the transfer function H = rho/rho_nfw.

    Fit H(r) = f_t/(1+(r/r_te)^DELTA) in log space (DELTA fixed at the Du+24
    cuspy value, both params else free) and return r_te -- the turnover where the
    stripped profile peels off the unstripped NFW. This is the C1 join the model
    anchors tail_xi on, measured the same way as the N-GADGET r_te. nan if the
    resolved range is too short to constrain it.
    """
    H = rho / rho_nfw
    ok = np.isfinite(H) & (H > 0.) & (r > 0.)
    r, H = r[ok], H[ok]
    if r.size < 6:
        return np.nan
    lnH = np.log(H)
    # guesses: f_t ~ inner plateau, r_te ~ where H first drops to f_t/2
    ft0 = np.median(H[:max(3, r.size // 3)])
    half = np.where(H < 0.5 * ft0)[0]
    rte0 = r[half[0]] if half.size else np.sqrt(r[0] * r[-1])

    def resid(p):
        lft, lrte = p
        return lft - np.log1p((r / np.exp(lrte))**DELTA) - lnH

    sol = least_squares(resid, [np.log(max(ft0, 1e-3)), np.log(rte0)],
                        bounds=([np.log(1e-3), np.log(r[0])],
                                [np.log(5.), np.log(r[-1])]), max_nfev=200)
    return float(np.exp(sol.x[1]))


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


def one_snapshot(bins, m_row, rho_row, pos, vel, sub_nfw, lt_choice='King62'):
    """xi at the requested slopes for one snapshot, or None if unusable.

    M(<r) (m_row) builds the NumericProfile for l_t; the simulation's binned
    density (rho_row) gives the break slope directly (differentiating the coarse
    M is noisy).
    """
    M_phys = m_row * MV_S
    r = bins * RVIR_S
    grow = np.where(np.diff(M_phys) > 1e-6 * M_phys[-1])[0]
    if grow.size < 6:
        return None
    hi = grow[-1] + 2                           # keep one flat point past the edge
    sub = NumericProfile(r[:hi], M_phys[:hi])
    lt = ev.ltidal(sub, [HOST], xv_cyl(pos, vel), lt_choice)
    if not np.isfinite(lt) or lt <= cfg.Rres:
        return None
    rho = rho_row * MV_S / RVIR_S**3            # to Msun/kpc^3 (rho in M_vir,s/r_vir,s^3)
    out = dict(lt=lt, r_orb=np.linalg.norm(pos) * RVIR_H)
    # primary: NFW-departure r_te = the actual C1 join (apples-to-apples w/ N-GADGET)
    rte = (transfer_rte(r[:hi], rho[:hi], sub_nfw.rho(r[:hi]))
           if sub_nfw is not None else np.nan)
    out['rte'] = rte
    out['xi_rte'] = np.log10(rte / lt) if np.isfinite(rte) else np.nan
    # asymptotic outer slope: power-law index of the resolved tail beyond the join
    r_edge = r[hi - 2]                          # last bin where M is still growing
    out['tail_slope'] = (tail_slope(r, rho, rte, r_edge)
                         if np.isfinite(rte) else np.nan)
    # diagnostics: fixed-slope crossings, kept only to show they sit past the join
    for s in SLOPES:
        rb = break_radius(r, rho, s)
        out[f'xi{int(-s)}'] = (np.log10(rb / lt) if np.isfinite(rb) else np.nan)
        out[f'rb{int(-s)}'] = rb
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
          f"f_b in [{FB_LO},{FB_HI}], l_t={lt_choice}", flush=True)

    recs = []
    for k, d in enumerate(runs):
        try:
            M = np.loadtxt(os.path.join(d, 'radprof_m.txt'))
            rho = np.loadtxt(os.path.join(d, 'radprof_rho.txt'))
            evo = np.loadtxt(os.path.join(d, 'subhalo_evo.txt'))
        except (OSError, ValueError):
            continue
        meta = by_dir.get(os.path.relpath(os.path.abspath(d), repo_root), {})
        cs = meta.get('cs', np.nan)
        sub_nfw = NFW(MV_S, cs) if np.isfinite(cs) and cs > 0. else None
        bins = M[0, 1:]
        fb = evo[:, 7]
        sel = np.where((fb > FB_LO) & (fb < FB_HI))[0]
        for sid in sel[::3]:                    # thin: every 3rd qualifying snapshot
            if 1 + sid >= M.shape[0] or 1 + sid >= rho.shape[0]:
                continue
            r = one_snapshot(bins, M[1 + sid, 1:], rho[1 + sid, 1:],
                             evo[sid, 1:4], evo[sid, 4:7], sub_nfw, lt_choice)
            if r is None:
                continue
            r.update(c_s=meta.get('cs', np.nan), eta=meta.get('eta', np.nan),
                     xc=meta.get('xc', np.nan), rapo=meta.get('rapo', np.nan),
                     fb=float(fb[sid]))
            recs.append(r)
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(runs)} runs, {len(recs)} snapshots", flush=True)

    out = os.path.join(os.path.dirname(__file__), '..', 'etc', 'calibration_runs',
                       f'dash_tail_join{suffix}.pkl')
    pickle.dump(recs, open(out, 'wb'))
    series = (('xi_rte (join)', np.array([r['xi_rte'] for r in recs])),
              ('xi(-4) diag', np.array([r['xi4'] for r in recs])),
              ('xi(-5) diag', np.array([r['xi5'] for r in recs])),
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
