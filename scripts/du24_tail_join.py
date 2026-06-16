"""Where does Du+24's own N-body truncation sit, in units of King62 l_t?

The DASH analysis (scripts/dash_tail_join.py) measures the join from the DASH
density profiles. This is the same question for the N-body Du+24 actually
calibrate against: their evolved-profile model is NFW x transfer function
H = f_t/(1+(r/r_te)^delta) (Du+24 eq. 23), with the effective tidal radius
r_te(x) fit to their simulations (eq. 30, Table II; gamma=1, delta=3 here).

We build that evolved profile at a grid of bound fractions x, then compute the
l_t of the same profile at the orbit's apocentre (Du+24 measure r_te at
apocentres). xi = log10(r_te / l_t), plus log10(r_break / l_t) where r_break is
the slope=-5 radius, to match the DASH definition. Self-contained: no cache, no
multiprocessing.

The l_t equation is selectable: 'King62' (yc=1, centrifugal term, default) or
'Tormen98' (yc=0, no centrifugal term -- the form Du+24 use). Non-default choices
write a suffixed pickle (du24_tail_join_<choice>.pkl).

    python scripts/du24_tail_join.py [lt_choice]
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import config as cfg
import evolve as ev
import stripping_common as sc
from profiles import NFW
from subhalo_functions import NumericProfile

cfg.Mres = 1e3

# Du+24 eq. 30 / Table II, (alpha,beta,gamma)=(1,3,1), delta=3 (cuspy NFW)
A_RTE, B_RTE, C_RTE = 0.9093, 0.6368, 2.185
D_FT, E_FT = 1.436, -0.2491
DELTA = 3.


def r_te_over_rvir(x):
    return (1. + A_RTE) * x**B_RTE / (1. + A_RTE * x**(2. * B_RTE)) \
        * np.exp(-C_RTE * (1. - x))


def f_t(x):
    return (1. + D_FT) * x**E_FT / (1. + D_FT * x**(2. * E_FT))


def evolved_profile(sub, x, rgrid):
    """NFW x Du+24 transfer function at bound fraction x -> NumericProfile."""
    rte = r_te_over_rvir(x) * sub.rh
    H = f_t(x) / (1. + (rgrid / rte)**DELTA)
    rho = sub.rho(rgrid) * H
    M = np.concatenate([[0.], np.cumsum(
        0.5 * (rho[1:] * 4 * np.pi * rgrid[1:]**2 + rho[:-1] * 4 * np.pi * rgrid[:-1]**2)
        * np.diff(rgrid))])
    return NumericProfile(rgrid, M), rte


def break_radius(rgrid, rho, slope=-5.):
    ok = rho > 0
    r, rho = rgrid[ok], rho[ok]
    lnr = np.log(r)
    sl = np.gradient(np.log(rho), lnr)
    below = np.where(sl <= slope)[0]
    if below.size == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return r[0]
    f = (slope - sl[i - 1]) / (sl[i] - sl[i - 1])
    return float(np.exp(lnr[i - 1] + f * (lnr[i] - lnr[i - 1])))


def main():
    lt_choice = sys.argv[1] if len(sys.argv) > 1 else 'King62'
    suffix = '' if lt_choice == 'King62' else '_' + lt_choice.lower()
    host = NFW(sc.DU24_MV_HOST, sc.DU24_C_HOST)
    sub = NFW(sc.DU24_MV_SUB, sc.DU24_C_SUB)
    rgrid = np.logspace(np.log10(cfg.Rres), np.log10(3. * sub.rh), 400)
    xs = np.logspace(np.log10(0.02), 0., 25)

    print(f"r_vir,sub = {sub.rh:.2f} kpc, c_sub = {sc.DU24_C_SUB}, l_t={lt_choice}")
    print(f"{'orbit':6s} {'x':>7s} {'r_te/rv':>8s} {'r_te':>7s} {'l_t':>7s} "
          f"{'xi_rte':>7s} {'xi_brk':>7s}")
    summary = {}
    for orbit in ('1/5', '1/20'):
        xv0, _ = sc.make_orbit(host, R0=0.7 * host.rh, z0=0.,
                               eta=sc.DU24_ETA[orbit])
        xi_rte, xi_brk = [], []
        for x in xs:
            prof, rte = evolved_profile(sub, x, rgrid)
            lt = ev.ltidal(prof, [host], xv0, lt_choice)
            if not np.isfinite(lt) or lt <= cfg.Rres:
                continue
            rho = prof.rho(rgrid)
            rb = break_radius(rgrid, rho, -5.)
            xr = np.log10(rte / lt)
            xb = np.log10(rb / lt) if np.isfinite(rb) else np.nan
            xi_rte.append(xr); xi_brk.append(xb)
            if x in (xs[0], xs[len(xs)//2], xs[-1]):
                print(f"{orbit:6s} {x:7.3f} {rte/sub.rh:8.3f} {rte:7.2f} "
                      f"{lt:7.2f} {xr:7.2f} {xb:7.2f}")
        xi_rte = np.array(xi_rte); xi_brk = np.array([v for v in xi_brk if np.isfinite(v)])
        summary[orbit] = dict(xi_rte=xi_rte, xi_brk=xi_brk)
        print(f"  -> {orbit}: xi_rte median={np.median(xi_rte):+.2f} "
              f"[{xi_rte.min():+.2f},{xi_rte.max():+.2f}]  "
              f"xi_brk median={np.median(xi_brk):+.2f} "
              f"[{xi_brk.min():+.2f},{xi_brk.max():+.2f}]")

    import pickle
    out = os.path.join(os.path.dirname(__file__), '..', 'etc',
                       'calibration_runs', f'du24_tail_join{suffix}.pkl')
    pickle.dump(summary, open(out, 'wb'))
    print(f"-> {out}")


if __name__ == '__main__':
    main()
