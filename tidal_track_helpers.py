# Shared helpers for the tidal-track universality notebook. Lives at the
# module level (not in a notebook cell) so multiprocessing.Pool workers can
# import and call it.

import numpy as np
from scipy.optimize import brentq
import config as cfg
from profiles import NFW, Vcirc
from orbit import orbit
from subhalo_functions import NumericProfile
import stripping_common as sc


def eta_for_rp_ra(host, R0, rp_ra):
    """Circularity eta = Vphi/Vcirc at apocentre R0 that yields an orbit with
    pericentre/apocentre ratio rp_ra in `host`. Du+24's NFW (gamma=1) tidal
    tracks use rp/ra = 1/5 and 1/20 (their Fig. 1, lower-left panel)."""
    T_circ = 2. * np.pi * R0 / Vcirc(host, R0, 0.)
    t = np.linspace(0., 3. * T_circ, 6000)

    def ratio_err(eta):
        xv0, _ = sc.make_orbit(host, R0=R0, eta=eta)
        o = orbit(xv0)
        o.integrate(t, host)
        r = np.sqrt(o.xvArray[:, 0]**2 + o.xvArray[:, 2]**2)
        return r.min() / r.max() - rp_ra

    return float(brentq(ratio_err, 0.02, 0.97, xtol=1e-4))  # type: ignore[arg-type]


def evolve_one(params):
    """Evolve one (subhalo, host, orbit) configuration with the Benson+Du22
    calibration. params is a dict with M_sub, c_sub, M_host, c_host, and
    optionally R0_frac (apocentre / r_vir,host) and Rp_Ra (pericentre /
    apocentre ratio). The orbit defaults to Du+24's NFW gamma=1 case,
    Rp/Ra = 1/20. Returns a dict with the inputs plus (R, V, m, t) arrays."""
    M_sub = params['M_sub']
    c_sub = params['c_sub']
    M_host = params['M_host']
    c_host = params['c_host']
    Rp_Ra = params.get('Rp_Ra', 1./20.)
    R0_frac = params.get('R0_frac', 0.7)
    cfg.Mres = max(1e-3 * M_sub, 1e1)
    hNFW = NFW(M_host, c_host)
    sat = NFW(M_sub, c_sub)
    rvals = np.logspace(np.log10(cfg.Rres), np.log10(10.*sat.rs), 200)
    M_sub_arr = sat.M(rvals)
    R0 = R0_frac * hNFW.rh
    eta = eta_for_rp_ra(hNFW, R0, Rp_Ra)
    xv0, _ = sc.make_orbit(hNFW, R0=R0, eta=eta)
    # Pullen+14 SIS-host heating coefficients with the Benson+Du22
    # second-order term. Matches the stripping_sis and stripping_nfw
    # notebooks; second-order brings the NFW track onto Du+24 eq. 19 within
    # a few percent over r/r_0 in [0.15, 1]. t_dyn_mode='sub_lt' strips on
    # T_dyn,sub(lt) (the tidal-tensor cumulant always decays on the host
    # orbital time); this exposes the post-pericentre quiescent stairsteps
    # in (rmax, vmax) that flag the transition into deep stripping.
    res = sc.evolve_heating(
        hNFW, NumericProfile(rvals, M_sub_arr), xv0,
        tmax=30., Nstep=30000,
        epsh=3., gamma=1.5, beta_h=1., alpha=1.,
        second_order=True,
        t_dyn_mode='sub_lt',
    )
    mask = np.isfinite(res.rmax) & np.isfinite(res.vmax)
    return {
        **params,
        'Rp_Ra': Rp_Ra, 'eta': eta,
        'R': res.rmax[mask] / res.rmax0,
        'V': res.vmax[mask] / res.vmax0,
        'rmax0': res.rmax0, 'vmax0': res.vmax0,
        'm_final_over_M_tot': res.m[res.m > 0][-1] / M_sub_arr[-1],
    }


def du24_eq19(x, mu_V=0.6175, eta_V=0.2895, mu_R=0.5529, eta_R=0.4675):
    """Du+24 eq. (19) tidal track for NFW subhalo (gamma=1) on NFW host.
    Parameters are Table I (alpha, beta, gamma) = (1, 3, 1).
    Returns (R_max/R_max0, V_max/V_max0) parametric in bound mass fraction."""
    V = 2**mu_V * x**eta_V / (1.0 + x)**mu_V
    R = 2**mu_R * x**eta_R / (1.0 + x)**mu_R
    return R, V


def du24_V_of_R(R):
    """Du+24 eq.19 V_max/V_max0 at given R_max/R_max0, parameterized via x."""
    x_grid = np.logspace(-6, 0, 10000)
    R_grid, V_grid = du24_eq19(x_grid)
    order = np.argsort(R_grid)
    return np.interp(R, R_grid[order], V_grid[order])


def median_track_ratio(R, V, lo=0.1, hi=1.0):
    """Median (V/V_Du+24) over R in [lo, hi]."""
    k = (R >= lo) & (R <= hi)
    if not k.any():
        return float('nan')
    return float(np.median(V[k] / du24_V_of_R(R[k])))
