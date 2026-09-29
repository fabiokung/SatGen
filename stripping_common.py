# host/subhalo setups, SatGen baseline evolvers, and plots for the stripping notebooks

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq

import config as cfg
import cosmo as co
import evolve as ev
from orbit import orbit
from profiles import NFW, Dekel, Green, Vcirc, tdyn
from subhalo_functions import NumericProfile, truncate_kazantzidis
from tidal_engine import EvolutionResult, PericentreUnresolvedError


def make_orbit(host, R0=1., z0=0., phi0=0., VR0=0., Vz0=0., eta=1.):
    """Build initial phase-space vector and orbit instance.

    eta: circularity = Vphi / Vcirc(r0). eta=1 is circular; eta < 1 is eccentric.
    """
    r0 = np.sqrt(R0**2 + z0**2)
    Vphi0 = eta * Vcirc(host, r0, 0.)
    xv0 = np.array([R0, phi0, z0, VR0, Vphi0, Vz0])
    return xv0, orbit(xv0)


# Du+24 Section II.A idealized setup (arXiv:2403.09597): NFW host 1e12 Msun,
# NFW subhalo 1e9 Msun -- the gamma_subhalo=1 reference for the stripping study.
DU24_MV_HOST, DU24_C_HOST = 1.0e12, 263.2 / 23.69   # rs=23.69, rvir=263.2 kpc (eqs.5-6)
DU24_MV_SUB, DU24_C_SUB = 1.0e9, 26.32 / 1.279      # rs=1.279, rvir=26.32 kpc (eqs.11-12)
# make_orbit eta = v_phi/v_c(r0) at apocentre, not the DASH circularity L/L_c(E)
DU24_ETA = {'1/5': 0.404, '1/20': 0.131}            # -> R_p/R_a

# Du+24 set Mvir/rvir via the virial overdensity, NOT SatGen's default Delta=200.
# Their host (eqs. 4-7) is rho0=3.797e6 Msun/kpc^3, rs=23.69, rvir=263.2, Mvir=1e12;
# building NFW(1e12, c) at Delta=200 puts the host at rvir=206, rs=18.6 (1.28x too
# compact) and shortens every dynamical time by 1.28^1.5 ~ 1.44x. Recover their
# overdensity from the stated host rvir: Delta = 3 Mvir / (4 pi rvir^3 rho_crit).
# rho_crit cancels in NFW's rho0 = rho_crit*Delta/3 * c^3/f(c) = Mvir/(4 pi rs^3 f(c)),
# so this reproduces Du+24's rho0/rs exactly, independent of the cfg cosmology. The
# subhalo's rvir (eq.12) implies the same Delta, so both share it.
DU24_DELTA = 3. * DU24_MV_HOST / (cfg.FourPi * 263.2**3
                                  * co.rhoc(0., cfg.h, cfg.Om, cfg.OL))


def load_du24_nbody(orbit):
    """Du+24 N-body mass-loss curve for orbit '1/5' or '1/20': (orbit_number,
    M_bound/M0, sigma_rel), apocenter samples from etc/du24_nbody
    (scripts/du24_nbody_reference.py). The clock is orbit number k = t/T_r (the
    period-normalized axis the model is compared on), not Gyr. The 1/5 orbit is the
    held-out generalization test (not in the calibration)."""
    tag = orbit.replace('/', '_')
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'etc', 'du24_nbody', f'mbound_t_{tag}.csv')
    a = np.loadtxt(fn, delimiter=',', skiprows=3)
    return a[:, 0], a[:, 1], a[:, 2]


def du24_nfw_setup(nr=200):
    """NFW host + subhalo on the Du+24 II.A ICs (virial overdensity DU24_DELTA,
    not Delta=200 -- matches Du+24's rho0/rs/rvir exactly).

    The subhalo IC is the full Du+24 profile: NFW within r_vir (eq. 8) plus the
    exponential outer truncation of eqs. 9-10 (r_decay = 0.1 r_vir), density- and
    slope-matched at r_vir. That tail carries ~16% of M_vir for the gamma=1 NFW
    (Du+24's M_sub = 1.02-1.2 M_vir), so a bare cut at r_vir leaves the bound-mass
    normalization ~16% low. truncate_kazantzidis is exactly eqs. 9-10 -- its
    kappa = r_t/r_decay + dln(rho)/dln(r)|_{r_t} is eq. 10. The NFW slope and density
    at r_vir are passed so the join is exact rather than finite-differenced off the
    NumericProfile's outer knot.

    Returns (host, sat, rvals, M_sub); sat is the analytic NFW, and the evolution
    starts from NumericProfile(rvals, M_sub).
    """
    host = NFW(DU24_MV_HOST, DU24_C_HOST, Delta=DU24_DELTA)
    sat = NFW(DU24_MV_SUB, DU24_C_SUB, Delta=DU24_DELTA)
    r_in = np.logspace(np.log10(cfg.Rres), np.log10(sat.rh), nr)
    nfw = NumericProfile(r_in, sat.M(r_in))
    sub = truncate_kazantzidis(nfw, r_t=sat.rh, r_decay=0.1 * sat.rh,
                               slope=-sat.s(sat.rh), rho_t=sat.rho(sat.rh))
    return host, sat, sub.ri, sub.Mr


# DASH (Ogiya+19, arXiv:1901.08601) scale-free ICs: analytic NFW host fixed at
# the origin, M_vir,h/M_vir,s = 1000, inner slope a_h=a_s=1. Everything is in
# units of (r_vir,h, v_vir,h, M_vir,s), so we anchor an arbitrary host (1e12,
# the fiducial c_h=10) and scale the dimensionless ICs by its r_vir, V_vir. All
# comparison observables are dimensionless (V/V0, R/R0 vs f_b; m/m0 vs t/T_r),
# so the host's Delta and cosmology cancel.
DASH_MV_HOST, DASH_C_HOST, DASH_RATIO = 1.0e12, 10.0, 1000.0


def _radial_period(host, xv0, tmax=40., n=8000):
    """Radial orbital period [Gyr] from peri-to-peri in the fixed host potential.

    DF-off -> param-independent, so callers compute this once per DASH cell and
    cache it as the clock for the m(t) comparison.
    """
    o = orbit(xv0)
    t = np.linspace(0., tmax, n)
    o.integrate(t, [host])
    r = np.hypot(o.xvArray[:, 0], o.xvArray[:, 2])
    mn = np.where((r[1:-1] < r[:-2]) & (r[1:-1] < r[2:]))[0] + 1
    if mn.size >= 2:
        return float(t[mn[1]] - t[mn[0]])
    mx = np.where((r[1:-1] > r[:-2]) & (r[1:-1] > r[2:]))[0] + 1
    if mx.size >= 2:
        return float(t[mx[1]] - t[mx[0]])
    raise ValueError("could not measure radial period (orbit too short for tmax)")


def dash_nfw_setup(cell, nr=200):
    """NFW host + subhalo + orbit on a DASH cell (subhalo_evo row-0 phase space).

    cell: a dash_index.pkl record (uses cell['cs'] and cell['file']). Returns
    (host, sat, rvals, M_sub, xv0, T_r). T_r is the radial period [Gyr]. The
    orbit ICs come from the actual N-body row 0 (com pos in r_vir,h, vel in
    v_vir,h), not an (xc, eta) reconstruction.
    """
    host = NFW(DASH_MV_HOST, DASH_C_HOST)
    rvir_h, vvir_h = host.rh, host.Vcirc(host.rh)
    sat = NFW(DASH_MV_HOST / DASH_RATIO, cell['cs'])
    rvals = np.logspace(np.log10(cfg.Rres), np.log10(sat.rh), nr)
    evo0 = np.loadtxt(cell['file'], comments='#')[0]
    pos, vel = evo0[1:4] * rvir_h, evo0[4:7] * vvir_h
    R = np.hypot(pos[0], pos[1])
    xv0 = np.array([R, np.arctan2(pos[1], pos[0]), pos[2],
                    (pos[0] * vel[0] + pos[1] * vel[1]) / R,
                    (pos[0] * vel[1] - pos[1] * vel[0]) / R, vel[2]])
    return host, sat, rvals, sat.M(rvals), xv0, _radial_period(host, xv0)


# Drakos+2020 (arXiv:2003.09452) idealized setup: NFW satellite (c=10, truncated
# at r_vir=10 r_s) in a fixed NFW host (c_host=10) with the same mean density
# within r_vir as the satellite, so r_vir,h/r_s = 10 (Mhost/Msat)^(1/3). Only the
# mass ratio, c=10, and the orbit enter the (scale-free) dynamics, so we anchor an
# arbitrary satellite mass and read the host mass off the ratio. Building both NFW
# at the same Delta (SatGen default) reproduces the equal-density condition, so
# r_vir,h/r_s comes out at Table 1's value independent of the cosmology. Satellite
# units for the orbit: vunit = sqrt(G Msat/r_s), and the digitized clock is t/torb.
DRAKOS_MV_SAT, DRAKOS_C = 1.0e9, 10.0


def drakos_nfw_setup(row, msat=DRAKOS_MV_SAT, nr=200):
    """NFW host + subhalo + orbit for one Drakos+2020 orbit (orbits.csv row).

    row: a dict from etc/drakos20_nbody/orbits.csv (uses Mhost_over_Msat,
    ra_over_rs, va_over_vunit, torb_over_tunit). The subhalo IC is the c=10 NFW
    sharply truncated at r_vir (as in dash_nfw_setup -- ICICLE ICs remove unbound
    particles, close to a lowered NFW rather than an outer exponential tail). The
    orbit is set at apocenter: r = ra, v_r = 0, v_phi = va. Returns (host, sat,
    rvals, M_sub, xv0, T_r); T_r is the radial period [Gyr] measured in the fixed
    host potential (matches Table 1's torb/tunit to ~1%).
    """
    sat = NFW(msat, DRAKOS_C)
    rs = sat.rh / DRAKOS_C
    host = NFW(msat * row['Mhost_over_Msat'], DRAKOS_C)
    rvals = np.logspace(np.log10(cfg.Rres), np.log10(sat.rh), nr)
    vunit = np.sqrt(cfg.G * msat / rs)
    tunit = np.sqrt(rs**3 / (cfg.G * msat))
    xv0 = np.array([row['ra_over_rs'] * rs, 0., 0.,
                    0., row['va_over_vunit'] * vunit, 0.])
    # size the period-search window off the paper's torb so the very wide S5
    # orbit (torb ~ 1800 tunit) still spans two pericentres.
    tmax = 2.5 * row['torb_over_tunit'] * tunit
    return host, sat, rvals, sat.M(rvals), xv0, _radial_period(host, xv0, tmax=tmax)


def _vmax_rmax(profile):
    # root of 4*pi*r^3*rho(r) = M(r); more stable than minimize_scalar on
    # the DASH table which can have a very flat Vcirc peak
    def f(r):
        return 4.0 * np.pi * r**3 * profile.rho(r) - profile.M(r)

    rr = np.logspace(np.log10(cfg.Rres), np.log10(profile.rh * 0.99), 80)
    fv = np.array([f(r) for r in rr])
    sign_changes = np.where(fv[:-1] * fv[1:] < 0)[0]

    if len(sign_changes) == 0:
        vc = np.sqrt(np.maximum(cfg.G * profile.M(rr) / rr, 0.))
        idx = int(np.argmax(vc))
        return float(vc[idx]), float(rr[idx])

    idx = sign_changes[-1]
    rmax = float(brentq(f, rr[idx], rr[idx + 1], xtol=1e-8))  # type: ignore[arg-type]
    vmax = float(np.sqrt(cfg.G * profile.M(rmax) / rmax))
    return vmax, rmax


# ---evolution routines

def evolve_satgen_dekel(host, sat, xv0, tmax=10., Nstep=10000, alpha=1.,
                        n_snapshots=10, label='SatGen (Dekel / P10 track)'):
    """Dekel/P10 tidal-track evolution."""
    assert cfg.Mres is not None, "cfg.Mres must be set before calling evolve_satgen_dekel"
    potential = host
    timesteps = np.linspace(0., tmax, Nstep + 1)[1:]
    mv0 = sat.Mh
    rmax0 = sat.rmax
    vmax0 = sat.Vmax
    aDekel0 = sat.alphah

    snap_npts = 100
    track_steps = np.round(np.linspace(0, Nstep - 1, n_snapshots)).astype(int)

    # NaN-init: unwritten slots (early-continue when r <= cfg.Rres) and
    # rmax slots without an interior crossing stay NaN; plotters mask via
    # np.isfinite. Using zeros would conflate "skipped" with "value of 0".
    t_arr = np.full(Nstep, np.nan)
    r_arr = np.full(Nstep, np.nan)
    m_arr = np.full(Nstep, np.nan)
    vmax_arr = np.full(Nstep, np.nan)
    rmax_arr = np.full(Nstep, np.nan)
    lt_arr = np.full(Nstep, np.nan)
    r_grids = np.zeros((n_snapshots, snap_npts))
    rho_snaps = np.zeros((n_snapshots, snap_npts))
    M_snaps = np.zeros((n_snapshots, snap_npts))

    s = sat
    o = orbit(xv0)
    r = np.sqrt(xv0[0]**2 + xv0[2]**2)
    m = mv0
    lt = cfg.Rres
    tprevious = 0.

    for i, t in enumerate(timesteps):
        dt = t - tprevious
        if r > cfg.Rres:
            o.integrate(t, potential, m)
            xv = o.xv
        else:
            tprevious = t
            continue
        r = np.sqrt(xv[0]**2 + xv[2]**2)

        a = s.alphah
        if m > cfg.Mres:
            m, lt = ev.msub(s, potential, xv, dt, choice='King62', alpha=alpha)
            c, Delta = ev.Dekel(m, mv0, rmax0, vmax0, aDekel0, z=0.)
        else:
            m, lt = cfg.Mres, cfg.Rres
            c, Delta = s.ch, 200.
        s = Dekel(m, c, a, Delta=Delta, z=0.)

        t_arr[i] = t
        r_arr[i] = r
        m_arr[i] = m
        vmax_arr[i] = s.Vmax
        rmax_arr[i] = s.rmax
        lt_arr[i] = lt

        slot = np.searchsorted(track_steps, i)
        if slot < n_snapshots and track_steps[slot] == i:
            rg = np.logspace(np.log10(cfg.Rres), np.log10(s.rh), snap_npts)
            r_grids[slot] = rg
            rho_snaps[slot] = s.rho(rg)
            M_snaps[slot] = s.M(rg)

        tprevious = t

    return EvolutionResult(
        t=t_arr, r=r_arr, m=m_arr, vmax=vmax_arr, rmax=rmax_arr, lt=lt_arr,
        r_grid=r_grids, rho_snapshots=rho_snaps, M_snapshots=M_snaps,
        snapshot_steps=track_steps, rmax0=rmax0, vmax0=vmax0, label=label,
    )


def _green_strip(s, potential, xv, dt, alpha):
    """One exactly-integrated King62 tidal strip on a Green profile. Returns (m, lt).

    Same closed form the heating engines use (evolve.msub takes a forward-Euler
    step instead): over a step lt -- hence M(<lt) and t_dyn -- is fixed, so the
    bound-mass excess m - M(<lt) relaxes as exp(-alpha dt / t_dyn) and asymptotes
    to M(<lt) from above, never crossing it, so m stays >= M(<lt) for any dt.
    t_dyn is the host dynamical time at the subhalo's position (GvdBJ21 eq.8,
    matching evolve.msub). King62 is fixed: alpha_from_c2 is DASH-calibrated only
    against the King62 lt (GvdBJ21 eq.9, centrifugal term included); a Tormen98 lt
    would strip weaker and need its own alpha refit.
    """
    lt = float(ev.ltidal(s, potential, xv, 'King62'))  # type: ignore[arg-type]
    if lt >= s.rh:
        return s.Mh, lt
    M_at_lt = min(float(s.M(lt)), float(s.Mh))
    T_strip = tdyn(potential, xv[0], xv[2])
    m = M_at_lt + (s.Mh - M_at_lt) * np.exp(-alpha * dt / T_strip)
    return max(m, cfg.Mres), lt  # type: ignore[type-var]


def evolve_satgen_green(host, ma, c2a, xv0, tmax=10.,
                        alpha='conc', Delta=200., z=0.,
                        step_frac=0.01, dt_growth_max=3.0, floor_orbit_frac=1e-3,
                        dt_abs_frac=1e-6, max_steps=200000,
                        n_snapshots=10, label='SatGen (Green / DASH track)'):
    """Green+21 DASH transfer-function evolution.

    alpha='conc' uses concentration-dependent stripping efficiency (ev.alpha_from_c2).
    The tidal radius is King62 -- alpha_from_c2 is DASH-calibrated against that lt
    definition (GvdBJ21 eq.9), so it is not a free choice here.

    step_frac sets the adaptive timestep dt <= step_frac * min(t_orb, t_orb/alpha),
    with t_orb the host dynamical time at the subhalo's position (the King62
    stripping timescale here). dt grows by at most dt_growth_max per step and is
    floored at max(dt_abs_frac*tmax, floor_orbit_frac*t_orb); max_steps caps the
    loop. The default 0.01 matches the heating engines used in the calibration.
    """
    assert cfg.Mres is not None, "cfg.Mres must be set before calling evolve_satgen_green"
    potential = host

    s = Green(ma, c2a, Delta=Delta, z=z)
    vmax0, rmax0 = _vmax_rmax(s)
    al = ev.alpha_from_c2(host.ch, s.ch) if alpha == 'conc' else float(alpha)

    snap_npts = 100
    snap_times = np.linspace(0., tmax, n_snapshots)
    snap_ptr = 0
    r_grids = np.zeros((n_snapshots, snap_npts))
    rho_snaps = np.zeros((n_snapshots, snap_npts))
    M_snaps = np.zeros((n_snapshots, snap_npts))
    snap_steps = np.zeros(n_snapshots, dtype=int)

    t_list, r_list, m_list = [], [], []
    vmax_list, rmax_list, lt_list = [], [], []

    o = orbit(xv0)
    r = np.sqrt(xv0[0]**2 + xv0[2]**2)
    m = ma
    lt = cfg.Rres
    dt_abs = dt_abs_frac * tmax
    dt_prev = np.inf
    t = 0.
    nstep = 0

    while t < tmax - dt_abs:
        if nstep >= max_steps:
            raise PericentreUnresolvedError(
                f"exceeded max_steps={max_steps} at t={t:.3f}/{tmax} Gyr")

        t_orb = tdyn(potential, r)
        dt_min = max(dt_abs, floor_orbit_frac * t_orb)
        dt = min(step_frac * t_orb, step_frac * t_orb / al,
                 dt_growth_max * dt_prev, tmax - t)
        dt = max(dt, dt_min)
        t += dt
        dt_prev = dt

        if r <= cfg.Rres:
            continue
        o.integrate(t, potential, m)
        r = np.sqrt(o.xv[0]**2 + o.xv[2]**2)

        if m <= cfg.Mres:
            m, lt = cfg.Mres, cfg.Rres
        else:
            m, lt = _green_strip(s, potential, o.xv, dt, al)
            s.update_mass(m)

        vm, rm = _vmax_rmax(s)
        t_list.append(t); r_list.append(r); m_list.append(m)
        vmax_list.append(vm); rmax_list.append(rm); lt_list.append(lt)
        nstep += 1

        # snapshot when the accepted time crosses each evenly-spaced threshold
        while snap_ptr < n_snapshots and t >= snap_times[snap_ptr]:
            rg = np.logspace(np.log10(cfg.Rres), np.log10(s.rh), snap_npts)
            r_grids[snap_ptr] = rg
            rho_snaps[snap_ptr] = s.rho(rg)
            M_snaps[snap_ptr] = s.M(rg)
            snap_steps[snap_ptr] = nstep - 1
            snap_ptr += 1

    # the loop halts just short of tmax, so the final snap_times[-1]==tmax threshold
    # never fires; fill any trailing slots with the final settled profile.
    while snap_ptr < n_snapshots and t_list:
        rg = np.logspace(np.log10(cfg.Rres), np.log10(s.rh), snap_npts)
        r_grids[snap_ptr] = rg
        rho_snaps[snap_ptr] = s.rho(rg)
        M_snaps[snap_ptr] = s.M(rg)
        snap_steps[snap_ptr] = nstep - 1
        snap_ptr += 1

    return EvolutionResult(
        t=np.asarray(t_list), r=np.asarray(r_list), m=np.asarray(m_list),
        vmax=np.asarray(vmax_list), rmax=np.asarray(rmax_list),
        lt=np.asarray(lt_list),
        r_grid=r_grids, rho_snapshots=rho_snaps, M_snapshots=M_snaps,
        snapshot_steps=snap_steps, rmax0=rmax0, vmax0=vmax0, label=label,
    )


# ---plotting

def _style_for(i, styles) -> dict:
    """Per-result kwargs override; default is dots connected by a thin
    line in time order so post-pericentre plateau-to-plateau jumps in the
    (rmax, vmax) tracks are visually traceable."""
    if styles is not None and i < len(styles) and styles[i] is not None:
        return dict(styles[i])
    return dict(marker='.', linestyle='-', ms=2, lw=0.4, alpha=0.5)


def markers_to_lines(handles):
    """Replace marker-only legend handles with line proxies of the same colour.
    Cosmetic only — the underlying scatter plot is unchanged. Useful for the
    (rmax, vmax) tracks where dots in the plot are clearer but a line in the
    legend is easier to read."""
    fixed = []
    for h in handles:
        if h.get_linestyle() == 'None' and h.get_marker() not in (None, 'None', ''):
            fixed.append(Line2D([], [], color=h.get_color(),
                                linewidth=2, label=h.get_label()))
        else:
            fixed.append(h)
    return fixed


def plot_tidal_track(results, ax=None, title='Tidal Tracks', styles=None, legend=True):
    if not isinstance(results, list):
        results = [results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for i, res in enumerate(results):
        # mask: drop NaN rmax slots (no interior Vc peak found) and the
        # NaN-initialised slots from skipped iterations
        mask = np.isfinite(res.rmax) & np.isfinite(res.vmax)
        ax.plot(res.rmax[mask] / res.rmax0, res.vmax[mask] / res.vmax0,
                label=res.label, **_style_for(i, styles))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r_{\rm max}/r_{\rm max,0}$')
    ax.set_ylabel(r'$V_{\rm max}/V_{\rm max,0}$')
    ax.set_title(title)
    if legend:
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=markers_to_lines(h), labels=l)
    return ax


def plot_mass_loss(results, ax=None, title='Mass Loss History', styles=None, legend=True):
    if not isinstance(results, list):
        results = [results]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for i, res in enumerate(results):
        # drop NaN-initialised skipped-iteration slots
        mask = np.isfinite(res.m)
        ax.plot(res.t[mask], res.m[mask],
                label=res.label, **_style_for(i, styles))
    ax.set_yscale('log')
    ax.set_xlabel('time [Gyr]')
    ax.set_ylabel(r'$m$ [$M_\odot$]')
    ax.set_title(title)
    if legend:
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=markers_to_lines(h), labels=l, fontsize=9)
    return ax


def profile_ylims(*results):
    """Shared y-axis limits across results, for use with plot_profile_snapshots."""
    rho_all, M_all, Vc_all = [], [], []
    for res in results:
        for r_grid, snap_rho, snap_M in zip(res.r_grid, res.rho_snapshots, res.M_snapshots):
            if np.all(snap_M == 0):
                continue
            Vc = np.sqrt(np.maximum(cfg.G * snap_M / r_grid, 0.))
            pos = snap_rho[snap_rho > 0]
            if len(pos):
                rho_all.append(pos.min())
                rho_all.append(pos.max())
            pos = snap_M[snap_M > 0]
            if len(pos):
                M_all.append(pos.min())
                M_all.append(pos.max())
            pos = Vc[Vc > 0]
            if len(pos):
                Vc_all.append(pos.min())
                Vc_all.append(pos.max())

    def _lims(vals):
        return (min(vals), max(vals)) if vals else (1e-10, 1e10)

    return [_lims(rho_all), _lims(M_all), _lims(Vc_all)]


def plot_profile_snapshots(result, axes=None, title_prefix='', ylims=None):
    """Plot rho(r), M(<r), Vc(r) at snapshot steps. ylims from profile_ylims()."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 4))
    n_snap = result.rho_snapshots.shape[0]
    cmap = plt.get_cmap('viridis')
    for i in range(n_snap):
        if np.all(result.M_snapshots[i] == 0):
            continue
        c = cmap(i / max(n_snap - 1, 1))
        rg = result.r_grid[i]
        rho = result.rho_snapshots[i]
        M = result.M_snapshots[i]
        Vc = np.sqrt(np.maximum(cfg.G * M / rg, 0.))
        lbl = f'step {result.snapshot_steps[i]}'
        axes[0].plot(rg, rho, color=c, label=lbl)
        axes[1].plot(rg, M, color=c, label=lbl)
        axes[2].plot(rg, Vc, color=c, label=lbl)
    labels = [r'$\rho(r)$ [$M_\odot/{\rm kpc}^3$]',
              r'$M(r)$ [$M_\odot$]',
              r'$V_{\rm c}(r)$ [kpc/Gyr]']
    titles = ['Density', 'Enclosed Mass', 'Circular Velocity']
    panel_ylims: list = ylims if ylims is not None else [None, None, None]
    for ax, ylab, ttl, ylim in zip(axes, labels, titles, panel_ylims):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$r$ [kpc]')
        ax.set_ylabel(ylab)
        ax.set_title(f'{title_prefix}{ttl}')
        if ylim is not None:
            ax.set_ylim(*ylim)
    return axes


def plot_orbit(results, ax=None, title='Orbital Radius', styles=None, legend=True):
    if not isinstance(results, list):
        results = [results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for i, res in enumerate(results):
        # drop NaN-initialised skipped-iteration slots
        mask = np.isfinite(res.r)
        kw = dict(styles[i]) if (styles is not None and i < len(styles)
                                 and styles[i] is not None) else {}
        ax.plot(res.t[mask], res.r[mask], label=res.label, **kw)
    ax.set_xlabel(r'$t$ [Gyr]')
    ax.set_ylabel(r'$r$ [kpc]')
    ax.set_title(title)
    if legend:
        ax.legend()
    return ax
