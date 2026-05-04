# shared evolution routines and plots for tidal stripping notebooks

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import bisect, brentq

import config as cfg
from profiles import NFW, Dekel, Green, Vcirc, tdyn
from orbit import orbit
import evolve as ev
from subhalo_functions import NumericProfile, heat_profile, tidalTensor


@dataclass
class EvolutionResult:
    t: np.ndarray  # time [Gyr]
    r: np.ndarray  # orbital radius [kpc]
    m: np.ndarray  # bound mass [Msun]
    vmax: np.ndarray  # Vmax [kpc/Gyr]
    rmax: np.ndarray  # rmax [kpc]
    lt: np.ndarray  # tidal radius [kpc]
    r_grid: np.ndarray  # radial grid for snapshots (fixed to initial extent)
    rho_snapshots: np.ndarray  # (n_snapshots, len(r_grid))
    M_snapshots: np.ndarray  # (n_snapshots, len(r_grid))
    snapshot_steps: np.ndarray  # step indices
    rmax0: float = 0.
    vmax0: float = 0.
    label: str = ''


def make_orbit(host, R0=1., z0=0., phi0=0., VR0=0., Vz0=0., eta=1.):
    """Build initial phase-space vector and orbit instance.

    eta: circularity = Vphi / Vcirc(r0). eta=1 is circular; eta < 1 is eccentric.
    """
    r0 = np.sqrt(R0**2 + z0**2)
    Vphi0 = eta * Vcirc(host, r0, 0.)
    xv0 = np.array([R0, phi0, z0, VR0, Vphi0, Vz0])
    return xv0, orbit(xv0)


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
        idx = np.argmax(vc)
        return vc[idx], rr[idx]

    idx = sign_changes[-1]
    rmax = brentq(f, rr[idx], rr[idx + 1], xtol=1e-8)
    vmax = np.sqrt(cfg.G * profile.M(rmax) / rmax)
    return vmax, rmax


#---evolution routines

def evolve_satgen_dekel(host, sat, xv0, tmax=10., Nstep=10000, alpha=1.,
                        n_snapshots=10, label='SatGen (Dekel / P10 track)'):
    """Dekel/P10 tidal-track evolution (Baseline A)."""
    potential = host
    timesteps = np.linspace(0., tmax, Nstep + 1)[1:]
    mv0 = sat.Mh
    rmax0 = sat.rmax
    vmax0 = sat.Vmax
    aDekel0 = sat.alphah

    r_grid = np.logspace(np.log10(cfg.Rres), np.log10(sat.rh), 100)
    track_steps = np.round(np.linspace(0, Nstep - 1, n_snapshots)).astype(int)

    t_arr = np.zeros(Nstep)
    r_arr = np.zeros(Nstep)
    m_arr = np.zeros(Nstep)
    vmax_arr = np.zeros(Nstep)
    rmax_arr = np.zeros(Nstep)
    lt_arr = np.zeros(Nstep)
    rho_snaps = np.zeros((n_snapshots, len(r_grid)))
    M_snaps = np.zeros((n_snapshots, len(r_grid)))

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

        if m > cfg.Mres:
            m, lt = ev.msub(s, potential, xv, dt, choice='King62', alpha=alpha)
            a = s.alphah
            c, Delta = ev.Dekel(m, mv0, rmax0, vmax0, aDekel0, z=0.)
        else:
            m = cfg.Mres
            lt = cfg.Rres
            c = s.ch
            a = s.alphah
            Delta = 200.
        s = Dekel(m, c, a, Delta=Delta, z=0.)

        t_arr[i] = t
        r_arr[i] = r
        m_arr[i] = m
        vmax_arr[i] = s.Vmax
        rmax_arr[i] = s.rmax
        lt_arr[i] = lt

        slot = np.searchsorted(track_steps, i)
        if slot < n_snapshots and track_steps[slot] == i:
            rho_snaps[slot] = s.rho(r_grid)
            M_snaps[slot] = s.M(r_grid)

        tprevious = t

    return EvolutionResult(
        t=t_arr, r=r_arr, m=m_arr, vmax=vmax_arr, rmax=rmax_arr, lt=lt_arr,
        r_grid=r_grid, rho_snapshots=rho_snaps, M_snapshots=M_snaps,
        snapshot_steps=track_steps, rmax0=rmax0, vmax0=vmax0, label=label,
    )


def evolve_satgen_green(host, ma, c2a, xv0, tmax=10., Nstep=10000,
                        alpha='conc', Delta=200., z=0.,
                        n_snapshots=10, label='SatGen (Green / DASH track)'):
    """Green+21 DASH transfer-function evolution (Baseline B).

    alpha='conc' uses concentration-dependent stripping efficiency (ev.alpha_from_c2).
    """
    potential = host
    timesteps = np.linspace(0., tmax, Nstep + 1)[1:]

    s = Green(ma, c2a, Delta=Delta, z=z)
    vmax0, rmax0 = _vmax_rmax(s)

    r_grid = np.logspace(np.log10(cfg.Rres), np.log10(s.rh), 100)
    track_steps = np.round(np.linspace(0, Nstep - 1, n_snapshots)).astype(int)

    t_arr = np.zeros(Nstep)
    r_arr = np.zeros(Nstep)
    m_arr = np.zeros(Nstep)
    vmax_arr = np.zeros(Nstep)
    rmax_arr = np.zeros(Nstep)
    lt_arr = np.zeros(Nstep)
    rho_snaps = np.zeros((n_snapshots, len(r_grid)))
    M_snaps = np.zeros((n_snapshots, len(r_grid)))

    o = orbit(xv0)
    r = np.sqrt(xv0[0]**2 + xv0[2]**2)
    m = ma
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

        if m > cfg.Mres:
            if alpha == 'conc':
                al = ev.alpha_from_c2(host.ch, s.ch)
            else:
                al = float(alpha)
            m, lt = ev.msub(s, potential, xv, dt, choice='King62', alpha=al)
            m = max(m, cfg.Mres)
            s.update_mass(m)
        else:
            m = cfg.Mres
            lt = cfg.Rres

        vm, rm = _vmax_rmax(s)

        t_arr[i] = t
        r_arr[i] = r
        m_arr[i] = m
        vmax_arr[i] = vm
        rmax_arr[i] = rm
        lt_arr[i] = lt

        slot = np.searchsorted(track_steps, i)
        if slot < n_snapshots and track_steps[slot] == i:
            rho_snaps[slot] = s.rho(r_grid)
            M_snaps[slot] = s.M(r_grid)

        tprevious = t

    return EvolutionResult(
        t=t_arr, r=r_arr, m=m_arr, vmax=vmax_arr, rmax=rmax_arr, lt=lt_arr,
        r_grid=r_grid, rho_snapshots=rho_snaps, M_snapshots=M_snaps,
        snapshot_steps=track_steps, rmax0=rmax0, vmax0=vmax0, label=label,
    )


def evolve_heating(host, numProfile0, xv0, tmax=10., Nstep=10000,
                   epsh=3., gamma=2.5, alpha=1.,
                   n_snapshots=10, label='Du+24 heating'):
    """Du+24 monotonic shell expansion + King62 stripping."""
    potential = host
    timesteps = np.linspace(0., tmax, Nstep + 1)[1:]

    numProfile = numProfile0
    rmax0 = numProfile.rmax
    vmax0 = numProfile.Vmax
    mv0 = numProfile.Mh

    r_grid = np.logspace(np.log10(cfg.Rres), np.log10(numProfile.rh), 100)
    track_steps = np.round(np.linspace(0, Nstep - 1, n_snapshots)).astype(int)

    t_arr = np.zeros(Nstep)
    r_arr = np.zeros(Nstep)
    m_arr = np.zeros(Nstep)
    vmax_arr = np.zeros(Nstep)
    rmax_arr = np.zeros(Nstep)
    lt_arr = np.zeros(Nstep)
    rho_snaps = np.zeros((n_snapshots, len(r_grid)))
    M_snaps = np.zeros((n_snapshots, len(r_grid)))

    o = orbit(xv0)
    r = np.sqrt(xv0[0]**2 + xv0[2]**2)
    m = mv0
    lt = cfg.Rres
    tprevious = 0.
    tt_int = np.zeros((3, 3))  # running tidal-tensor time integral [kpc/Gyr]^2

    for i, t in enumerate(timesteps):
        dt = t - tprevious
        if r > cfg.Rres:
            o.integrate(t, potential, m)
            xv = o.xv
        else:
            tprevious = t
            continue
        r = np.sqrt(xv[0]**2 + xv[2]**2)
        V = np.sqrt(xv[3]**2 + xv[4]**2 + xv[5]**2)
        x = xv[0] * np.cos(xv[1])
        y = xv[0] * np.sin(xv[1])
        z_c = xv[2]
        t_orb = tdyn(potential, r)

        tt_cur = tidalTensor(potential, [x, y, z_c])
        tt_int += (tt_cur - tt_int / t_orb) * dt  # Benson+Du22 eq.16
        tidalHR = (epsh / 3  # orbit-averaged heating rate; Benson+Du22 eq.18
                   * (1 + (2 * np.pi * r / V / t_orb)**2)**(-gamma)
                   * np.sum(tt_cur * tt_int))

        if m > cfg.Mres:
            def eps_r(r_): return dt * tidalHR * r_**2
            newProfile = heat_profile(numProfile, eps_r)
            lt = ev.ltidal(newProfile, potential, xv, 'King62')
            if lt < newProfile.rh:
                dm = alpha * (newProfile.Mh - newProfile.M(lt)) * dt / tdyn(potential, r)
                dm = max(dm, 0.)
                m_new = max(newProfile.Mh - dm, cfg.Mres)
                if m_new > cfg.Mres:
                    rmaxNew = bisect(lambda x_: newProfile.M(x_) - m_new,
                                     cfg.Rres, newProfile.rh)
                    rvals = np.logspace(np.log10(cfg.Rres), np.log10(rmaxNew), 100)
                    numProfile = NumericProfile(rvals, newProfile.M(rvals))
                m = m_new
            else:
                numProfile = newProfile
                m = numProfile.Mh
        else:
            m = cfg.Mres
            lt = cfg.Rres

        t_arr[i] = t
        r_arr[i] = r
        m_arr[i] = m
        vmax_arr[i] = numProfile.Vmax
        rmax_arr[i] = numProfile.rmax
        lt_arr[i] = lt

        slot = np.searchsorted(track_steps, i)
        if slot < n_snapshots and track_steps[slot] == i:
            rho_snaps[slot] = numProfile.rho(r_grid)
            M_snaps[slot] = numProfile.M(r_grid)

        tprevious = t

    return EvolutionResult(
        t=t_arr, r=r_arr, m=m_arr, vmax=vmax_arr, rmax=rmax_arr, lt=lt_arr,
        r_grid=r_grid, rho_snapshots=rho_snaps, M_snapshots=M_snaps,
        snapshot_steps=track_steps, rmax0=rmax0, vmax0=vmax0, label=label,
    )


#---plotting

def plot_tidal_track(results, ax=None, title='Tidal Tracks'):
    if not isinstance(results, list):
        results = [results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    for res in results:
        mask = (res.rmax > 0) & (res.vmax > 0)
        ax.plot(res.rmax[mask] / res.rmax0, res.vmax[mask] / res.vmax0,
                '.', label=res.label, ms=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$r_{\rm max}/r_{\rm max,0}$')
    ax.set_ylabel(r'$V_{\rm max}/V_{\rm max,0}$')
    ax.set_title(title)
    ax.legend()
    return ax


def plot_mass_loss(results, ax=None, title='Mass Loss History'):
    if not isinstance(results, list):
        results = [results]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for res in results:
        mask = res.m > 0
        ax.plot(res.t[mask], res.m[mask], '.', label=res.label, ms=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('time [Gyr]')
    ax.set_ylabel(r'$m$ [$M_\odot$]')
    ax.set_title(title)
    ax.legend(fontsize=9)
    return ax


def profile_ylims(*results):
    """Shared y-axis limits across results, for use with plot_profile_snapshots."""
    rho_all, M_all, Vc_all = [], [], []
    for res in results:
        r_grid = res.r_grid
        for snap_rho, snap_M in zip(res.rho_snapshots, res.M_snapshots):
            if np.all(snap_M == 0):
                continue
            Vc = np.sqrt(cfg.G * snap_M / r_grid)
            pos = snap_rho[snap_rho > 0]
            if len(pos):
                rho_all.append(pos.min()); rho_all.append(pos.max())
            pos = snap_M[snap_M > 0]
            if len(pos):
                M_all.append(pos.min()); M_all.append(pos.max())
            pos = Vc[Vc > 0]
            if len(pos):
                Vc_all.append(pos.min()); Vc_all.append(pos.max())

    def _lims(vals):
        return (min(vals), max(vals)) if vals else (1e-10, 1e10)

    return [_lims(rho_all), _lims(M_all), _lims(Vc_all)]


def plot_profile_snapshots(result, axes=None, title_prefix='', ylims=None):
    """Plot rho(r), M(<r), Vc(r) at snapshot steps. ylims from profile_ylims()."""
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 4))
    r_grid = result.r_grid
    n_snap = result.rho_snapshots.shape[0]
    cmap = plt.get_cmap('viridis')
    for i in range(n_snap):
        if np.all(result.M_snapshots[i] == 0):
            continue
        c = cmap(i / max(n_snap - 1, 1))
        Vc = np.sqrt(cfg.G * result.M_snapshots[i] / r_grid)
        lbl = f'step {result.snapshot_steps[i]}'
        axes[0].plot(r_grid, result.rho_snapshots[i], color=c, label=lbl)
        axes[1].plot(r_grid, result.M_snapshots[i], color=c, label=lbl)
        axes[2].plot(r_grid, Vc, color=c, label=lbl)
    labels = [r'$\rho(r)$ [$M_\odot/{\rm kpc}^3$]',
              r'$M(r)$ [$M_\odot$]',
              r'$V_{\rm c}(r)$ [kpc/Gyr]']
    titles = ['Density', 'Enclosed Mass', 'Circular Velocity']
    panel_ylims = ylims if ylims is not None else [None, None, None]
    for ax, ylab, ttl, ylim in zip(axes, labels, titles, panel_ylims):
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$r$ [kpc]')
        ax.set_ylabel(ylab)
        ax.set_title(f'{title_prefix}{ttl}')
        if ylim is not None:
            ax.set_ylim(*ylim)
    return axes


def plot_orbit(results, ax=None, title='Orbital Radius'):
    if not isinstance(results, list):
        results = [results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for res in results:
        mask = res.r > 0
        ax.plot(res.t[mask], res.r[mask], label=res.label)
    ax.set_xlabel(r'$t$ [Gyr]')
    ax.set_ylabel(r'$r$ [kpc]')
    ax.set_title(title)
    ax.legend()
    return ax
