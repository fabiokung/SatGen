# shared evolution routines and plots for tidal stripping notebooks

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    # per-snapshot radial grid that follows the bound profile's extent —
    # a fixed grid at the initial range would extrapolate later snapshots
    # to regions with no bound mass and produce flat/zero artefacts in plots
    r_grid: np.ndarray  # (n_snapshots, len_grid)
    rho_snapshots: np.ndarray  # (n_snapshots, len_grid)
    M_snapshots: np.ndarray  # (n_snapshots, len_grid)
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


class _HeatingStepper:
    """Per-step Benson+Du22 second-order bookkeeping. Hides the peri-to-peri
    cumulant H(t), the pericentre detection, and the frozen sigma_r^2
    behind a uniform interface, so the evolve_heating loop doesn't branch
    on second_order at every per-step decision.

    For second_order=False, .step returns the plain Pullen+14 first-order
    eps(r) = dt * tidalHR * r**2 and never signals a reset. For
    second_order=True it tracks the cumulant from the previous pericentre
    and emits a per-step Benson+Du22 increment that adds up over each
    peri-to-peri segment to the per-orbit kick (see evolve_heating
    docstring for the derivation). .reset is called by the loop after
    each peri to start a fresh segment.
    """

    def __init__(self, numProfile, second_order):
        self.second_order = second_order
        if not second_order:
            return
        f2, chi_v = 0.406, -0.333  # Benson & Du 2022 eq. (4)
        self.c2 = np.sqrt(2.) * f2 * (1. + chi_v)
        self.H = 0.
        self.sqrt_H = 0.
        self.t_last_reset = 0.
        self.r_p1 = None  # previous step's r
        self.r_p2 = None  # two-step-back r
        # sigma_r^2(r) is frozen at the start of each peri-to-peri segment
        # so the per-step contributions add up exactly to the per-orbit
        # eq. (4) total. Drifting it within a segment under-counts because
        # heating raises specific energy and lowers sigma_r^2.
        self.sig2 = numProfile._sig2
        self.rh = numProfile.rh

    def step(self, dt, tidalHR, r, t_now, t_orb):
        """Returns (eps_r, should_reset). Advances internal state."""
        if not self.second_order:
            return (lambda r_: dt * tidalHR * r_**2), False

        # pericentre detector: r_p1 was a local minimum if r_p2 > r_p1 < r.
        # 4*t_dyn fallback ensures near-circular orbits still reset.
        peri = (self.r_p1 is not None and self.r_p2 is not None
                and self.r_p1 < self.r_p2 and self.r_p1 < r)
        fallback = (not peri) and (t_now - self.t_last_reset >= 4. * t_orb)
        self.r_p2 = self.r_p1
        self.r_p1 = r

        d_H = max(tidalHR * dt, 0.)
        H_new = self.H + d_H
        sqrt_H_new = np.sqrt(H_new)
        d_sqrt = sqrt_H_new - self.sqrt_H
        c2, sig2, rh = self.c2, self.sig2, self.rh

        def eps_r(r_, _h=d_H, _ds=d_sqrt, _c2=c2, _s2=sig2, _rh=rh):
            e1 = _h * r_**2
            s2 = max(float(_s2(r_)), 0.) if r_ <= _rh else 0.
            return e1 + _c2 * r_ * np.sqrt(s2) * _ds

        self.H = H_new
        self.sqrt_H = sqrt_H_new
        return eps_r, peri or fallback

    def reset(self, numProfile, t_now):
        """Reset cumulant after a peri detection. sigma_r^2 is re-frozen
        on the current (post-strip) numProfile."""
        if not self.second_order:
            return
        self.H = 0.
        self.sqrt_H = 0.
        self.t_last_reset = t_now
        self.sig2 = numProfile._sig2
        self.rh = numProfile.rh


def evolve_heating(host, numProfile0, xv0, tmax=10., Nstep=10000,
                   epsh=3., gamma=2.5, alpha=1.,
                   second_order=False,
                   n_snapshots=10, label=None):
    """Du+24 monotonic shell expansion + King62 stripping.

    second_order=True adds the Benson+Du22 second-order correction (eq. 4):
        dE = dE_1 + c2 * sqrt(dE_1 * sigma_r^2),  c2 = sqrt(2) f_2 (1+chi_v)
    with f_2=0.406, chi_v=-0.333.

    Benson+Du22 is a per-shock budget: dE_1 and sigma_r^2 are quantities accumulated
    over one full orbital encounter. The sqrt does not split linearly across
    timesteps — a naive sum_i sqrt(dE_1,i * sigma^2) scales as sqrt(N) instead
    of sqrt(1). We track a running cumulant H(t) = int_0^t H_dot(t') dt' from
    the previous pericentre and inject

        d(dE_2)(r) = c2 * r * sqrt(sigma_r^2(r)) * [sqrt(H(t_i)) - sqrt(H(t_{i-1}))]

    per step. The increments add up over a peri-to-peri segment to
    c2 * r * sqrt(sigma_r^2 * H_orbit), the per-orbit Benson+Du22 kick.
    H resets at each pericentre (with a 4*t_dyn fallback for near-circular orbits).
    """
    if label is None:
        label = 'Du+24 heating + Benson+Du22 2nd-order' if second_order else 'Du+24 heating'
    potential = host
    timesteps = np.linspace(0., tmax, Nstep + 1)[1:]

    numProfile = numProfile0
    rmax0 = numProfile.rmax
    vmax0 = numProfile.Vmax
    mv0 = numProfile.Mh

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

    o = orbit(xv0)
    r = np.sqrt(xv0[0]**2 + xv0[2]**2)
    m = mv0
    lt = cfg.Rres
    tprevious = 0.
    tt_int = np.zeros((3, 3))  # running tidal-tensor time integral [kpc/Gyr]^2

    heater = _HeatingStepper(numProfile, second_order)

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
        if dt > 0.5 * t_orb:
            raise ValueError(
                f"dt/t_dyn = {dt/t_orb:.2f} > 0.5 at r={r:.1f} kpc, t={t:.2f} Gyr; increase Nstep"
            )

        tt_cur = tidalTensor(potential, [x, y, z_c])
        tt_int += (tt_cur - tt_int / t_orb) * dt  # Benson+Du22 eq.16
        # adiabatic correction (Pullen+14 / Gnedin+99, Benson+Du22 eq. 3):
        # omega_p at the subhalo half-mass radius, T_shock = r/V (the
        # instantaneous orbital timescale at the current position — small
        # at peri so the shock heats efficiently, large at apo so the
        # adiabatic factor suppresses heating)
        T_shock = r / V if V > 0. else 1e10
        adiabatic = (1. + (numProfile.omega_p * T_shock)**2)**(-gamma)
        tidalHR = (epsh/3) * adiabatic * np.sum(tt_cur * tt_int)

        eps_r, should_reset = heater.step(dt, tidalHR, r, t, t_orb)

        if m > cfg.Mres:
            newProfile = heat_profile(numProfile, eps_r)
            lt = ev.ltidal(newProfile, potential, xv, 'King62')
            if lt < newProfile.rh:
                dm = alpha * (newProfile.Mh - newProfile.M(lt)) * dt / t_orb
                dm = max(dm, 0.)
                m_new = max(newProfile.Mh - dm, cfg.Mres)
                if m_new > cfg.Mres:
                    # clamp m_new to the spline value at rh to keep bisect bracket valid
                    m_new = min(m_new, newProfile.M(newProfile.rh))
                if m_new > cfg.Mres:
                    rmaxNew = bisect(lambda x_: newProfile.M(x_) - m_new,
                                     newProfile.ri[0], newProfile.rh)
                    # Rebuild on newProfile's own knots truncated to
                    # <= rmaxNew, appending rmaxNew at m_new as the new
                    # outermost knot. A logspace(r_lo, rmaxNew, 100) grid
                    # would oversample the post-bulk-shell region (where M
                    # is essentially flat near Mh) and the FD-derived rho
                    # on those wasted points collapses to ~zero in the
                    # outer plot region.
                    mask = newProfile.ri < rmaxNew
                    rvals = np.append(newProfile.ri[mask], rmaxNew)
                    Mr = np.append(newProfile.Mr[mask], m_new)
                    numProfile = NumericProfile(rvals, Mr)
                m = m_new
            else:
                numProfile = newProfile
                m = numProfile.Mh
        else:
            m = cfg.Mres
            lt = cfg.Rres

        if should_reset:
            heater.reset(numProfile, t)

        t_arr[i] = t
        r_arr[i] = r
        m_arr[i] = m
        vmax_arr[i] = numProfile.Vmax
        rmax_arr[i] = numProfile.rmax
        lt_arr[i] = lt

        slot = np.searchsorted(track_steps, i)
        if slot < n_snapshots and track_steps[slot] == i:
            rg = np.logspace(np.log10(numProfile.ri[0]),
                             np.log10(numProfile.rh), snap_npts)
            r_grids[slot] = rg
            rho_snaps[slot] = numProfile.rho(rg)
            # cubic-spline M(r) can overshoot below Mr[0] just inside the
            # innermost knot (~1% wobble at the smallest few r). Enforce
            # monotonicity for plotting; the live spline used by ltidal
            # and Vcirc is unaffected.
            M_snaps[slot] = np.maximum.accumulate(numProfile.M(rg))

        tprevious = t

    return EvolutionResult(
        t=t_arr, r=r_arr, m=m_arr, vmax=vmax_arr, rmax=rmax_arr, lt=lt_arr,
        r_grid=r_grids, rho_snapshots=rho_snaps, M_snapshots=M_snaps,
        snapshot_steps=track_steps, rmax0=rmax0, vmax0=vmax0, label=label,
    )


#---plotting

def _style_for(i, styles):
    """Per-result kwargs override; falls back to scatter-dot default when None."""
    if styles is not None and i < len(styles) and styles[i] is not None:
        return dict(styles[i])
    return dict(marker='.', linestyle='None', ms=2)


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
        mask = np.isfinite(res.m)  # drop NaN-initialised skipped-iteration slots
        ax.plot(res.t[mask], res.m[mask],
                label=res.label, **_style_for(i, styles))
    ax.set_xscale('log')
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


def plot_orbit(results, ax=None, title='Orbital Radius', styles=None, legend=True):
    if not isinstance(results, list):
        results = [results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for i, res in enumerate(results):
        mask = np.isfinite(res.r)  # drop NaN-initialised skipped-iteration slots
        kw = dict(styles[i]) if (styles is not None and i < len(styles)
                                 and styles[i] is not None) else {}
        ax.plot(res.t[mask], res.r[mask], label=res.label, **kw)
    ax.set_xlabel(r'$t$ [Gyr]')
    ax.set_ylabel(r'$r$ [kpc]')
    ax.set_title(title)
    if legend:
        ax.legend()
    return ax
