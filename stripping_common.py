# shared evolution routines and plots for tidal stripping notebooks

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq

import config as cfg
import cosmo as co
import evolve as ev
from orbit import orbit
from profiles import NFW, Dekel, Green, Vcirc, tdyn
from subhalo_functions import (NumericProfile, heat_profile, tidalTensor,
                               truncate_kazantzidis, truncate_powerlaw)


class OverstripError(ValueError):
    """King62 step is too coarse: alpha*dt/T_strip exceeds STRIP_CFL_MAX, so
    the bound-mass excess would underflow to a numerical hard cut. An
    infeasible (params, dt) point a calibrator can catch and score as -inf,
    distinct from a bare ValueError signalling a real bug. Subclasses
    ValueError so existing `except ValueError` / pytest.raises(ValueError)
    paths still match."""


# Ceiling on the per-step stripping number cfl = alpha*dt/T_strip. The name is by
# analogy to the Courant-Friedrichs-Lewy condition (a step-size / timescale
# ratio); the limit here is floating-point underflow, not integration stability.
#
# Each King62 step relaxes the unbound-mass excess (m - M(<lt)) toward M(<lt) by a
# factor exp(-cfl). At cfl = 10 that factor is exp(-10) ~ 5e-5: a single step
# already removes ~5 orders of magnitude of the excess. Much beyond that the
# excess underflows to numerical zero -- a spurious instantaneous hard cut instead
# of resolved gradual stripping. So when a step would exceed STRIP_CFL_MAX,
# evolve_heating raises OverstripError (dt too coarse / alpha too large). Normal
# steps sit at cfl ~ 1, even at a deep pericenter, well below the ceiling.
STRIP_CFL_MAX = 10.0


# Stripping bookkeeping: Zentner+05 / Du+24 eq. 35 with T_dyn = t_orb
# (the orbital dynamical time at the satellite). Each step removes
#     dm = alpha * (M_heated.Mh - M_heated(<lt)) * dt / t_orb
# from the heated profile, then we hard-truncate at rmaxNew = M^-1(m_new):
# rebuild the grid on newProfile's own knots <= rmaxNew, appending the
# exact (rmaxNew, m_new) as the new outermost point. The truncation
# couples the bound mass to the spatial profile (m = numProfile.Mh by
# construction).


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
    # per-step shell-crossing clamp activity, aligned to t/m/lt (NaN on skipped
    # steps). clamp is the shells clamped that step; clamp_worst the largest
    # overshoot the clamp removed (%). None for non-heating evolvers. Bin by
    # m (bound fraction) or t for clamp-vs-stage analysis.
    clamp: Optional[np.ndarray] = None
    clamp_worst: Optional[np.ndarray] = None


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
DU24_ETA = {'1/5': 0.404, '1/20': 0.131}            # circularity -> R_p/R_a

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


def du24_nfw_setup(nr=200):
    """NFW host + subhalo on the Du+24 II.A ICs (virial overdensity DU24_DELTA,
    not Delta=200 -- matches Du+24's rho0/rs/rvir exactly).

    Returns (host, sat, rvals, M_sub). sat is the analytic NFW (callers that
    need r_vir or the virial-edge slope use it); the stripped evolution starts
    from NumericProfile(rvals, M_sub).
    """
    host = NFW(DU24_MV_HOST, DU24_C_HOST, Delta=DU24_DELTA)
    sat = NFW(DU24_MV_SUB, DU24_C_SUB, Delta=DU24_DELTA)
    rvals = np.logspace(np.log10(cfg.Rres), np.log10(sat.rh), nr)
    return host, sat, rvals, sat.M(rvals)


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
    """Dekel/P10 tidal-track evolution (Baseline A)."""
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


def evolve_satgen_green(host, ma, c2a, xv0, tmax=10., Nstep=10000,
                        alpha='conc', Delta=200., z=0.,
                        n_snapshots=10, label='SatGen (Green / DASH track)'):
    """Green+21 DASH transfer-function evolution (Baseline B).

    alpha='conc' uses concentration-dependent stripping efficiency (ev.alpha_from_c2).
    """
    assert cfg.Mres is not None, "cfg.Mres must be set before calling evolve_satgen_green"
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

        if m <= cfg.Mres:
            m, lt = cfg.Mres, cfg.Rres
        else:
            al = ev.alpha_from_c2(host.ch, s.ch) if alpha == 'conc' else float(alpha)
            m, lt = ev.msub(s, potential, xv, dt, choice='King62', alpha=al)
            m = max(m, cfg.Mres)
            s.update_mass(m)

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

    def __init__(self, numProfile, second_order, f2=0.406, chi_v=-0.333):
        self.second_order = second_order
        if not second_order:
            return
        # Benson+Du22 eq. (4) calibrates f_2=0.406 against an SIS host;
        # Du+24 Table IV refits f_2=0.547 for an NFW host. chi_v fixed at
        # -0.333 in both (Du+24 absorbs its uncertainty into f_2).
        self.c2 = np.sqrt(2.) * f2 * (1. + chi_v)
        self.H = 0.
        self.sqrt_H = 0.
        self.t_last_reset = 0.
        self.r_p1 = None  # previous step's r
        self.r_p2 = None  # two-step-back r
        # sigma_r^2(r) is refrozen at each peri-to-peri segment so per-step
        # contributions sum to the per-orbit eq. (4) total. Drifting it
        # within a segment under-counts because heating raises specific
        # energy and lowers sigma_r^2.
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
            # vectorized over r_ (heat_profile passes the full shell grid):
            # sigma_r^2 is only defined within the half-mass radius, zero
            # beyond. np.where masks the out-of-range spline values (the
            # interpolator's clamp/NaN tail) before the sqrt.
            r_ = np.asarray(r_, dtype=float)
            e1 = _h * r_**2
            s2 = np.where(r_ <= _rh, np.maximum(_s2(r_), 0.), 0.)
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


def _truncate_hard(newProfile, m_new):
    """Hard cut: rebuild on a fresh log grid truncated at rmaxNew = M^-1(m_new).

    A fresh grid (rather than inheriting the parent's) avoids shedding knots as
    rmaxNew shrinks, which clumps the (rmax, Vmax) track at deep stripping. The
    inner bound max(Rres, ri[0]) keeps samples off the PCHIP boundary clamp
    below ri[0] (a constant-M plateau that np.gradient turns into alternating
    zero / spurious rho inside NumericProfile).
    """
    m_new = min(m_new, float(newProfile.M(newProfile.rh)))  # keep bisect bracket valid
    rmaxNew = brentq(lambda r_: float(newProfile.M(r_)) - m_new,
                     newProfile.ri[0], newProfile.rh, xtol=1e-8)
    rvals_new = np.logspace(np.log10(max(cfg.Rres, float(newProfile.ri[0]))),
                            np.log10(rmaxNew), len(newProfile.ri))
    Mr_new = newProfile.M(rvals_new)
    Mr_new[-1] = m_new  # heat_profile spline can land M(rmaxNew) slightly off m_new
    return NumericProfile(rvals_new, Mr_new)


def _strip_and_truncate(profile, newProfile, potential, xv, t_orb, dt, alpha,
                        t_dyn_mode, truncation, t, tail_n=5., tail_xi=0.,
                        lt_choice='King62'):
    """One King62 strip + retruncate step. Returns (profile, m, lt).

    Zentner+05 / Du+24 eq. 35 relaxes the bound mass toward M(<lt) on T_strip;
    `profile` is the current bound profile and `newProfile` its heated form for
    this step. lt beyond the bound profile leaves it intact; a bound mass that
    falls to the floor keeps the last bound `profile`.
    """
    lt = float(ev.ltidal(newProfile, potential, xv, lt_choice))  # type: ignore[arg-type]
    if lt >= newProfile.rh:
        return newProfile, newProfile.Mh, lt
    # PCHIP eval at lt < rh can return M(lt) > Mh at floating-point precision
    # (~1e-16 relative) due to roundoff, even though the spline is monotone on
    # monotone input. Clamp to M(<r) <= Mh so the relaxation excess stays >= 0.
    M_at_lt = min(float(newProfile.M(lt)), float(newProfile.Mh))
    if t_dyn_mode == 'sub_lt':
        T_strip = (np.pi / 2.) * (
            np.sqrt(lt**3 / (cfg.G * M_at_lt)) if M_at_lt > 0.
            else newProfile.rmax / newProfile.Vmax
        )
    else:
        T_strip = t_orb
    # Over a step lt (hence M(<lt), T_strip) is fixed, so the excess m - M(<lt)
    # relaxes as exp(-alpha dt/T_strip). The closed form asymptotes to M(<lt)
    # from above and never crosses it, so m_new >= M(<lt) for any dt.
    cfl = alpha * dt / T_strip
    if cfl > STRIP_CFL_MAX:
        raise OverstripError(
            f"Stripping step alpha*dt/T_strip = {cfl:.1f} > {STRIP_CFL_MAX} "
            f"at lt={lt:.2f} kpc, t={t:.2f} Gyr; the bound-mass excess would "
            f"underflow. Increase Nstep to reduce dt or decrease alpha"
        )
    m_new = max(M_at_lt + (newProfile.Mh - M_at_lt) * np.exp(-cfl), cfg.Mres)
    if m_new <= cfg.Mres:
        return profile, m_new, lt
    if truncation == 'hard':
        return _truncate_hard(newProfile, m_new), m_new, lt
    if truncation == 'powerlaw':
        # slope-deficit tail: C1 join at r_join = lt * 10^xi, asymptoting to
        # rho ~ r^-n (see truncate_powerlaw); beta solved so the tail carries
        # the King62 budget m_new - M(<r_join). xi=0 joins at the tidal
        # radius; xi and n are calibration knobs.
        r_join = lt * 10.**tail_xi
        return truncate_powerlaw(newProfile, r_join, n=tail_n,
                                 m_total=m_new), m_new, lt
    # Kazantzidis+06 exponential tail at lt: keep M(<lt) and attach a tail
    # carrying the King62 budget m_new - M(<lt) so the loosely-bound envelope is
    # shed smoothly rather than clipped.
    return truncate_kazantzidis(newProfile, lt, m_total=m_new), m_new, lt


def evolve_heating(host, numProfile0, xv0, tmax=10., Nstep=10000,
                   epsh=3., gamma=2.5, alpha=1., beta_h=1.,
                   second_order=False, f2=0.406, chi_v=-0.333,
                   t_dyn_mode='sub_lt', truncation='hard',
                   tail_n=5., tail_xi=0., lt_choice='King62',
                   n_snapshots=10, label=None, early_terminate=False,
                   dynamical_friction=True):
    # t_dyn_mode controls only the timescale T in the King62 stripping rate
    # (Du+24 eq. 35):
    #   'host':   T = T_dyn,host(r_sub) = tdyn(host, r_sub).
    #   'sub_lt': T = T_dyn,sub(lt) = (pi/2) sqrt(lt^3 / G M_sub(<lt)),
    #             evaluated at the current-step lt. When M(<lt) drops to
    #             zero (lt below grid resolution), the fallback is
    #             T_dyn,sub(rmax) = (pi/2) rmax/Vmax -- still subhalo-
    #             internal, never host-side.
    # The tidal-tensor cumulant decay (Du+24 eq. 39), the adiabatic T_shock
    # (= r/V), the per-orbit cumulant reset fallback, and the CFL timestep
    # guard all use host-side t_orb regardless. Du+24 integrate the tensor
    # cumulant on the host orbital time and strip on the subhalo dynamical
    # time (eqs. 35, 38-39); 'sub_lt' reproduces that split.
    # The defaults are the SIS-host set from stripping_sis.ipynb (epsh=3,
    # gamma=2.5, alpha=1, beta_h=1, f2=0.406, chi_v=-0.333, t_dyn_mode='sub_lt').
    # Of these only f2/chi_v come from the Benson+Du22 second-order fit;
    # Benson+Du22 do not calibrate the King62 mass-loss (alpha, T) -- that is
    # Yang+20. The calibrated NFW-host set (Du+24 Table IV, gamma=1) lives in
    # model_params.DU24_TABLE_IV; it also strips on t_dyn_mode='sub_lt'.
    """Du+24 monotonic shell expansion + King62 stripping.

    second_order=True adds the Benson+Du22 second-order correction (eq. 4):
        dE = dE_1 + c2 * sqrt(dE_1 * sigma_r^2),  c2 = sqrt(2) f_2 (1+chi_v)
    Defaults f_2=0.406, chi_v=-0.333 are the Benson+Du22 SIS-host fit; the
    calibrated NFW-host set is model_params.DU24_TABLE_IV.

    beta_h sets the cumulant decay rate in eq. 39 of Du+24:
        dG_ab/dt = g_ab - beta_h * G_ab / T_orbit
    The default beta_h=1 matches Benson+Du22 eq. 16.

    truncation selects how the stripped profile is rebuilt each step:
        'hard'        -- truncate at rmaxNew = M^-1(m_new) (default).
        'kazantzidis' -- keep M(<lt), attach a Kazantzidis+06 exponential
                         tail carrying the King62 budget m_new - M(<lt).
        'powerlaw'    -- keep M(<r_join), attach a slope-deficit power-law
                         tail (truncate_powerlaw) asymptoting to
                         rho ~ r^-tail_n, C1 at r_join = lt * 10^tail_xi,
                         carrying the King62 budget. tail_xi=0 joins at the
                         tidal radius; tail_n and tail_xi are calibration
                         knobs (stripped N-body envelopes are r^-5..-6,
                         Springel+08 / Green & van den Bosch 2019).

    lt_choice selects the tidal-radius equation passed to ev.ltidal:
        'King62'   -- keeps the centrifugal term Omega^2 r^3 / G M (default).
        'Tormen98' -- drops it -> larger lt, weaker stripping. This is the
                      form Du+24 use (their yc=0); King62 is yc=1.

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
        label = '1st+2nd order heating' if second_order else '1st order heating'
    assert cfg.Mres is not None, "cfg.Mres must be set before calling evolve_heating"
    assert truncation in ('hard', 'kazantzidis', 'powerlaw'), \
        f"truncation must be 'hard', 'kazantzidis' or 'powerlaw', got {truncation!r}"
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

    heater = _HeatingStepper(numProfile, second_order, f2=f2, chi_v=chi_v)
    clamp_arr = np.full(Nstep, np.nan)
    clamp_worst_arr = np.full(Nstep, np.nan)

    for i, t in enumerate(timesteps):
        dt = t - tprevious
        if r <= cfg.Rres:
            tprevious = t
            continue
        if early_terminate and m <= cfg.Mres:
            # subhalo dissolved -- King62 rate keeps m clamped at Mres, so
            # later steps just no-op heating + reuse the frozen profile.
            # Record the terminal state once and exit; later slots stay NaN.
            t_arr[i] = t
            r_arr[i] = r
            m_arr[i] = cfg.Mres
            vmax_arr[i] = numProfile.Vmax
            rmax_arr[i] = numProfile.rmax
            lt_arr[i] = cfg.Rres
            clamp_arr[i] = 0.
            clamp_worst_arr[i] = 0.
            break
        o.integrate(t, potential, m if dynamical_friction else None)
        xv = o.xv
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
        # Du+24 eq. 39: dG_ab/dt = g_ab - beta_h * G_ab / T_orb, integrated
        # on the host orbital time. beta_h=1 recovers Benson+Du22 eq. 16;
        # Du+24 Table IV calibrates beta_h=0.278 for an NFW subhalo on an
        # NFW host.
        tt_int += (tt_cur - beta_h * tt_int / t_orb) * dt
        # adiabatic correction (Pullen+14 / Gnedin+99, Benson+Du22 eq. 3):
        # omega_p at the subhalo half-mass radius, T_shock = r/V (the
        # instantaneous orbital timescale at the current position — small
        # at peri so the shock heats efficiently, large at apo so the
        # adiabatic factor suppresses heating)
        T_shock = r / V if V > 0. else 1e10
        adiabatic = (1. + (numProfile.omega_p * T_shock)**2)**(-gamma)
        tidalHR = (epsh/3) * adiabatic * np.sum(tt_cur * tt_int)

        eps_r, should_reset = heater.step(dt, tidalHR, r, t, t_orb)

        if m <= cfg.Mres:
            m, lt = cfg.Mres, cfg.Rres
            clamp_arr[i] = 0.
            clamp_worst_arr[i] = 0.
        else:
            tally = {}
            newProfile = heat_profile(numProfile, eps_r, tally=tally)
            clamp_arr[i] = tally['shells']
            clamp_worst_arr[i] = tally['worst_pct']
            numProfile, m, lt = _strip_and_truncate(
                numProfile, newProfile, potential, xv, t_orb, dt, alpha,
                t_dyn_mode, truncation, t, tail_n=tail_n, tail_xi=tail_xi,
                lt_choice=lt_choice)

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
        clamp=clamp_arr, clamp_worst=clamp_worst_arr,
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
