# heating + stripping forward models and the results they produce

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

import config as cfg
import evolve as ev
from orbit import orbit
from profiles import Vcirc, fDF, tdyn
from subhalo_functions import (NumericProfile, HeatingUnbindsError, _expand_shells,
                               _log_pchip, heat_profile, tidalTensor,
                               truncate_kazantzidis, truncate_powerlaw)


class OverstripError(ValueError):
    """King62 step is too coarse: alpha*dt/T_strip exceeds STRIP_CFL_MAX, so the
    bound-mass excess would underflow to a numerical hard cut. Marks an infeasible
    (params, dt) point, distinct from a bare ValueError signalling a real bug.
    Subclasses ValueError so `except ValueError` paths still match."""


class PericentreUnresolvedError(OverstripError):
    """evolve_heating_revirial: the King62 strip still saturates (realized strip
    number > strip_number_max) at the finest allowed step dt_min, so the pericentre
    is out of the resolvable regime. Subclasses OverstripError, separating an
    unresolvable orbit from the coarse-dt case.

    dens_hist/dens_hist_k/t_last/r_last/m_last carry the run up to the raise;
    r_lo/r_hi the min/max orbital radius over the circuit breaker's recent
    not-stripping streak. Default None/0 when unset."""
    dens_hist: Optional[np.ndarray] = None
    dens_hist_k: Optional[np.ndarray] = None
    t_last: float = 0.
    r_last: float = 0.
    r_lo: float = 0.
    r_hi: float = 0.


# Ceiling on the per-step stripping number cfl = alpha*dt/T_strip. Named by analogy
# to the Courant-Friedrichs-Lewy condition; the limit here is floating-point
# underflow, not integration stability. Each King62 step relaxes the unbound-mass
# excess (m - M(<lt)) by exp(-cfl), so at cfl = 10 a single step already removes ~5
# orders of magnitude of it; much beyond that the excess underflows to zero, a
# spurious hard cut in place of resolved stripping. Exceeding it raises
# OverstripError. Normal steps sit at cfl ~ 1 even at a deep pericenter.
STRIP_CFL_MAX = 10.0


@dataclass
class EvolutionResult:
    t: np.ndarray  # time [Gyr]
    r: np.ndarray  # orbital radius [kpc]
    m: np.ndarray  # bound mass [Msun]
    vmax: np.ndarray  # Vmax [kpc/Gyr]
    rmax: np.ndarray  # rmax [kpc]
    lt: np.ndarray  # tidal radius [kpc]
    # grid follows each snapshot's bound extent; a grid fixed at the initial range
    # extrapolates later snapshots into regions with no bound mass
    r_grid: np.ndarray  # (n_snapshots, len_grid)
    rho_snapshots: np.ndarray  # (n_snapshots, len_grid)
    M_snapshots: np.ndarray  # (n_snapshots, len_grid)
    snapshot_steps: np.ndarray  # step indices
    rmax0: float = 0.
    vmax0: float = 0.
    label: str = ''
    # per-step cylindrical [R, phi, z, VR, Vphi, Vz] (kpc, rad, kpc/Gyr), (nsteps, 6)
    xv: Optional[np.ndarray] = None
    # True where the step re-virialized (once per apocenter): m/vmax/rmax there are
    # the post-reshape equilibrium, carried stale on the steps in between
    apo: Optional[np.ndarray] = None
    # shell-crossing clamp activity in the re-virialization reshape: shells clamped,
    # shells heated (the clamp-fraction denominator), largest overshoot removed [%],
    # its radius [kpc]. Revirial engine: apocenter steps only, NaN elsewhere;
    # uniform engine: every heating step.
    clamp: Optional[np.ndarray] = None
    clamp_total: Optional[np.ndarray] = None
    clamp_worst: Optional[np.ndarray] = None
    clamp_worst_r: Optional[np.ndarray] = None
    # same telemetry from the per-step _ExpandedProfile that locates l_t, kept apart
    # from the apocenter reshape above -- heating alone, no tail, discarded each
    # step. NaN on disrupted and parked steps, which build no expansion.
    expand_clamp: Optional[np.ndarray] = None
    expand_clamp_total: Optional[np.ndarray] = None
    expand_clamp_worst: Optional[np.ndarray] = None
    expand_clamp_worst_r: Optional[np.ndarray] = None
    # dt-weighted residence [Msun Gyr] on the caller's (radial, time) grid:
    # dens_hist[i, j] sums m * dt over accepted steps in time bin j at a radius in
    # bin i, so the time-averaged density is dens_hist[:, j] / (dt_j * V_shell).
    # The end radial bins are catch-alls (inside redges[0], beyond redges[-1]) so
    # the total conserves int m dt; their shell volume is fictional. A clump parked
    # at r_stop stops depositing -- that tail is the caller's to add.
    dens_hist: Optional[np.ndarray] = None
    # companions weighting each deposit by w_k(r, v), v the orbital speed at the
    # step, one per kernel in dens_kernels. (n_kernels, n_rbin, n_tbin).
    dens_hist_k: Optional[np.ndarray] = None
    # the circuit breaker stopped the run: a tight orbit whose bound mass had not
    # changed for freeze_orbits consecutive orbits. The final (t, r, m, profile) is
    # that settled state, held by the caller for the remaining time.
    frozen: bool = False
    # the freeze was the mass floor (freeze_mfrac), so the held tail is the physical
    # continuation at that bound rather than a bracketed approximation
    frozen_mfloor: bool = False
    # the freeze was the step-count backstop (freeze_nstep): the budget ran out,
    # with no evidence the orbit had stopped evolving. Steps rather than wall-clock
    # keep the freeze -- and the residence with it -- reproducible.
    frozen_budget: bool = False
    # the state the sink-time criterion saw, NaN when the run did not freeze:
    # freeze_r_lo the settled streak's minimum radius, freeze_t_sink the sink time
    # from it (+inf when the clamped Coulomb log leaves no drag), set only when that
    # criterion froze the orbit. Both predate the closing re-virialization flush.
    freeze_r_lo: float = np.nan
    freeze_t_sink: float = np.nan
    # the quiet-regime half of the same criterion at r_lo: the tidal radius there
    # and the eta * (bound extent) it is compared against
    freeze_lt_lo: float = np.nan
    freeze_eta_rout: float = np.nan


def _dens_accumulate(hist, redges, tedges, r, m, tmid, dt,
                     hist_k=None, kernels=None, v=None):
    """Add m*dt into the (radial, time) residence histogram: radial bin holding
    orbital radius r, time bin holding the step midpoint tmid. Both indices are
    clamped to the grid, so the first/last radial bins are catch-alls for residence
    inside redges[0] / beyond redges[-1] -- nothing is dropped, total residence
    equals the integral of m dt (the end bins' shell volume is fictional, so the
    density reconstruction ignores them). When kernel histograms are passed,
    hist_k[k] additionally accumulates kernels[k](r, v) * m * dt into the same bin
    (v the orbital speed at the step)."""
    i = min(max(int(np.searchsorted(redges, r)) - 1, 0), hist.shape[0] - 1)
    j = min(max(int(np.searchsorted(tedges, tmid)) - 1, 0), hist.shape[1] - 1)
    hist[i, j] += m * dt
    if hist_k is not None and kernels is not None:
        for k, ker in enumerate(kernels):
            hist_k[k, i, j] += ker(r, v) * m * dt


def _cumulant_step(G, g, beta_h, t_orb, dt):
    """One step of Du+24 eq. 39: dG_ab/dt = g_ab - beta_h G_ab / T_orb.

    Linear ODE with a stiff decay term, integrated exactly over the step with g
    and T_orb held at their step values:

        G_{n+1} = G_n e^{-b} + (T_orb g / beta_h)(1 - e^{-b}),  b = beta_h dt / T_orb

    The exact e^{-b} stays in (0, 1]; forward Euler's truncated 1 - b goes negative
    for b > 1 and rings the cumulant. beta_h -> 0 reduces to G_n + g dt. beta_h=1
    recovers Benson+Du22 eq. 16; Du+24 Table IV calibrates 0.278 for NFW on NFW.
    """
    if beta_h <= 0.:
        return G + g * dt
    b = beta_h * dt / t_orb
    # -expm1(-b) = 1 - e^{-b}, accurate as b -> 0 where the plain 1 - e^{-b}
    # cancels catastrophically; the forcing weight (t_orb/beta_h)(1-e^{-b})
    # then -> g dt continuously.
    return G * np.exp(-b) - g * (t_orb / beta_h) * np.expm1(-b)


class _HeatingStepper:
    """Per-step Benson+Du22 second-order bookkeeping. Hides the peri-to-peri
    cumulant H(t), the pericentre detection, and the frozen sigma_r^2
    behind a uniform interface, so the evolve_heating loop doesn't branch
    on second_order at every per-step decision.

    For second_order=False, .step returns the plain Pullen+14 first-order
    eps(r) = dI * r**2, dI the trapezoidal time-integral of the heating rate
    over the step, and never signals a reset. For
    second_order=True it tracks the cumulant from the previous pericentre
    and emits a per-step Benson+Du22 increment that adds up over each
    peri-to-peri segment to the per-orbit kick (see evolve_heating
    docstring for the derivation). .reset is called by the loop after
    each peri to start a fresh segment.
    """

    def __init__(self, numProfile, second_order, f2=0.406, chi_v=-0.333):
        self.second_order = second_order
        self.hr_prev = None  # previous step's heating rate, for trapezoidal
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
        # trapezoidal time-integral of the heating rate over the step (2nd order
        # vs the rectangle dt*tidalHR): dI = (tidalHR_prev + tidalHR) dt / 2.
        # The per-orbit H reset lands on a sample point and only zeroes the
        # accumulator, so reusing tidalHR at that boundary stays exact. The
        # first step has no left sample -> rectangle.
        hr_prev = tidalHR if self.hr_prev is None else self.hr_prev
        dI = 0.5 * (hr_prev + tidalHR) * dt
        self.hr_prev = tidalHR

        if not self.second_order:
            return (lambda r_, dI=dI: dI * r_**2), False

        # pericentre detector: r_p1 was a local minimum if r_p2 > r_p1 < r.
        # 4*t_dyn fallback ensures near-circular orbits still reset.
        peri = (self.r_p1 is not None and self.r_p2 is not None
                and self.r_p1 < self.r_p2 and self.r_p1 < r)
        fallback = (not peri) and (t_now - self.t_last_reset >= 4. * t_orb)
        self.r_p2 = self.r_p1
        self.r_p1 = r

        d_H = max(dI, 0.)
        H_new = self.H + d_H
        sqrt_H_new = np.sqrt(H_new)
        d_sqrt = sqrt_H_new - self.sqrt_H
        c2, sig2, rh = self.c2, self.sig2, self.rh

        def eps_r(r_, d_H=d_H, d_sqrt=d_sqrt, c2=c2, sig2=sig2, rh=rh):
            # vectorized over r_ (heat_profile passes the full shell grid):
            # sigma_r^2 is only defined within the half-mass radius, zero
            # beyond. np.where masks the out-of-range spline values (the
            # interpolator's clamp/NaN tail) before the sqrt.
            r_ = np.asarray(r_, dtype=float)
            e1 = d_H * r_**2
            s2 = np.where(r_ <= rh, np.maximum(sig2(r_), 0.), 0.)
            return e1 + c2 * r_ * np.sqrt(s2) * d_sqrt

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
                   dynamical_friction=True, r_stop=None):
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
    clamp_total_arr = np.full(Nstep, np.nan)
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
            clamp_total_arr[i] = 0.
            clamp_worst_arr[i] = 0.
            break
        o.integrate(t, potential, m if dynamical_friction else None)
        xv = o.xv
        r = np.sqrt(xv[0]**2 + xv[2]**2)
        if r_stop is not None and r < r_stop:
            # parked: the orbit has decayed to r_stop (e.g. the dynamical-friction
            # stall radius). Stop here rather than plunge toward the cusp centre,
            # where the DF force diverges. Record the terminal state and exit.
            t_arr[i] = t
            r_arr[i] = r
            m_arr[i] = m
            vmax_arr[i] = numProfile.Vmax
            rmax_arr[i] = numProfile.rmax
            lt_arr[i] = lt
            clamp_arr[i] = 0.
            clamp_total_arr[i] = 0.
            clamp_worst_arr[i] = 0.
            break
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
        # Du+24 eq. 39 cumulant, integrated exactly over the step on the host
        # orbital time (see _cumulant_step). The exact exponential decay avoids
        # forward Euler's sign flip at beta_h dt/t_orb > 1.
        tt_int = _cumulant_step(tt_int, tt_cur, beta_h, t_orb, dt)
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
            clamp_total_arr[i] = 0.
            clamp_worst_arr[i] = 0.
        else:
            tally = {}
            newProfile = heat_profile(numProfile, eps_r, tally=tally)
            clamp_arr[i] = tally['shells']
            clamp_total_arr[i] = tally['total']
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
        clamp=clamp_arr, clamp_total=clamp_total_arr, clamp_worst=clamp_worst_arr,
    )


class _ExpandedProfile:
    """Reference profile expanded by the accumulated heating scalar Q (Du+24
    eq.36 shell map, 1st + 2nd order), exposing only M(r) and rh -- all ev.ltidal
    needs from the subhalo. No PCHIP/tail rebuild; the full reshape happens once
    per apocenter.

    Uses the same _expand_shells map as the apocenter heat_profile, so the
    per-step tidal radius sees this-orbit's partial expansion consistently: outer
    shells whose injected energy exceeds their binding energy are unbound (a
    contiguous outer block), not retained, and crossings are clamped identically.
    Shell energy Delta_eps(r) = Q r^2 + c2 sqrt(Q r^2 sigma_r^2(r)).

    Raises HeatingUnbindsError when the expansion leaves <=2 bound shells: the
    accumulated heating disrupts the profile, so there is no tidal radius to find.
    """
    __slots__ = ('rh', '_r', '_M', '_mint')

    def __init__(self, r_ref, M_ref, sig2_ref, Q, c2, tally=None):
        de = Q * r_ref**2
        if c2:
            de = de + c2 * np.sqrt(np.maximum(de * sig2_ref, 0.))
        r_bound, M_bound = _expand_shells(r_ref, M_ref, de, tally=tally)
        if len(r_bound) <= 2:
            raise HeatingUnbindsError("heating unbinds the analytic profile")
        self._r, self._M = r_bound, M_bound
        self.rh = float(r_bound[-1])
        self._mint = (_log_pchip(r_bound, M_bound)
                      or PchipInterpolator(r_bound, M_bound, extrapolate=False))

    def M(self, r):
        return self._mint(r)


def _t_strip(profile, potential, xv, lt_choice, t_dyn_mode, t_orb):
    """(T_strip, lt, M(<lt)) for the King62 rate on `profile` at xv."""
    lt = float(ev.ltidal(profile, potential, xv, lt_choice))  # type: ignore[arg-type]
    if lt >= profile.rh:
        return np.inf, lt, float(profile.M(profile.rh))
    M_at_lt = float(profile.M(lt))
    if t_dyn_mode == 'sub_lt':
        T_strip = (np.pi / 2.) * (np.sqrt(lt**3 / (cfg.G * M_at_lt))
                                  if M_at_lt > 0. else np.inf)
    else:
        T_strip = t_orb
    return T_strip, lt, M_at_lt


def _reference_arrays(prof, c2):
    """Frozen-reference grid (radii, enclosed mass, sigma_r^2) that the per-step
    analytic expansion and the apocenter reshape read. sigma_r^2 is masked to zero
    beyond the half-mass edge and only built for the 2nd-order term (c2 != 0)."""
    r_ref = prof.ri
    M_ref = np.asarray(prof.M(r_ref), float)
    if c2:
        sig2 = np.where(r_ref <= prof.rh, np.maximum(prof._sig2(r_ref), 0.), 0.)
    else:
        sig2 = np.zeros_like(r_ref)
    return r_ref, M_ref, np.asarray(sig2, float)


def _revirialize(ref, Q, c2, m, lt_join, truncation, tail_n, tail_xi):
    """Expand the frozen reference by the accumulated heating (Q, 1st+2nd order) in
    one shell mapping, then reshape to bound mass m. Returns (profile, m). The join
    for the tail sits at the orbit's deepest tidal radius (lt_join)."""
    def eps_fn(rr, Q=Q, c2=c2, rh=ref.rh, sig2=ref._sig2):
        rr = np.asarray(rr, float)
        de = Q * rr**2
        if c2:
            s2 = np.where(rr <= rh, np.maximum(sig2(rr), 0.), 0.)
            de = de + c2 * np.sqrt(np.maximum(Q * rr**2 * s2, 0.))
        return de
    tally = {}
    heated = heat_profile(ref, eps_fn, tally=tally)
    m = min(m, float(heated.M(heated.rh)))
    if truncation == 'hard':
        return _truncate_hard(heated, m), m, tally
    if truncation == 'powerlaw':
        return truncate_powerlaw(heated, lt_join * 10.**tail_xi,
                                 n=tail_n, m_total=m), m, tally
    return truncate_kazantzidis(heated, lt_join, m_total=m), m, tally


def df_sink_time(potential, m, r_from, r_to, n_grid=128):
    """Adiabatic circular-orbit dynamical-friction inspiral time [Gyr] from
    r_from down to r_to, for a point satellite of mass m in `potential`, under
    the SAME drag law the orbit integrator uses (profiles.fDF with the active
    cfg.lnL_type / lnL_pref). The circular-orbit decay follows the angular
    momentum: d(r Vc)/dt = a_t r / r = a_t with a_t the tangential drag, so
    dr/dt = a_t r / dL/dr and t = int dL/dr / (|a_t| r) dr. The path integral
    handles accelerating late-stage friction correctly -- the slow outer
    segment bounds the arrival time from below. Returns np.inf when the drag
    vanishes anywhere on the path (a clamped Coulomb log: the orbit stalls
    before reaching r_to) or when r_from <= r_to."""
    if not r_from > r_to > 0.:
        return np.inf
    rg = np.geomspace(r_to, r_from, n_grid)
    vc = np.array([Vcirc(potential, r) for r in rg])
    a_t = np.array([fDF(potential, np.array([r, 0., 0., 0., v, 0.]), m)[1]
                    for r, v in zip(rg, vc)])
    if np.any(a_t >= 0.):
        return np.inf
    dLdr = np.gradient(rg * vc, rg)
    return float(np.trapezoid(dLdr / (np.abs(a_t) * rg), rg))


def _freeze_event_radius(profile, potential, m, r_lo, r_stop, lt_choice, eta,
                         n_scan=32, tally=None):
    """Largest host radius at or below r_lo where the settled clump would leave
    its quiet regime: the tidal radius (circular orbit, current
    profile) drops below eta times the bound extent -- stripping resumes -- or
    the r_stop halt would fire. None when no such radius exists (nothing below
    the current orbit can change the clump's state).

    tally, when given, records the regime test at r_lo: lt there and the eta * r_out
    it is compared against."""
    r_out = float(profile.rh)
    r_min = max(1e-3 * r_lo, r_stop if r_stop is not None else 0.)
    r_ev = r_stop
    if r_min < r_lo:
        rg = np.geomspace(r_min, r_lo, n_scan)
        for i in range(n_scan - 1, -1, -1):
            r = rg[i]
            xv = np.array([r, 0., 0., 0., float(Vcirc(potential, r)), 0.])
            lt = float(ev.ltidal(profile, potential, xv, lt_choice))
            if tally is not None and i == n_scan - 1:
                tally['lt_lo'], tally['eta_rout'] = lt, eta * r_out
            if lt < eta * r_out:
                r_ev = r if r_ev is None else max(r_ev, r)
                break
    return r_ev


def evolve_heating_revirial(host, numProfile0, xv0, tmax=10.,
                            step_frac=0.01, dt_growth_max=3.0,
                            strip_number_max=1.0, floor_orbit_frac=1e-3,
                            dt_abs_frac=1e-6, max_steps=200000,
                            revir_fallback=4.0,
                            epsh=3., gamma=2.5, alpha=1., beta_h=1.,
                            second_order=False, f2=0.406, chi_v=-0.333,
                            t_dyn_mode='sub_lt', truncation='hard',
                            tail_n=5., tail_xi=0., lt_choice='King62',
                            label=None, early_terminate=False,
                            dynamical_friction=True, r_stop=None,
                            dens_redges=None, dens_tedges=None, dens_kernels=None,
                            freeze_strip=None, freeze_orbits=20,
                            freeze_tdyn=None, freeze_nstep=None,
                            freeze_rspan=0.5, freeze_mfrac=None,
                            freeze_tsink=None, log_every=None):
    """Re-virialization-cadence forward model (Pullen+14/Du+24, impulse).

    Heating and stripping run on their own cadences:
      - per adaptive dt: accumulate the position-independent energy scalar Q on
        frozen reference radii and the Du+24 eq.39 tidal-tensor cumulant; relax the
        bound mass by King62 (eq.35) with lt from the ANALYTICALLY expanded profile
        (_ExpandedProfile), so the rate sees this orbit's partial expansion without
        a heat_profile call.
      - once per apocenter: apply the full shell expansion (heat_profile on the
        frozen reference at the accumulated Q) and reshape to the bound mass under
        `truncation` ('hard', 'kazantzidis', 'powerlaw'). Reset Q and the cumulant.

    Pullen+14 eqs. 15-18 track Q = E/x^2 at frozen shell radius x (impulse), the
    shell then expanding once from the energy at its initial radius. That response
    takes ~a dynamical time, so applying it once per orbit ties the heating count to
    the orbit rather than the timestep, and the tracks converge in dt.

    step_frac caps dt on both timescales: dt <= step_frac * min(t_orb, T_strip/alpha).
    T_strip collapses at pericentre, so the strip stays resolved there.
    revir_fallback re-virializes after that many t_orb without an apocenter, for a
    near-circular or distorted orbit with no clean r maximum. lt beyond the profile
    edge halts stripping. (Vmax, rmax) update at each apocenter, where the profile
    is an equilibrium, and carry between; t/r/m are recorded per accepted step.
    evolve_heating shares the per-step physics.

    Circuit breaker, off unless freeze_strip is set: freeze a settled, no-longer-
    stripping clump at its current state, which the caller holds for the remaining
    time. freeze_strip / freeze_orbits / freeze_rspan set the mass-streak and
    radial-span tests (_update_breaker), freeze_tdyn gates on orbit tightness, and
    freeze_tsink, e.g. {'eta': 1.2, 'safety': 2.}, on a forward-looking sink time
    (_sink_time_allows_freeze). freeze_nstep is a step-count backstop for expensive
    orbits the t_dyn test misses; it trips only together with the mass streak, and
    sets res.frozen_budget.

    freeze_mfrac (off by default, independent of the breaker) freezes at a
    re-virialization once m < freeze_mfrac * m(t=0), setting res.frozen_mfloor. A
    dense remnant keeps ~its initial mean density as it strips, so its heating dt
    never grows and the grind down to Mres costs ~1e5 more steps. With the floor on
    no run reaches cfg.Mres -- a disruption-bound subhalo ends floor-frozen -- so
    leave it off wherever disruption or mass-function statistics matter.

    log_every (seconds) prints a periodic heartbeat of the orbital state.
    """
    run = _SubhaloRun(host, numProfile0, xv0, tmax, step_frac, dt_growth_max,
                      strip_number_max, floor_orbit_frac, dt_abs_frac, max_steps,
                      revir_fallback, epsh, gamma, alpha, beta_h, second_order,
                      f2, chi_v, t_dyn_mode, truncation, tail_n, tail_xi,
                      lt_choice, label, early_terminate, dynamical_friction,
                      r_stop, dens_redges, dens_tedges, dens_kernels,
                      freeze_strip, freeze_orbits, freeze_tdyn, freeze_nstep,
                      freeze_rspan, freeze_mfrac, freeze_tsink, log_every)
    while run.t < run.tmax - run.dt_abs:
        if not run.step():
            break
    return run.finish()


class _SubhaloRun:
    """One subhalo's evolution state under the re-virialization engine. `step()`
    advances it by one accepted timestep and reads nothing about any other subhalo,
    so a population driver can interleave many on their own timesteps; `finish()`
    assembles the EvolutionResult.

    Holds everything the loop reads or writes -- the orbit, the frozen reference
    profile and its arrays, the heating ledger (Q, the tidal-tensor cumulant), the
    circuit-breaker streak, and the per-step output lists. evolve_heating_revirial
    documents the physics.
    """

    def __init__(self, host, numProfile0, xv0, tmax, step_frac, dt_growth_max,
                 strip_number_max, floor_orbit_frac, dt_abs_frac, max_steps,
                 revir_fallback, epsh, gamma, alpha, beta_h, second_order,
                 f2, chi_v, t_dyn_mode, truncation, tail_n, tail_xi,
                 lt_choice, label, early_terminate, dynamical_friction,
                 r_stop, dens_redges, dens_tedges, dens_kernels,
                 freeze_strip, freeze_orbits, freeze_tdyn, freeze_nstep,
                 freeze_rspan, freeze_mfrac, freeze_tsink, log_every):
        if label is None:
            label = ('1st+2nd order heating (revirial)' if second_order
                     else '1st order heating (revirial)')
        assert cfg.Mres is not None, "cfg.Mres must be set before calling"
        assert truncation in ('hard', 'kazantzidis', 'powerlaw'), truncation
        self.tmax = tmax
        self.step_frac, self.dt_growth_max = step_frac, dt_growth_max
        self.strip_number_max, self.floor_orbit_frac = strip_number_max, floor_orbit_frac
        self.max_steps, self.revir_fallback = max_steps, revir_fallback
        self.epsh, self.gamma, self.alpha, self.beta_h = epsh, gamma, alpha, beta_h
        self.t_dyn_mode, self.truncation = t_dyn_mode, truncation
        self.tail_n, self.tail_xi, self.lt_choice = tail_n, tail_xi, lt_choice
        self.label, self.early_terminate = label, early_terminate
        self.dynamical_friction, self.r_stop = dynamical_friction, r_stop
        self.freeze_strip, self.freeze_orbits = freeze_strip, freeze_orbits
        self.freeze_tdyn, self.freeze_nstep = freeze_tdyn, freeze_nstep
        self.freeze_rspan, self.freeze_mfrac = freeze_rspan, freeze_mfrac
        self.freeze_tsink, self.log_every = freeze_tsink, log_every
        self.potential = host
        self.dt_abs = dt_abs_frac * tmax
        self.c2 = np.sqrt(2.) * f2 * (1. + chi_v) if second_order else 0.

        self.ref = numProfile0
        self.r_ref, self.M_ref, self.sig2_ref = _reference_arrays(self.ref, self.c2)
        self.rmax0, self.vmax0 = self.ref.rmax, self.ref.Vmax

        self.o = orbit(xv0)
        self.r = np.sqrt(xv0[0]**2 + xv0[2]**2)
        self.m = self.m0 = self.ref.Mh
        self.Q = 0.
        self.tt_int = np.zeros((3, 3))
        self.hr_prev = None
        self.lt_min = np.inf
        self.lt = cfg.Rres
        self.r_p1 = self.r_p2 = None  # apocenter detector (r local max)
        self.omega_p = self.ref.omega_p

        self.t_list, self.r_list, self.m_list, self.xv_list = [], [], [], []
        self.vmax_list, self.rmax_list, self.lt_list = [], [], []
        self.apo_list = []                       # True where the step re-virialized (apocenter)
        self.clamp_list, self.cw_list, self.cwr_list = [], [], []  # revir: shells, worst %, r
        self.ct_list = []                        # revir reshape: shells heated (denom)
        self.ec_list, self.ecw_list, self.ecwr_list = [], [], []   # per-step expansion clamp
        self.ect_list = []                       # per-step: shells heated (denom)
        self.cur_vmax, self.cur_rmax = self.ref.Vmax, self.ref.rmax
        self.t_last_revir = 0.
        self.r_apo_ref = self.r   # running-max radius: the fallback's stable apocenter clock
        self.dt_prev = np.inf
        self.nstep = 0
        self.t = 0.

        # circuit breaker: freeze a tight, no-longer-stripping orbit. cb_streak counts
        # consecutive re-virializations (~orbits) whose fractional mass loss stayed below
        # freeze_strip; when it reaches freeze_orbits and the orbit is tight (t_dyn at the
        # apocenter below freeze_tdyn) or the step budget is spent, the loop breaks and
        # the settled state is returned frozen. Off unless freeze_strip is set.
        self.cb_on = freeze_strip is not None
        self.cb_streak, self.cb_m_ref = 0, None
        self.cb_tsink_next = -np.inf  # next time the sink-time criterion is worth evaluating
        self.cb_t_sink = np.nan  # sink time of the criterion that passed
        self.cb_lt_lo = self.cb_eta_rout = np.nan   # the regime test the same criterion saw
        self.r_lo = self.r_hi = self.r  # min / max radius over the current not-stripping streak
        self.frozen = self.frozen_mfloor = self.frozen_budget = False
        self.t_wall0 = self.t_log = time.time()   # call start; last heartbeat

        # dt-weighted residence histogram (optional): caller supplies the radial and
        # time bin edges, so the grid is chosen for the ensemble's orbits, not fixed here.
        # dens_kernels adds one kernel-weighted companion histogram per callable w(r, v)
        # (see EvolutionResult.dens_hist_k); the stellar physics stays with the caller.
        self.dens_kernels = dens_kernels
        self.dens_on = dens_redges is not None and dens_tedges is not None
        if self.dens_on:
            self.dens_redges = np.asarray(dens_redges, float)
            self.dens_tedges = np.asarray(dens_tedges, float)
            self.dens_hist = np.zeros((len(self.dens_redges) - 1,
                                       len(self.dens_tedges) - 1))
            self.dens_hist_k = (np.zeros((len(dens_kernels),) + self.dens_hist.shape)
                                if dens_kernels else None)
        else:
            self.dens_redges = dens_redges
            self.dens_tedges = dens_tedges
            self.dens_hist = None
            self.dens_hist_k = None

    def step(self):
        """Advance this subhalo by one accepted timestep. Returns False when the run
        is over (disrupted, parked at r_stop, mass floor with early_terminate, or a
        circuit-breaker freeze), True to keep going. An unresolvable pericentre or
        heating that unbinds the profile raises, with the residence and terminal
        state attached."""
        if self.nstep >= self.max_steps:
            err = PericentreUnresolvedError(
                f"exceeded max_steps={self.max_steps} at t={self.t:.3f}/{self.tmax} Gyr")
            raise self._attach_state(err, span=True)
        self._heartbeat()

        t_orb = tdyn(self.potential, self.r)
        dt_min = max(self.dt_abs, self.floor_orbit_frac * t_orb)
        dt, disrupted = self._predict_dt(t_orb, dt_min)
        dt, r_new, xv, heat, strip, expand, disrupted = self._accept_step(
            dt, dt_min, t_orb, disrupted)

        self.t += dt
        self.r = r_new
        xv_step = xv.copy()  # full 6D cylindrical state at the accepted step
        self.r_apo_ref = max(self.r_apo_ref, r_new)
        if self.cb_on:
            # radial span over the streak
            self.r_lo, self.r_hi = min(self.r_lo, r_new), max(self.r_hi, r_new)
        self.Q, self.tt_int, self.hr_prev = heat
        self._deposit(r_new, xv_step, dt)

        if disrupted:
            self.m = cfg.Mres
            self._record(r_new, xv_step, False, {}, expand)
            return False
        if r_new <= cfg.Rres:
            self.dt_prev = dt
            return True
        if self.r_stop is not None and r_new < self.r_stop:
            # parked at the decay/stall radius: record the terminal state (bound
            # mass preserved, not floored) and stop before plunging to the cusp.
            self._record(r_new, xv_step, False, {}, {})
            return False

        if self.m > cfg.Mres:
            self._king62(*strip)

        did_revir = self._revir_due(r_new)
        revir_tally = self._apply_heating(r_new) if did_revir else {}
        self._record(r_new, xv_step, did_revir, revir_tally, expand)

        self.r_p2, self.r_p1 = self.r_p1, r_new
        self.dt_prev = dt
        self.nstep += 1
        if self.early_terminate and self.m <= cfg.Mres:
            return False
        return not self.frozen   # circuit breaker or mass floor

    def _heartbeat(self):
        """Periodic breadcrumb of the orbital state (off unless log_every set), so a
        run that stalls -- grinding a tight orbit -- can be diagnosed from the log
        after the fact. Checked ~every 1000 steps to keep time() off the hot path."""
        if self.log_every is None or (self.nstep & 1023) != 0:
            return
        now = time.time()
        if now - self.t_log >= self.log_every:
            print(f"[evolve_heating_revirial] {self.label}: "
                  f"t={self.t:.3f}/{self.tmax:.1f} Gyr "
                  f"r={self.r:.4g} kpc m={self.m:.4g} nstep={self.nstep} "
                  f"wall={now - self.t_wall0:.0f}s", flush=True)
            self.t_log = now

    def _predict_dt(self, t_orb, dt_min):
        """Predictor: step_frac of both timescales -- the orbital t_dyn and the King62
        T_strip on the profile as heated so far -- capped against runaway growth and the
        end of the run. Also reports the subhalo disrupted, i.e. the accumulated heating
        already unbinds the analytic profile so it has no tidal radius; the caller
        drives m to the floor and terminates."""
        disrupted = False
        if self.r > cfg.Rres and self.m > cfg.Mres:
            try:
                exp = _ExpandedProfile(self.r_ref, self.M_ref, self.sig2_ref,
                                       self.Q, self.c2)
                T_strip, _, _ = _t_strip(exp, self.potential, self.o.xv,
                                         self.lt_choice, self.t_dyn_mode, t_orb)
            except HeatingUnbindsError:
                disrupted, T_strip = True, np.inf
        else:
            T_strip = np.inf
        dt = min(self.step_frac * t_orb, self.step_frac * T_strip / self.alpha,
                 self.dt_growth_max * self.dt_prev, self.tmax - self.t)
        return max(dt, dt_min), disrupted

    def _accept_step(self, dt, dt_min, t_orb, disrupted):
        """Corrector: integrate a trial step, halving dt (down to dt_min) whenever the
        trial heating unbinds the profile or the King62 strip number overshoots
        strip_number_max, until one is accepted. Returns the accepted step as
        (dt, r, xv, heating, strip, expansion tally, disrupted), which the caller
        commits."""
        # defaults so a terminal step (r <= Rres, or inside r_stop) leaves the
        # accumulators unchanged rather than referencing unset trial vars
        Q, tt_int, hr = self.Q, self.tt_int, self.hr_prev
        strip = 0., self.lt, 0.
        expand = {}
        t0, xv0_step = self.o.t, self.o.xv.copy()
        while True:
            self.o.t, self.o.xv = t0, xv0_step.copy()
            self.o.integrate(self.t + dt, self.potential,
                             self.m if self.dynamical_friction else None)
            xv = self.o.xv
            r_new = np.sqrt(xv[0]**2 + xv[2]**2)
            if r_new <= cfg.Rres or (self.r_stop is not None and r_new < self.r_stop):
                return dt, r_new, xv, (Q, tt_int, hr), strip, expand, disrupted
            Q, tt_int, hr = self._heat(xv, r_new, t_orb, dt)

            if self.m <= cfg.Mres or disrupted:
                strip = 0., cfg.Rres, 0.
            else:
                rate, expand = self._strip_rate(xv, Q, t_orb, dt)
                if rate is not None:
                    strip = rate
                elif dt > dt_min:
                    # the trial heating unbinds the profile; a smaller step lowers
                    # Q and may resolve it -- else the subhalo is disrupted.
                    dt = max(0.5 * dt, dt_min)
                    continue
                else:
                    disrupted = True
                    strip = 0., cfg.Rres, 0.

            strip_number = strip[0]
            if strip_number <= self.strip_number_max:
                return dt, r_new, xv, (Q, tt_int, hr), strip, expand, disrupted
            if dt > dt_min:
                dt = max(0.5 * dt, dt_min)
                continue
            err = PericentreUnresolvedError(
                f"strip number={strip_number:.1f} > {self.strip_number_max} at "
                f"dt_min={dt_min:.2e} Gyr, t={self.t:.3f} Gyr; unresolvable")
            raise self._attach_state(err)

    def _heat(self, xv, r_new, t_orb, dt):
        """Pending heating for a trial step: the Du+24 eq.39 tidal-tensor cumulant and
        the position-independent energy scalar Q, trapezoid-integrated in the heating
        rate over dt. Returns (Q, cumulant, rate); nothing is committed, since the step
        may still be rejected."""
        V = np.sqrt(xv[3]**2 + xv[4]**2 + xv[5]**2)
        x, y, z_c = xv[0] * np.cos(xv[1]), xv[0] * np.sin(xv[1]), xv[2]
        tt_cur = tidalTensor(self.potential, [x, y, z_c])
        tt_int = _cumulant_step(self.tt_int, tt_cur, self.beta_h, t_orb, dt)
        T_shock = r_new / V if V > 0. else 1e10
        adiabatic = (1. + (self.omega_p * T_shock)**2)**(-self.gamma)
        tidalHR = (self.epsh / 3) * adiabatic * np.sum(tt_cur * tt_int)
        dQ = 0.5 * ((tidalHR if self.hr_prev is None else self.hr_prev) + tidalHR) * dt
        return max(self.Q + dQ, 0.), tt_int, tidalHR

    def _strip_rate(self, xv, Q, t_orb, dt):
        """King62 strip number for a trial step, with the tidal radius read off the
        ANALYTICALLY expanded profile -- this orbit's partial expansion at the trial
        heating Q, without a heat_profile call. Returns ((strip number, lt, M(<lt)),
        clamp tally), the first None when the trial heating leaves the profile unbound."""
        tally = {}
        try:
            exp = _ExpandedProfile(self.r_ref, self.M_ref, self.sig2_ref, Q, self.c2,
                                   tally=tally)
        except HeatingUnbindsError:
            return None, tally
        T_strip, lt, M_at_lt = _t_strip(exp, self.potential, xv, self.lt_choice,
                                        self.t_dyn_mode, t_orb)
        return (self.alpha * dt / T_strip, lt, M_at_lt), tally

    def _king62(self, strip_number, lt_step, M_at_lt):
        """Relax the bound mass toward M(<lt) at the King62 rate. M(<lt) comes off the
        analytically expanded profile, whose total is the last re-virialization's bound
        mass and so exceeds the King62-reduced m; capping it at m keeps the relaxation
        one-sided, so an lt enclosing all bound mass strips nothing."""
        m_lt = min(M_at_lt, self.m)
        self.m = max(m_lt + (self.m - m_lt) * np.exp(-strip_number), cfg.Mres)
        self.lt = lt_step
        self.lt_min = min(self.lt_min, self.lt)

    def _revir_due(self, r_new):
        """Whether this step re-virializes: at an apocenter (r a local max) or, for a
        near-circular / distorted orbit with no clean maximum, once revir_fallback
        apocenter dynamical times have passed without one. The fallback clock runs on
        t_dyn(r_apo_ref), NOT t_dyn(current r) -- the latter collapses toward pericentre
        and would fire mid-orbit on an eccentric orbit that already has a well-defined
        apocenter."""
        apo = (self.r_p1 is not None and self.r_p2 is not None
               and self.r_p1 > self.r_p2 and self.r_p1 > r_new)
        due = apo or (self.t - self.t_last_revir >= self.revir_fallback
                      * tdyn(self.potential, self.r_apo_ref))
        return due and self.Q > 0. and self.m > cfg.Mres

    def _apply_heating(self, r_new):
        """Re-virialize: apply the heating accumulated since the last apocenter in one
        shell expansion, reshape to the bound mass m, and reset the ledger. Returns the
        expansion's clamp tally. The freeze tests run on this cadence, so a settled
        clump costs one check per orbit rather than one per step."""
        lt_join = self.lt_min if np.isfinite(self.lt_min) else self.lt
        try:
            self.ref, self.m, tally = _revirialize(
                self.ref, self.Q, self.c2, self.m, lt_join,
                self.truncation, self.tail_n, self.tail_xi)
        except HeatingUnbindsError as err:
            # the accumulated heating unbinds the profile at re-virialization. The engine
            # still raises -- the profile has no tidal radius past this point -- but
            # carries the evolution so far, as PericentreUnresolvedError does.
            raise self._attach_state(err, span=True)
        self.r_ref, self.M_ref, self.sig2_ref = _reference_arrays(self.ref, self.c2)
        self.omega_p = self.ref.omega_p
        self.cur_vmax, self.cur_rmax = self.ref.Vmax, self.ref.rmax
        self.Q, self.tt_int, self.hr_prev = 0., np.zeros((3, 3)), None
        self.lt_min = np.inf
        self.t_last_revir = self.t
        # relative-mass-floor freeze: below the floor the remnant's remaining
        # deposit anywhere is bounded by freeze_mfrac * m0 * (tmax - t), so no
        # orbit-shape gate is needed (see the docstring).
        if self.freeze_mfrac is not None and self.m < self.freeze_mfrac * self.m0:
            self.frozen = self.frozen_mfloor = True
        if self.cb_on:
            self._update_breaker(r_new)
        return tally

    def _update_breaker(self, r_new):
        """Circuit breaker, one check per re-virialization: freeze only a genuinely
        settled clump. Count consecutive orbits that shed less than freeze_strip of
        their bound mass; a stripping orbit resets the streak. Not-stripping is not
        enough: the orbit radius must also stay within a freeze_rspan fractional band
        (min/max over the streak). That rejects a still-decaying orbit (span across
        orbits) and -- the case that matters -- a radial orbit whose deep pericentre
        passages sample small radii (span within an orbit): those rare passages are real
        central residence and must keep being integrated even though DF is off there
        (clamped log) and no mass strips. Freeze when the streak is long, the radius
        settled, and the orbit is tight (t_dyn at the outer radius below freeze_tdyn) or
        the step budget is spent."""
        loss = (0. if self.cb_m_ref is None
                else (self.cb_m_ref - self.m) / self.cb_m_ref)
        if loss < self.freeze_strip:
            self.cb_streak += 1
        else:
            # reset streak + radial span
            self.cb_streak, self.r_lo, self.r_hi = 0, r_new, r_new
        self.cb_m_ref = self.m
        settled = (self.r_hi - self.r_lo) <= self.freeze_rspan * self.r_hi
        if self.cb_streak < self.freeze_orbits or not settled:
            return
        tight = (self.freeze_tdyn is not None
                 and tdyn(self.potential, self.r_hi) < self.freeze_tdyn)
        if tight and self._sink_time_allows_freeze():
            self.frozen = True
            return
        spent = self.freeze_nstep is not None and self.nstep > self.freeze_nstep
        if spent and not self.frozen:
            self.frozen = self.frozen_budget = True

    def _sink_time_allows_freeze(self):
        """The streak shows past quiescence only; with DF on, a settled orbit can still
        decay into the stripping regime later (accelerating friction inward). Allow the
        freeze only when the orbit cannot reach a regime-changing radius (the stripping
        margin or the r_stop halt) within a safety factor of the remaining time.
        Evaluated only when the streak criteria would already have frozen, and memoized:
        a failed test cannot pass before the remaining time shrinks below t_sink/safety,
        so it never runs on the hot path."""
        if self.freeze_tsink is None or not self.dynamical_friction:
            return True
        if self.t < self.cb_tsink_next:
            return False
        regime = {}
        r_ev = _freeze_event_radius(self.ref, self.potential, self.m, self.r_lo,
                                    self.r_stop, self.lt_choice,
                                    self.freeze_tsink.get('eta', 1.2), tally=regime)
        t_sink = (np.inf if r_ev is None
                  else 0. if r_ev >= self.r_lo
                  else df_sink_time(self.potential, self.m, self.r_lo, r_ev))
        safety = self.freeze_tsink.get('safety', 2.)
        if t_sink <= safety * (self.tmax - self.t):
            # earliest time it could pass with unchanged state; re-check at least
            # every 0.5 Gyr (mass loss weakens the drag and lengthens t_sink)
            self.cb_tsink_next = min(self.tmax - t_sink / safety, self.t + 0.5)
            return False
        self.cb_t_sink = t_sink
        self.cb_lt_lo = regime.get('lt_lo', np.nan)
        self.cb_eta_rout = regime.get('eta_rout', np.nan)
        return True

    def _deposit(self, r_new, xv_step, dt):
        """Accumulate this step's dt-weighted residence at the accepted orbital radius.
        m is the mass carried through the step (King62 strips after), and every step
        outcome contributes, terminal ones included."""
        if not self.dens_on:
            return
        v_step = (np.sqrt(xv_step[3]**2 + xv_step[4]**2 + xv_step[5]**2)
                  if self.dens_hist_k is not None else None)
        _dens_accumulate(self.dens_hist, self.dens_redges, self.dens_tedges,
                         r_new, self.m, self.t - 0.5 * dt, dt,
                         hist_k=self.dens_hist_k, kernels=self.dens_kernels, v=v_step)

    def _record(self, r_new, xv_step, did_revir, revir, expand):
        """Append one accepted step to the output lists. Vmax/rmax step-update at
        re-virialization and carry between (aligned to t). revir and expand are the
        heat_profile clamp tallies of this step's re-virialization and analytic
        expansion; an empty dict records NaN, i.e. no such call on this step."""
        self.t_list.append(self.t); self.r_list.append(r_new)
        self.m_list.append(self.m); self.xv_list.append(xv_step)
        self.vmax_list.append(self.cur_vmax); self.rmax_list.append(self.cur_rmax)
        self.lt_list.append(self.lt)
        self.apo_list.append(did_revir)
        self.clamp_list.append(revir.get('shells', np.nan))
        self.cw_list.append(revir.get('worst_pct', np.nan))
        self.cwr_list.append(revir.get('worst_r', np.nan))
        self.ct_list.append(revir.get('total', np.nan))
        self.ec_list.append(expand.get('shells', np.nan))
        self.ecw_list.append(expand.get('worst_pct', np.nan))
        self.ecwr_list.append(expand.get('worst_r', np.nan))
        self.ect_list.append(expand.get('total', np.nan))

    def _attach_state(self, err, span=False):
        """Carry the residence and terminal orbital state on a raised error. span adds
        the circuit breaker's settled radial range."""
        err.dens_hist, err.dens_hist_k = self.dens_hist, self.dens_hist_k
        err.t_last, err.r_last, err.m_last = self.t, self.r, self.m
        if span:
            err.r_lo, err.r_hi = self.r_lo, self.r_hi
        return err

    def finish(self):
        """Flush the pending heating into a closing re-virialization and assemble the
        EvolutionResult. Call once, after the driver stops stepping."""
        # Flush heating accumulated since the last apocenter into a final
        # re-virialization, so the reported final (m, Vmax, rmax, profile) describe one
        # self-consistent equilibrium rather than a current m against a stale last-
        # apocenter structure. If the run ends mid-orbit this is the equilibrium the
        # subhalo relaxes toward (settling takes ~a dynamical time). Skipped once
        # disrupted (m at the floor) -- there is nothing to settle.
        if self.Q > 0. and self.m > cfg.Mres and self.t_list:
            lt_join = self.lt_min if np.isfinite(self.lt_min) else self.lt
            self.ref, self.m, tally = _revirialize(
                self.ref, self.Q, self.c2, self.m, lt_join,
                self.truncation, self.tail_n, self.tail_xi)
            self.m_list[-1] = self.m
            self.vmax_list[-1], self.rmax_list[-1] = self.ref.Vmax, self.ref.rmax
            # the flushed final state is a re-virialized equilibrium
            self.apo_list[-1] = True
            self.clamp_list[-1], self.cw_list[-1], self.cwr_list[-1] = (
                tally['shells'], tally['worst_pct'], tally['worst_r'])
            self.ct_list[-1] = tally['total']

        # Final equilibrium profile sampled onto a log grid over the bound extent.
        # The per-dt loop tracks only scalars; the revirial cadence emits one profile
        # snapshot -- the settled remnant rho(r)/M(r) for callers that inspect the
        # profile (the calibration likelihood ignores these fields).
        if np.isfinite(self.ref.rh) and self.ref.rh > cfg.Rres:
            rg = np.logspace(np.log10(cfg.Rres), np.log10(self.ref.rh), 100)
            r_grid = rg[None, :]
            rho_grid = np.asarray(self.ref.rho(rg), float)[None, :]
            M_grid = np.asarray(self.ref.M(rg), float)[None, :]
            snap_steps = np.asarray([self.nstep])
        else:
            r_grid = rho_grid = M_grid = np.zeros((1, 1))
            snap_steps = np.asarray([])

        return EvolutionResult(
            t=np.asarray(self.t_list), r=np.asarray(self.r_list),
            m=np.asarray(self.m_list),
            vmax=np.asarray(self.vmax_list), rmax=np.asarray(self.rmax_list),
            lt=np.asarray(self.lt_list),
            r_grid=r_grid, rho_snapshots=rho_grid,
            M_snapshots=M_grid, snapshot_steps=snap_steps,
            rmax0=self.rmax0, vmax0=self.vmax0, label=self.label,
            xv=np.asarray(self.xv_list).reshape(-1, 6),
            apo=np.asarray(self.apo_list, dtype=bool),
            clamp=np.asarray(self.clamp_list), clamp_total=np.asarray(self.ct_list),
            clamp_worst=np.asarray(self.cw_list),
            clamp_worst_r=np.asarray(self.cwr_list),
            expand_clamp=np.asarray(self.ec_list),
            expand_clamp_total=np.asarray(self.ect_list),
            expand_clamp_worst=np.asarray(self.ecw_list),
            expand_clamp_worst_r=np.asarray(self.ecwr_list),
            dens_hist=self.dens_hist,
            dens_hist_k=self.dens_hist_k,
            frozen=self.frozen,
            frozen_budget=self.frozen_budget,
            frozen_mfloor=self.frozen_mfloor,
            # a breaker freeze breaks the loop on the spot, so r_lo here is still the
            # streak minimum the sink-time criterion measured from. The mass-floor freeze
            # evaluates no streak -- and with the breaker off r_lo is never updated at
            # all -- so it has no such radius to report.
            freeze_r_lo=(self.r_lo if self.frozen and not self.frozen_mfloor else np.nan),
            freeze_t_sink=self.cb_t_sink,
            freeze_lt_lo=self.cb_lt_lo,
            freeze_eta_rout=self.cb_eta_rout,
        )
