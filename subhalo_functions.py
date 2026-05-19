# SatGen imports
# other imports
import warnings
from typing import cast

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator
from scipy.optimize import brentq
from scipy.special import gammaincc, gammaln

import config as cfg
import cosmo as co


# heat_profile is called every ODE step (~30k/run); the local monotonic
# clamp fires routinely near the inner grid edge. One warning per process
# is enough to flag the issue without spamming the log.
_heat_profile_clamp_warned = False


class SIS(object):
    """
    Singular Isothermal Sphere truncated at the virial radius.

        rho(r) = sigma_v^2 / (2 pi G r^2)   for r <= rh
        M(<r)  = 2 sigma_v^2 r / G           for r <= rh
        Vcirc  = sqrt(2) sigma_v = const

    Syntax: host = SIS(M, Delta=200., z=0.)
    """
    def __init__(self, M, Delta=200., z=0.):
        self.Mh = M
        self.Deltah = Delta
        self.z = z
        rhoc = co.rhoc(z, cfg.h, cfg.Om, cfg.OL)
        self.rh = (3.*M / (cfg.FourPi * Delta * rhoc))**(1./3.)
        self.sigma_v2 = cfg.G * M / (2. * self.rh)
        self.Vc2 = 2. * self.sigma_v2
        self.Vc = np.sqrt(self.Vc2)
        self.Vmax = self.Vc
        self.rmax = self.rh
        self.rs = self.rh
        self.ch = 1.

    def rho(self, R, z=0.):
        r = np.sqrt(R**2. + z**2.)
        return np.where((r > 0.) & (r <= self.rh),
                        self.sigma_v2 / (2.*np.pi*cfg.G*r**2.), 0.)

    def M(self, R, z=0.):
        r = np.sqrt(R**2. + z**2.)
        return np.minimum(2.*self.sigma_v2*r/cfg.G, self.Mh)

    def rhobar(self, R, z=0.):
        r = np.sqrt(R**2. + z**2.)
        return self.M(r) / (cfg.FourPiOverThree * r**3.)

    def tdyn(self, R, z=0.):
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R, z))

    def Phi(self, R, z=0.):
        r = np.sqrt(R**2. + z**2.)
        phi_in = 2.*self.sigma_v2*(np.log(np.minimum(r, self.rh)/self.rh) - 1.)
        phi_out = -cfg.G*self.Mh/r
        return np.where(r <= self.rh, phi_in, phi_out)

    def fgrav(self, R, z):
        r2 = R**2. + z**2.
        r = np.sqrt(r2)
        fac_in  = -self.Vc2 / r2
        fac_out = -cfg.G*self.Mh / (r2 * r)
        fac = np.where(r <= self.rh, fac_in, fac_out)
        return fac*R, fac*0., fac*z

    def Vcirc(self, R, z=0.):
        r = np.sqrt(R**2. + z**2.)
        vc = np.where(r <= self.rh, self.Vc, np.sqrt(cfg.G*self.Mh/r))
        return vc

    def sigma(self, R, z=0.):
        r = np.sqrt(R**2. + z**2.)
        # Jeans isotropic solution for truncated SIS: sigma_r^2 = sigma_v^2*(1-(r/rh)^2)
        return np.sqrt(self.sigma_v2 * np.maximum(1. - (r/self.rh)**2., 0.))


def _log_pchip(x, y, eps_rel=1e-30, clamp_below='value', clamp_above='value',
               leading_only=False):
    """log-log PCHIP on positive (x, y). Power-law inputs become straight
    lines in log-log so PCHIP is exact between knots — useful for cusps
    (M ~ r^2-3, rho ~ r^-1.5) where linear PCHIP carries O(h^2) error.

    Returns None when the data has fewer than 2 strictly-positive knots
    (above eps_rel * y.max()); the caller can then fall back to linear
    PCHIP. Queries outside [x_inside[0], x_inside[-1]] are clamped — never
    extrapolated. clamp_below/clamp_above: 'value' returns the boundary y;
    'zero' returns 0.

    leading_only=True restricts the support to the leading positive run
    (truncates at the first zero/below-eps knot). For rho this preserves
    interior plateaus exactly: any region after the first dM/dr=0 reads
    as clamp_above. Real heated profiles are monotone so this is a no-op;
    synthetic profiles with interior plateaus get the right zero floor.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.size == 0 or y.max() <= 0.:
        return None
    pos = y > eps_rel * y.max()
    mask = (np.cumprod(pos.astype(int)) > 0) if leading_only else pos
    if mask.sum() < 2:
        return None
    xm, ym = x[mask], y[mask]
    # drop log-space near-duplicates: when consecutive xm values agree to
    # within machine epsilon (~1e-16 relative), log(xm) collapses and
    # PchipInterpolator rejects the non-strictly-increasing input. Happens
    # naturally on the outer Kazantzidis tail where M(r) plateaus and the
    # heat_profile shell expansion produces clustered r_f at floating-point
    # precision. 1e-12 leaves a comfortable margin above double precision.
    log_xm = np.log(xm)
    if len(log_xm) > 1:
        good = np.concatenate(([True], np.diff(log_xm) > 1e-12))
        if good.sum() < 2:
            return None
        xm, ym, log_xm = xm[good], ym[good], log_xm[good]
    pchip = PchipInterpolator(log_xm, np.log(ym), extrapolate=False)
    x_lo, x_hi = float(xm[0]), float(xm[-1])
    y_lo, y_hi = float(ym[0]), float(ym[-1])
    lo_val = y_lo if clamp_below == 'value' else 0.
    hi_val = y_hi if clamp_above == 'value' else 0.

    def f(r):
        scalar_in = np.ndim(r) == 0
        r_arr = np.atleast_1d(np.asarray(r, dtype=float))
        out = np.empty_like(r_arr)
        below = r_arr < x_lo
        above = r_arr > x_hi
        inside = ~(below | above)
        if inside.any():
            out[inside] = np.exp(pchip(np.log(r_arr[inside])))
        out[below] = lo_val
        out[above] = hi_val
        return out.item() if scalar_in else out
    return f


def tidalTensor(hostProfile, coords):
    x,y,z = coords
    r = np.sqrt(x**2+y**2+z**2)
    Mh = hostProfile.M(r)
    rho = hostProfile.rho(r)
    rarb = np.array([[x**2,x*y,x*z],[x*y,y**2,y*z],[x*z,z*y,z**2]])
    return cfg.G*Mh/r**3 * (3*rarb/r**2 - np.identity(3))- 4*np.pi*cfg.G*rho*rarb/r**2

class NumericProfile(object):
    def __init__(self, ri, Mr):
        self.ri = ri
        self.Mr = Mr
        self.rh = np.max(ri)
        self.Mh = np.max(Mr)

        # log-log PCHIP for M and rho. Power-law cusps (M ~ r^2-3,
        # rho ~ r^-1.5) become straight lines in log-log so PCHIP is exact
        # between knots; linear-axis PCHIP carries O(h^2) error there.
        # Fall back to linear PCHIP on degenerate inputs (all-zero,
        # fewer-than-2 positive knots).
        self.MInt = (_log_pchip(self.ri, self.Mr)
                     or PchipInterpolator(self.ri, self.Mr, extrapolate=False))
        # density from a finite-difference dM/dr (np.gradient: 2nd-order
        # central interior, one-sided at the ends). Spline-derivative rho
        # rings on near-flat M(r) regions (most mass concentrated in a
        # narrow shell post-heating); FD is exact on plateaus.
        dMdr = np.gradient(self.Mr, self.ri)
        self.rhovals = np.maximum(dMdr, 0.) / (4.0*np.pi*self.ri**2)
        self.rhoInt = (_log_pchip(self.ri, self.rhovals,
                                  clamp_below='zero', clamp_above='zero',
                                  leading_only=True)
                       or PchipInterpolator(self.ri, self.rhovals, extrapolate=False))

        # rmax / Vmax via root-finding of f(r) = 4*pi*r^3*rho - M = 0,
        # the dVc/dr = 0 condition. Heated/stripped profiles can be
        # multimodal: a small outer Vc bump (from the heating-induced
        # shoulder) sits beside the inner cusp peak, and tiny inter-knot
        # interpolation noise tips the argmax between them step-to-step,
        # producing diagonal jumps in the (rmax, Vmax) track. brentq each
        # + → - sign change of f and pick the GLOBAL Vc maximum across
        # roots. For unimodal profiles there is one root and this matches
        # the old behaviour exactly. No interior crossing means Vc is
        # monotonic over the bound region; return NaN so the track mask
        # drops the point rather than reporting a boundary value.
        rr = np.logspace(np.log10(self.ri[0]), np.log10(self.rh), 200)
        f_grid = 4.0 * np.pi * rr**3 * self.rho(rr) - self.M(rr)
        # exclude the last few grid points: rho rolls off as the
        # interpolator runs out of data near r=rh, so f goes negative
        # there even when no real peak is present
        idx_pm = np.where((f_grid[:-1] > 0.) & (f_grid[1:] < 0.))[0]
        idx_pm = idx_pm[idx_pm < len(rr) - 5]
        if len(idx_pm) == 0:
            self.rmax = np.nan
            self.Vmax = float(np.max(self.Vcirc(rr)))
        else:
            best_r, best_V = -1., -1.
            for k in idx_pm:
                rk = cast(float, brentq(
                    lambda r: 4.0*np.pi*r**3 * float(self.rho(r)) - float(self.MInt(r)),
                    rr[int(k)], rr[int(k)+1], xtol=1e-8,
                ))
                Vk = float(self.Vcirc(rk))
                if Vk > best_V:
                    best_r, best_V = rk, Vk
            self.rmax = float(best_r)
            self.Vmax = float(best_V)

        # sigma_r^2 at ri grid points via exact spline antiderivative (Jeans, isotropic)
        fvals = np.maximum(self.rhovals, 0.) * np.maximum(self.Mr, 0.) / self.ri**2
        f_spl = InterpolatedUnivariateSpline(self.ri, fvals, k=3, ext=1)
        F = f_spl.antiderivative()
        cumint = F(self.ri[-1]) - F(self.ri)
        sig2_vals = np.zeros_like(self.rhovals)
        np.divide(cfg.G * cumint, self.rhovals, out=sig2_vals, where=self.rhovals > 0)
        sig2_pos = np.maximum(sig2_vals, 0.)
        self._sig2 = (_log_pchip(self.ri, sig2_pos)
                      or InterpolatedUnivariateSpline(self.ri, sig2_pos, k=3, ext=3))

        # half-mass radius and angular frequency at r_half, for the
        # adiabatic correction in tidal heating (Pullen+14 / Gnedin+99,
        # Benson+Du22 eq. 3, Du+24 sec. IV.C — omega_p evaluated at the
        # subhalo half-mass radius)
        M_target = 0.5 * self.Mh
        m_inner = float(self.MInt(self.ri[0]))
        m_outer = float(self.MInt(self.ri[-1]))
        if M_target <= m_inner:
            self.r_half = self.ri[0]
        elif M_target >= m_outer:
            self.r_half = self.ri[-1]
        else:
            try:
                self.r_half = brentq(lambda r: float(self.MInt(r)) - M_target,
                                     self.ri[0], self.ri[-1])
            except (ValueError, RuntimeError):
                idx = max(1, min(np.searchsorted(self.Mr, M_target), len(self.ri) - 1))
                dM = max(self.Mr[idx] - self.Mr[idx-1], 1e-30)
                frac = (M_target - self.Mr[idx-1]) / dM
                self.r_half = self.ri[idx-1] + frac * (self.ri[idx] - self.ri[idx-1])
        # use M(<r_half), not M_target = Mh/2: when r_half clamps to a grid
        # edge (degenerate cases where Mh/2 falls outside [m_inner, m_outer])
        # the two are not equal and the formula must follow the actual r_half
        M_at_rhalf = float(self.MInt(self.r_half))
        # max() guards against r_half == 0 (impossible by construction, but
        # keeps the division robust if ri[0] is ever set to zero)
        self.omega_p = np.sqrt(cfg.G * M_at_rhalf / max(self.r_half, self.ri[0])**3)

    def rho(self, R, z=0.):
        r = np.sqrt(R**2 + z**2)
        return self.rhoInt(r)

    def M(self, R, z=0.):
        r = np.sqrt(R**2 + z**2)
        return self.MInt(r)

    def rhobar(self, R, z=0.):
        r = np.sqrt(R**2.+z**2.)
        return self.M(r)/(cfg.FourPiOverThree*r**3)

    def tdyn(self,R,z=0.):
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))

    def Phi(self,R,z=0.):
        r = np.sqrt(R**2.+z**2.)
        phi1 = -cfg.G*self.Mh*quad(lambda x: 1/x**2, self.rh, np.inf)[0]

        if isinstance(r,list) or isinstance(r,np.ndarray):
            phiList =[]
            for i in range(len(r)):
                phiList.append(-cfg.G*quad(lambda x: self.M(x)/x**2, r[i], self.rh)[0])
            phi2 = np.array(phiList)
        else:
            phi2 = -cfg.G*quad(lambda x: self.M(x)/x**2, r, self.rh)[0]

        return phi1 + phi2

    def fgrav(self,R,z=0.):
        r = np.sqrt(R**2.+z**2.)
        fac = -cfg.G*self.M(r)/r**2
        return fac*R/r, 0., fac*z/r

    def Vcirc(self,R,z=0.):
        r = np.sqrt(R**2 + z**2)
        # spline can undershoot M(r) below 0 at very small r; clamp to keep sqrt real
        return np.sqrt(np.maximum(cfg.G*self.M(r)/r, 0.))

    def sigma(self,R,z=0.):
        r = np.sqrt(R**2.+z**2.)
        if isinstance(r,list) or isinstance(r,np.ndarray):
            intList =[]
            for i in range(len(r)):
                intList.append(cfg.G*quad(lambda x: self.rho(x)*self.M(x)/x**2,r[i],self.rh)[0])
            integ = np.array(intList)
        else:
            integ = cfg.G*quad(lambda x: self.rho(x)*self.M(x)/x**2,r,self.rh)[0]
        return np.sqrt(integ/self.rho(r))


def heat_profile(profile: NumericProfile, eps, count_per_decade=100):
    """
    Apply monotonic heating algorithm to a NumericProfile.

    Parameters
    ----------
    profile : NumericProfile
        Initial halo profile
    eps : function
        Specific energy injection function eps(r)
    count_per_decade : int
        Radial resolution

    Returns
    -------
    NumericProfile
        Heated profile
    """

    global _heat_profile_clamp_warned
    G = cfg.G

    rvir = profile.rh

    # --------------------------------------------------
    # Radial grid
    # --------------------------------------------------
    rmin = min(profile.ri)
    rmax = 10.0 * rvir

    decades = np.log10(rmax / rmin)
    count_r = int(decades * count_per_decade + 1)

    ri = np.logspace(np.log10(rmin), np.log10(rmax), count_r)

    Menc = profile.M(ri)
    perturb = np.zeros_like(ri)

    # clamp telemetry (first-occurrence warning below)
    clamp_count = 0
    clamp_worst_excess = 0.0
    clamp_worst_r = np.nan

    # --------------------------------------------------
    # Compute perturbation outer → inner
    # --------------------------------------------------
    for i in reversed(range(count_r)):

        r = ri[i]
        m = Menc[i]

        if m > 0:
            perturb[i] = 2.0 * eps(r) * r / (G * m)
        else:
            perturb[i] = 0.0

        # Local shell-crossing clamp. r_f(r_i) <= r_f(r_{i+1}) gives a
        # per-shell upper bound on xi (with an extra (M_i/M_{i+1})^(-1/3)
        # factor that tightens the constraint to keep adjacent shells from
        # crowding). Skip when either Menc is non-positive (spline
        # undershoot at small r); the (M_i/M_{i+1})^(-1/3) factor is
        # undefined there.
        # TODO(fabio): add the full Du+24 §IV.C shell-crossing handling.
        # The clamp below is a local, shell-by-shell limit. Du+24 eqs. 41-44
        # do more: find the single radius r_crossing where dxi/dr = 0
        # (eq. 44) and xi is continuous (eq. 43), then for r < r_crossing
        # replace eps(r) with
        # xi(r_crossing) * G*M(r) / (2r) (eq. 42), making eps proportional
        # to the gravitational potential inward of r_crossing. The clamp
        # approximates this globally-flattened xi(r) with a chain of local
        # inequalities; the two coincide for smooth-enough profiles but
        # not in general (see Du+24 Fig. 7 for the cored gamma=0 case).
        if i < count_r - 1 and Menc[i] > 0 and Menc[i + 1] > 0:
            limit = (
                1.0
                - r / ri[i + 1]
                * (Menc[i] / Menc[i + 1]) ** (-1.0 / 3.0)
                * (1.0 - perturb[i + 1])
            )
            if perturb[i] > limit:
                clamp_count += 1
                excess = (perturb[i] / limit) - 1.0 if limit > 0 else np.inf
                if excess > clamp_worst_excess:
                    clamp_worst_excess = excess
                    clamp_worst_r = r
                perturb[i] = limit

    if clamp_count > 0 and not _heat_profile_clamp_warned:
        _heat_profile_clamp_warned = True
        warnings.warn(
            f"heat_profile clamped {clamp_count} shell(s) to the local "
            f"monotonic limit; worst overshoot "
            f"{clamp_worst_excess * 100.0:.2f}% at r={clamp_worst_r:.3e} kpc. "
            f"Full Du+24 §IV.C r_crossing handling not yet implemented "
            f"(TODO in heat_profile). Suppressed after first occurrence.",
            RuntimeWarning,
            stacklevel=2,
        )

    # --------------------------------------------------
    # Final energies
    # --------------------------------------------------
    Ef = G * Menc / ri * (-1.0 + perturb)

    # --------------------------------------------------
    # Shell masses
    # --------------------------------------------------
    Mshell = np.zeros_like(Menc)
    Mshell[0] = Menc[0]
    Mshell[1:] = np.diff(Menc)

    # --------------------------------------------------
    # Bound shells
    # --------------------------------------------------
    bound = (Ef < 0) & (Mshell > 0)

    rf = np.full_like(ri, np.inf)
    rf[bound] = -G * Menc[bound] / Ef[bound]

    # --------------------------------------------------
    # If halo destroyed
    # --------------------------------------------------
    if np.count_nonzero(bound) <= 2:
        raise RuntimeError("Heating unbinds the halo")

    r_bound = rf[bound]
    Mshell_bound = Mshell[bound]

    # M(<r) at the post-expansion radii. The shell-crossing guard above
    # rules out crossings on the bulk grid (Menc strictly positive), so
    # this sort is a no-op in the common case. The guard is bypassed when
    # Menc[i+1] <= 0 (spline undershoot at very small r), where the
    # (Menc[i]/Menc[i+1])^(-1/3) factor is undefined; sort+cumsum recovers
    # a self-consistent M(<r) in that fallback. Menc[bound][order] would
    # be wrong here -- those values are cumulative masses at the *original*
    # shell positions, not at the post-sort radii.
    order = np.argsort(r_bound)
    r_bound = r_bound[order]
    M_bound = np.cumsum(Mshell_bound[order])

    # collapse exact duplicates in r — keep the last index of each run so
    # M_bound[k] retains the full cumulative mass at r_bound[k]. np.unique
    # would return the first index and undercount when duplicates exist.
    keep = np.ones(len(r_bound), dtype=bool)
    keep[:-1] = r_bound[:-1] != r_bound[1:]
    r_bound = r_bound[keep]
    M_bound = M_bound[keep]

    # Rebin to a uniform log-spaced grid via log-log PCHIP. Shell expansion
    # produces irregular knot spacing in r_bound (large gaps near the outer
    # edge where shells move farthest); log-log PCHIP is exact for power-law
    # M(r) between knots and stays monotone on the cumsum input.
    n_clean = max(len(r_bound), 200)
    ri_clean = np.logspace(np.log10(r_bound[0]), np.log10(r_bound[-1]), n_clean)
    # snap endpoints to exact knot positions; logspace float roundoff can
    # nudge samples fractionally outside [r_bound[0], r_bound[-1]] and the
    # interpolators with extrapolate=False would clamp/NaN there.
    ri_clean[0] = r_bound[0]
    ri_clean[-1] = r_bound[-1]
    M_pchip = (_log_pchip(r_bound, M_bound)
               or PchipInterpolator(r_bound, M_bound, extrapolate=False))
    Mr_clean = M_pchip(ri_clean)

    return NumericProfile(ri_clean, Mr_clean)


def _ln_gamma_upper(a, x):
    """ln of the unnormalized upper incomplete gamma int_x^inf t^(a-1) e^-t dt,
    for a > 0. gammaln + log(gammaincc) keeps a large `a` from overflowing
    gamma(a). _solve_r_decay caps r_decay so the tail exponent a = 3 + kappa
    stays positive; truncate_kazantzidis routes any remaining a <= 0 (a join
    slope steeper than r^-3) to a hard cut -- so a > 0 always holds here.
    """
    if a <= 0.:
        raise ValueError(f"_ln_gamma_upper requires a > 0, got a={a}")
    x = np.asarray(x, dtype=float)
    gc = gammaincc(a, x)
    # gammaincc underflows to 0 once x encloses essentially the whole tail;
    # the tiny floor then gives a large-negative ln (-> exp -> 0), the correct
    # limit M(<r) -> M_t + M_tail_to_z, not a masked error
    return gammaln(a) + np.log(np.where(gc > 0., gc, np.finfo(float).tiny))


def _tail_mass_inf(r_decay, r_t, rho_t, s):
    """Total Kazantzidis-tail mass beyond r_t (analytic, overflow-safe)."""
    z = r_t / r_decay
    kappa = z + s
    lnM = (np.log(cfg.FourPi * rho_t) + 3. * np.log(r_decay) + z
           - kappa * np.log(z) + _ln_gamma_upper(3. + kappa, z))
    return float(np.exp(lnM))


def _solve_r_decay(target, r_t, rho_t, s):
    """Find r_decay so the Kazantzidis tail beyond r_t carries `target` mass.

    Tail mass is monotone in r_decay; for a tail slope shallower than r^-3 it
    grows without bound, for a steeper slope it saturates -- a target above
    the saturation value has no solution and raises.
    """
    f = lambda rd: _tail_mass_inf(rd, r_t, rho_t, s) - target
    # cap r_decay so the tail exponent a = 3 + r_t/r_decay + s stays > 0;
    # for s < -3 that bounds r_decay below r_t/(-3-s) (a <= 0 -- a join
    # steeper than r^-3 -- has no incomplete-gamma tail mass)
    hi_max = 1e4 * r_t
    if s < -3.:
        hi_max = min(hi_max, 0.999 * r_t / (-3. - s))
    hi = min(r_t, hi_max)
    while hi < hi_max and f(hi) <= 0.:
        hi = min(hi * 2., hi_max)
    if f(hi) <= 0.:
        raise ValueError(
            f"Kazantzidis tail cannot carry M_tail={target:.3e} Msun at "
            f"r_t={r_t:.3e} kpc (join slope {s:.2f}): tail mass saturates"
        )
    lo = r_t * 1e-2
    for _ in range(80):
        if f(lo) < 0.:
            break
        lo *= 0.5
    else:
        raise ValueError(f"could not bracket r_decay below at r_t={r_t:.3e}")
    return cast(float, brentq(f, lo, hi))


def _profile_cut(profile, m_total):
    """Hard cut: truncate `profile` at the radius enclosing m_total.

    The truncate_kazantzidis fallback when no tail can be built. The cut
    radius is M^-1(m_total) -- the same construction as evolve_heating's
    'hard' branch -- so M(<r) has no jump at the outer edge.
    """
    m_total = min(m_total, float(profile.M(profile.rh)))
    r_cut = cast(float, brentq(lambda r: float(profile.M(r)) - m_total,
                               profile.ri[0], profile.rh, xtol=1e-8))
    n = max(100, int(np.log10(r_cut / profile.ri[0]) * 100.))
    rvals = np.logspace(np.log10(profile.ri[0]), np.log10(r_cut), n)
    Mr = np.asarray(profile.M(rvals), dtype=float)
    Mr[-1] = m_total
    return NumericProfile(rvals, Mr)


def truncate_kazantzidis(profile, r_t, r_decay=None, m_total=None, slope=None):
    """Kazantzidis+06 exponentially-truncated tail stitched onto a
    NumericProfile at r_t, in place of a hard cut.

    For r <= r_t the profile is unchanged. For r > r_t,

        rho(r) = rho'(r_t) (r/r_t)^kappa exp(-(r - r_t)/r_decay)
        kappa  = r_t/r_decay + dln(rho')/dln(r)|_{r_t}

    kappa makes the logarithmic density slope continuous across r_t
    (Kazantzidis+06 eq. 3-4). Enclosed mass beyond r_t is the analytic
    incomplete-gamma integral of that density.

    Exactly one of r_decay, m_total is given:
      r_decay -- a fixed decay scale.
      m_total -- total enclosed mass of the result; r_decay is solved so the
                 tail carries m_total - M(<r_t).

    `slope` overrides the join logarithmic density slope dln(rho)/dln(r)|_{r_t}.
    Pass it when r_t sits at profile.rh -- a NumericProfile's np.gradient
    density is unreliable at its outer knot. Left None it is finite-differenced.

    Returns a NumericProfile spanning [profile.ri[0], r_t + 50 r_decay].
    """
    if (r_decay is None) == (m_total is None):
        raise ValueError("pass exactly one of r_decay, m_total")

    r_t = float(min(max(r_t, profile.ri[0] * (1. + 1e-6)), profile.rh))
    rho_t = float(profile.rho(r_t))
    M_t = float(profile.M(r_t))

    # a Kazantzidis tail needs a positive join density and a defined join
    # slope; a heated profile in deep stripping can have rho ~ 0 near r_t.
    # There fall back to a hard cut (mass-conserving; m_total must be given).
    degenerate = not (rho_t > 0.)
    if slope is not None:
        s = float(slope)
    else:
        # central difference when r_t is interior; one step back from the
        # outer edge otherwise (less accurate -- callers pass `slope` there)
        delta = 3e-3
        if r_t * (1. + delta) <= profile.rh:
            r_lo, r_hi = r_t * (1. - delta), r_t * (1. + delta)
        else:
            r_lo, r_hi = r_t * (1. - 2. * delta), r_t
        rho_lo, rho_hi = float(profile.rho(r_lo)), float(profile.rho(r_hi))
        if rho_lo > 0. and rho_hi > 0.:
            s = (np.log(rho_hi) - np.log(rho_lo)) / (np.log(r_hi) - np.log(r_lo))
        else:
            s, degenerate = 0., True

    if degenerate:
        if m_total is None:
            raise ValueError("truncate_kazantzidis: degenerate join, pass m_total")
        return _profile_cut(profile, m_total)

    if m_total is not None:
        target = m_total - M_t
        if target <= max(1e-9 * M_t, 0.):
            return _profile_cut(profile, m_total)  # nothing to put in the tail
        try:
            r_decay = _solve_r_decay(target, r_t, rho_t, s)
        except ValueError:
            # no tail of the local join slope can carry this budget (a steep
            # join in deep stripping) -- hard-cut at M^-1(m_total) instead
            return _profile_cut(profile, m_total)
    assert r_decay is not None  # one of r_decay/m_total is set (checked above)

    z = r_t / r_decay
    kappa = z + s
    a = 3. + kappa
    if a <= 0.:
        # a <= 0 (join steeper than r^-3) has no incomplete-gamma tail mass.
        # _solve_r_decay keeps the m_total path a > 0, so this is only the
        # fixed-r_decay case -- hard-cut if we have a budget, else raise.
        if m_total is None:
            raise ValueError(
                f"truncate_kazantzidis: tail exponent a={a:.2f} <= 0 "
                f"(join slope {s:.2f}, r_decay too large for this join)")
        return _profile_cut(profile, m_total)
    # M(<r) = M_t + C [Gamma(a, z) - Gamma(a, r/r_decay)],
    # C = 4 pi rho_t r_decay^3 exp(z) / z^kappa
    lnC = (np.log(cfg.FourPi * rho_t) + 3. * np.log(r_decay)
           + z - kappa * np.log(z))
    M_tail_to_z = np.exp(lnC + _ln_gamma_upper(a, z))

    # one uniform log grid across [ri[0], r_t + 50 r_decay] -- the tail is
    # cut at 50 decay lengths (inspired by Galacticus); a single spacing
    # keeps np.gradient density continuous across r_t (a split inner/outer
    # grid would jump).
    r_out = r_t + 50. * r_decay
    n = max(300, int(np.log10(r_out / profile.ri[0]) * 100.))
    rvals = np.logspace(np.log10(profile.ri[0]), np.log10(r_out), n)
    inner = rvals <= r_t
    Mr = np.empty_like(rvals)
    Mr[inner] = profile.M(rvals[inner])
    r_tail = rvals[~inner]
    Mr[~inner] = (M_t + M_tail_to_z
                  - np.exp(lnC + _ln_gamma_upper(a, r_tail / r_decay)))
    if m_total is not None:
        Mr[-1] = m_total  # the exp(-50) tail remainder past r_out
    if not np.all(np.isfinite(Mr)):
        # the analytic incomplete-gamma tail can lose precision at extreme
        # deep-stripping parameters; hard-cut that step (mass still conserved)
        if m_total is None:
            raise ValueError("truncate_kazantzidis: non-finite tail at fixed r_decay")
        return _profile_cut(profile, m_total)
    return NumericProfile(rvals, Mr)
