# SatGen imports
# other imports
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline, PchipInterpolator
from scipy.optimize import brentq

import config as cfg
import cosmo as co


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
    pchip = PchipInterpolator(np.log(xm), np.log(ym), extrapolate=False)
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
        # the dVc/dr = 0 condition. Argmax of Vc on a discrete grid is not
        # robust: heated/stripped profiles develop a flat outer shoulder
        # (wide M(r) ~ r^2 region, constant rho, Vc rising slowly), where
        # tiny inter-knot interpolation noise tips the discrete argmax
        # between widely separated grid points and draws diagonal jumps in
        # the (rmax, Vmax) track. The OUTERMOST + → - sign change of f is
        # the true Vc peak. If no interior crossing exists the profile has
        # no interior maximum (Vc monotonic over the bound region or peak
        # below the resolution) — return NaN so the track mask drops the
        # point rather than reporting a boundary value.
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
            k = int(idx_pm[-1])
            # bracket has a confirmed sign change so brentq cannot fail;
            # if it ever does, that is a real bug and should propagate
            rmax = brentq(
                lambda r: 4.0*np.pi*r**3 * float(self.rho(r)) - float(self.MInt(r)),
                rr[k], rr[k+1], xtol=1e-8,
            )
            self.rmax = float(rmax)
            self.Vmax = float(self.Vcirc(rmax))

        # sigma_r^2 at ri grid points via exact spline antiderivative (Jeans, isotropic)
        fvals = np.maximum(self.rhovals, 0.) * np.maximum(self.Mr, 0.) / self.ri**2
        f_spl = InterpolatedUnivariateSpline(self.ri, fvals, k=3, ext='zeros')
        F = f_spl.antiderivative()
        cumint = F(self.ri[-1]) - F(self.ri)
        sig2_vals = np.zeros_like(self.rhovals)
        np.divide(cfg.G * cumint, self.rhovals, out=sig2_vals, where=self.rhovals > 0)
        sig2_pos = np.maximum(sig2_vals, 0.)
        self._sig2 = (_log_pchip(self.ri, sig2_pos)
                      or InterpolatedUnivariateSpline(self.ri, sig2_pos, k=3, ext='const'))

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

        # enforce monotonic shell ordering; skip when either Menc is non-positive
        # (spline undershoot at small r), where the (m_i/m_{i+1})**(-1/3) factor
        # is ill-defined
        if i < count_r - 1 and Menc[i] > 0 and Menc[i + 1] > 0:
            limit = (
                1.0
                - r / ri[i + 1]
                * (Menc[i] / Menc[i + 1]) ** (-1.0 / 3.0)
                * (1.0 - perturb[i + 1])
            )
            perturb[i] = min(perturb[i], limit)

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

    # M(<r) at the post-expansion radii. Du+24's monotonic-shell clamp
    # above guarantees rf is already sorted under their assumptions
    # (Menc strictly positive on the sample grid), so this sort is a no-op
    # in the common case. The clamp is bypassed when Menc[i+1] <= 0 (spline
    # undershoot at very small r), where the (Menc[i]/Menc[i+1])^(-1/3)
    # factor is undefined; sort+cumsum recovers a self-consistent M(<r) in
    # that fallback. Menc[bound][order] would be wrong here — those values
    # are cumulative masses at the *original* shell positions, not at the
    # post-sort radii.
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
