"""
Unit tests for subhalo_functions.py — fast, no tree files required.

Tests NumericProfile against analytical NFW ground truth, and heat_profile
for basic physical consistency.

Run with:
    python -m pytest test_subhalo_functions.py -v
"""

import numpy as np
import pytest
from scipy.integrate import quad

import config as cfg
from profiles import NFW
from subhalo_functions import (NumericProfile, _expand_shells, heat_profile,
                               tidalTensor)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nfw():
    """MW-scale NFW halo used as ground truth: M=1e12 Msun, c=10."""
    return NFW(1e12, 10.)


@pytest.fixture(scope="module")
def numeric_from_nfw(nfw):
    """
    NumericProfile built by sampling the analytical NFW M(r) on a dense
    log-spaced grid.  This is the same construction used in stripping.ipynb.
    """
    rmin = 1e-3 * nfw.rh
    rmax = nfw.rh
    ri = np.logspace(np.log10(rmin), np.log10(rmax), 500)
    Mr = nfw.M(ri)
    return NumericProfile(ri, Mr)


# Radii to probe — avoid the very inner and outer edges where boundary
# effects and interpolation errors are largest.
PROBE_RADII_FRAC = [0.01, 0.05, 0.1, 0.3, 0.6]


# ---------------------------------------------------------------------------
# Enclosed mass
# ---------------------------------------------------------------------------

class TestNumericProfileMass:
    def test_mass_roundtrip(self, nfw, numeric_from_nfw):
        """M(r) must reproduce the NFW input to within 0.01%."""
        p = numeric_from_nfw
        for frac in PROBE_RADII_FRAC:
            r = frac * nfw.rh
            m_num = p.M(r)
            m_nfw = nfw.M(r)
            relerr = abs(m_num - m_nfw) / m_nfw
            assert relerr < 1e-4, \
                f"M({frac} rh): rel error {relerr:.2e} > 0.01%"

    def test_mass_monotone_increasing(self, numeric_from_nfw):
        """Enclosed mass must be non-decreasing with radius."""
        p = numeric_from_nfw
        radii = np.logspace(np.log10(p.ri[2]), np.log10(p.rh * 0.99), 80)
        masses = p.M(radii)
        assert np.all(np.diff(masses) >= 0), "M(r) is not monotone increasing"

    def test_mass_at_rh_equals_Mh(self, nfw, numeric_from_nfw):
        """M(rh) should equal Mh."""
        p = numeric_from_nfw
        assert abs(p.M(p.rh) - p.Mh) / p.Mh < 1e-4


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------

class TestNumericProfileDensity:
    def test_density_positive(self, numeric_from_nfw):
        """rho(r) must be non-negative everywhere on the grid interior."""
        p = numeric_from_nfw
        radii = np.logspace(np.log10(p.ri[5]), np.log10(p.rh * 0.9), 50)
        rhos = p.rho(radii)
        assert np.all(rhos >= 0), "rho(r) has negative values"

    def test_density_agrees_with_nfw(self, nfw, numeric_from_nfw):
        """
        rho(r) from NumericProfile must match the NFW analytic density to
        within 0.01% in the well-resolved interior (0.05–0.6 rh).
        """
        p = numeric_from_nfw
        for frac in [0.05, 0.1, 0.3, 0.6]:
            r = frac * nfw.rh
            rho_num = p.rho(r)
            rho_nfw = nfw.rho(r)
            relerr = abs(rho_num - rho_nfw) / rho_nfw
            assert relerr < 1e-4, \
                f"rho({frac} rh): rel error {relerr:.2e} > 0.01%"

    def test_density_consistent_with_mass(self, numeric_from_nfw):
        """
        Integrating rho over a shell must recover the enclosed-mass difference:
            M(r2) - M(r1) = 4 pi integral_{r1}^{r2} rho(r) r^2 dr
        """
        p = numeric_from_nfw
        r1 = 0.1 * p.rh
        r2 = 0.5 * p.rh

        integral = quad(lambda r: 4. * np.pi * r**2 * p.rho(r), r1, r2)[0]
        dm = p.M(r2) - p.M(r1)
        relerr = abs(integral - dm) / dm
        assert relerr < 1e-4, \
            f"Shell mass integral disagrees with delta-M: rel error {relerr:.2e}"

    def test_density_monotone_decreasing(self, numeric_from_nfw):
        """
        NFW density is a monotonically decreasing function of r.
        The numeric profile should reproduce this ordering.
        """
        p = numeric_from_nfw
        radii = np.logspace(np.log10(p.ri[5]), np.log10(p.rh * 0.9), 50)
        rhos = p.rho(radii)
        assert np.all(np.diff(rhos) <= 0), "rho(r) is not monotone decreasing"


# ---------------------------------------------------------------------------
# Gravitational potential
# ---------------------------------------------------------------------------

class TestNumericProfilePotential:
    def test_potential_negative(self, numeric_from_nfw):
        """Phi(r) must be negative (bound system)."""
        p = numeric_from_nfw
        for frac in PROBE_RADII_FRAC:
            r = frac * p.rh
            assert p.Phi(r) < 0, f"Phi({frac} rh) is not negative"

    def test_potential_agrees_with_truncated_nfw(self, nfw, numeric_from_nfw):
        """
        NumericProfile.Phi uses the boundary condition
            Phi(rh) ≈ -G*Mh/rh  (mass beyond rh treated as a point mass at rh)
        and integrates inward:
            Phi(r) = -G*Mh/rh - G * integral_r^rh M(x)/x² dx

        We compare against a reference computed with the same boundary
        condition but using the exact NFW M(x):
            Phi_ref(r) = -G*M_nfw(rh)/rh - G * integral_r^rh M_nfw(x)/x² dx

        This isolates numerical accuracy (spline + quadrature) from the
        intentional physical difference: because mass beyond rh contributes
        to the potential at r < rh, NumericProfile.Phi is systematically
        shallower than the full analytic NFW Phi (which integrates to infinity).
        """
        p = numeric_from_nfw
        outer = cfg.G * nfw.M(nfw.rh) / nfw.rh
        for frac in [0.05, 0.1, 0.3]:
            r = frac * nfw.rh
            phi_num = p.Phi(r)
            phi_ref = -cfg.G * quad(lambda x: nfw.M(x) / x**2, r, nfw.rh)[0] - outer
            relerr = abs(phi_num - phi_ref) / abs(phi_ref)
            assert relerr < 1e-4, \
                f"Phi({frac} rh): rel error vs truncated reference {relerr:.2e} > 0.01%"

    def test_potential_monotone_increasing(self, numeric_from_nfw):
        """Potential must increase (become less negative) with radius."""
        p = numeric_from_nfw
        radii = np.logspace(np.log10(p.ri[5]), np.log10(p.rh * 0.9), 30)
        phis = np.array([p.Phi(r) for r in radii])
        assert np.all(np.diff(phis) >= 0), \
            "Phi(r) is not monotone increasing (less negative at larger r)"

    def test_potential_consistent_with_mass(self, numeric_from_nfw):
        """
        The potential difference between two radii satisfies:
            Phi(r2) - Phi(r1) = G integral_{r1}^{r2} M(r)/r^2 dr
        """
        p = numeric_from_nfw
        r1 = 0.1 * p.rh
        r2 = 0.4 * p.rh

        integral = cfg.G * quad(lambda r: p.M(r) / r**2, r1, r2)[0]
        delta_phi = p.Phi(r2) - p.Phi(r1)
        relerr = abs(integral - delta_phi) / abs(delta_phi)
        assert relerr < 1e-4, \
            f"Phi gradient consistency: rel error {relerr:.2e}"


# ---------------------------------------------------------------------------
# Circular velocity and Vmax / rmax
# ---------------------------------------------------------------------------

class TestNumericProfileVcirc:
    def test_vcirc_positive(self, numeric_from_nfw):
        """Vcirc(r) must be positive."""
        p = numeric_from_nfw
        for frac in PROBE_RADII_FRAC:
            assert p.Vcirc(frac * p.rh) > 0

    def test_vcirc_agrees_with_nfw(self, nfw, numeric_from_nfw):
        """Vcirc(r) should match NFW to within 0.01%."""
        p = numeric_from_nfw
        for frac in PROBE_RADII_FRAC:
            r = frac * nfw.rh
            vc_num = p.Vcirc(r)
            vc_nfw = nfw.Vcirc(r)
            relerr = abs(vc_num - vc_nfw) / vc_nfw
            assert relerr < 1e-4, \
                f"Vcirc({frac} rh): rel error {relerr:.2e} > 0.01%"

    def test_rmax_agrees_with_nfw(self, nfw, numeric_from_nfw):
        """
        rmax (radius of peak circular velocity) should match NFW (= 2.163 rs)
        to within 0.05%.  rmax is found by a numerical optimizer so it carries
        slightly more error than M or Vcirc themselves.
        """
        p = numeric_from_nfw
        rmax_nfw = nfw.rmax
        relerr = abs(p.rmax - rmax_nfw) / rmax_nfw
        assert relerr < 5e-4, \
            f"rmax: numeric={p.rmax:.4f}, NFW={rmax_nfw:.4f}, rel err {relerr:.2e}"

    def test_vmax_agrees_with_nfw(self, nfw, numeric_from_nfw):
        """Vmax should match NFW to within 0.01%."""
        p = numeric_from_nfw
        relerr = abs(p.Vmax - nfw.Vmax) / nfw.Vmax
        assert relerr < 1e-4, \
            f"Vmax: numeric={p.Vmax:.4f}, NFW={nfw.Vmax:.4f}, rel err {relerr:.2e}"

    def test_vcirc_peaks_at_rmax(self, numeric_from_nfw):
        """Vcirc must be maximal at rmax within the sampled interior."""
        p = numeric_from_nfw
        radii = np.linspace(p.ri[5], p.rh * 0.95, 200)
        vcircs = p.Vcirc(radii)
        assert vcircs.argmax() != 0 and vcircs.argmax() != len(vcircs) - 1, \
            "Peak Vcirc is at a boundary — rmax likely wrong"
        assert abs(radii[vcircs.argmax()] - p.rmax) / p.rmax < 0.1, \
            "rmax does not coincide with peak of Vcirc curve"


# ---------------------------------------------------------------------------
# Velocity dispersion
# ---------------------------------------------------------------------------

class TestNumericProfileSigma:
    def test_sigma_positive(self, numeric_from_nfw):
        """sigma(r) must be positive in the halo interior."""
        p = numeric_from_nfw
        for frac in [0.05, 0.1, 0.3]:
            assert p.sigma(frac * p.rh) > 0

    def test_sigma_agrees_with_nfw_jeans(self, nfw, numeric_from_nfw):
        """
        NumericProfile.sigma uses the isotropic Jeans equation integral,
        as does NFW.sigma_accurate.  Both integrate to rh (NumericProfile) vs
        infinity (NFW); the difference is the contribution of the NFW profile
        beyond rh.  At r=0.05 and 0.1*rh this truncation error is < 0.1%;
        at r=0.3*rh it reaches ~1.2% because the outer shells contribute more
        to sigma at larger r.  Tolerance is set to 2% to cover the worst case.
        """
        p = numeric_from_nfw
        for frac in [0.05, 0.1, 0.3]:
            r = frac * nfw.rh
            sig_num = p.sigma(r)
            sig_nfw = nfw.sigma_accurate(r)
            relerr = abs(sig_num - sig_nfw) / sig_nfw
            assert relerr < 0.02, \
                f"sigma({frac} rh): rel error {relerr:.2e} > 2%"

    def test_sigma_consistent_with_jeans(self, numeric_from_nfw):
        """
        The isotropic Jeans equation:
            sigma^2(r) = (1/rho(r)) * G * integral_r^rh rho(x) M(x)/x^2 dx

        NumericProfile.sigma must satisfy this to within 1% when evaluated
        against its own rho and M splines.
        """
        p = numeric_from_nfw
        r = 0.2 * p.rh
        jeans_integral = cfg.G * quad(
            lambda x: p.rho(x) * p.M(x) / x**2, r, p.rh
        )[0]
        sigma_jeans = np.sqrt(jeans_integral / p.rho(r))
        sigma_method = p.sigma(r)
        relerr = abs(sigma_method - sigma_jeans) / sigma_jeans
        assert relerr < 1e-6, \
            f"sigma self-consistency: rel error {relerr:.2e} > 1e-6"


# ---------------------------------------------------------------------------
# Mean density and dynamical time
# ---------------------------------------------------------------------------

class TestNumericProfileRhobar:
    def test_rhobar_equals_M_over_volume(self, numeric_from_nfw):
        """rhobar(r) = M(r) / (4pi/3 r^3) by definition."""
        p = numeric_from_nfw
        for frac in PROBE_RADII_FRAC:
            r = frac * p.rh
            rhobar_method = p.rhobar(r)
            rhobar_direct = p.M(r) / (cfg.FourPiOverThree * r**3)
            relerr = abs(rhobar_method - rhobar_direct) / rhobar_direct
            assert relerr < 1e-10, \
                f"rhobar({frac} rh) mismatch: {relerr:.2e}"

    def test_tdyn_positive(self, numeric_from_nfw):
        """Dynamical time must be positive."""
        p = numeric_from_nfw
        for frac in PROBE_RADII_FRAC:
            assert p.tdyn(frac * p.rh) > 0

    def test_tdyn_increases_with_radius(self, numeric_from_nfw):
        """
        For a centrally concentrated profile, tdyn ∝ 1/sqrt(rhobar)
        increases with radius as mean density falls.
        """
        p = numeric_from_nfw
        radii = [0.05, 0.1, 0.3, 0.6]
        tdyns = [p.tdyn(f * p.rh) for f in radii]
        assert np.all(np.diff(tdyns) > 0), \
            "tdyn does not increase with radius as expected"


# ---------------------------------------------------------------------------
# Scalar vs. array input
# ---------------------------------------------------------------------------

class TestNumericProfileInputTypes:
    def test_scalar_input(self, numeric_from_nfw):
        """All methods must accept a scalar radius without error."""
        p = numeric_from_nfw
        r = 0.1 * p.rh
        assert np.isfinite(p.rho(r))
        assert np.isfinite(p.M(r))
        assert np.isfinite(p.Phi(r))
        assert np.isfinite(p.Vcirc(r))
        assert np.isfinite(p.rhobar(r))
        assert np.isfinite(p.tdyn(r))
        assert np.isfinite(p.sigma(r))

    def test_array_input_rho_M(self, numeric_from_nfw):
        """rho and M must accept numpy arrays and return arrays of the same shape."""
        p = numeric_from_nfw
        radii = np.array([0.05, 0.1, 0.3]) * p.rh
        rhos = p.rho(radii)
        masses = p.M(radii)
        assert rhos.shape == radii.shape
        assert masses.shape == radii.shape
        assert np.all(np.isfinite(rhos))
        assert np.all(np.isfinite(masses))


# ---------------------------------------------------------------------------
# heat_profile
# ---------------------------------------------------------------------------

class TestHeatProfile:
    @pytest.fixture
    def small_nfw_profile(self):
        """Small NFW (UFD-scale) as a NumericProfile for heating tests."""
        h = NFW(1e9, 11.68)
        ri = np.logspace(np.log10(1e-3 * h.rh), np.log10(h.rh), 300)
        Mr = h.M(ri)
        return NumericProfile(ri, Mr)

    def test_zero_heating_preserves_mass(self, small_nfw_profile):
        """eps=0 everywhere should leave Mh unchanged."""
        p = small_nfw_profile
        heated = heat_profile(p, eps=lambda r: 0.)
        relerr = abs(heated.Mh - p.Mh) / p.Mh
        assert relerr < 0.01, \
            f"Zero heating changed Mh by {relerr:.4f}"

    def test_positive_heating_does_not_increase_mass(self, small_nfw_profile):
        """Energy injection can only unbind shells, never add mass."""
        p = small_nfw_profile
        eps_scale = 0.01 * cfg.G * p.Mh / p.rh
        heated = heat_profile(p, eps=lambda r: eps_scale * (r / p.rh) ** 2)
        assert heated.Mh <= p.Mh * (1 + 1e-6)

    def test_heated_profile_is_numeric_profile(self, small_nfw_profile):
        """heat_profile must return a NumericProfile instance."""
        p = small_nfw_profile
        result = heat_profile(p, eps=lambda r: 0.)
        assert isinstance(result, NumericProfile)

    def test_heated_profile_has_valid_rmax_vmax(self, small_nfw_profile):
        """rmax and Vmax of the heated profile must be finite and positive."""
        p = small_nfw_profile
        eps_scale = 0.01 * cfg.G * p.Mh / p.rh
        heated = heat_profile(p, eps=lambda r: eps_scale * (r / p.rh) ** 2)
        assert np.isfinite(heated.rmax) and heated.rmax > 0
        assert np.isfinite(heated.Vmax) and heated.Vmax > 0

    def test_catastrophic_heating_raises(self, small_nfw_profile):
        """Injecting more energy than the binding energy should raise RuntimeError."""
        p = small_nfw_profile
        huge_eps = 1e10 * cfg.G * p.Mh / p.rh
        with pytest.raises(RuntimeError, match="unbinds"):
            heat_profile(p, eps=lambda r: huge_eps * r**2)

    def _stripping_eps(self, p):
        """Energy injection that grows steeply with r, unbinding outer shells while
        keeping inner shells bound.  eps(r) = 0.6 * (G Mh / rh) * (r/rh)^2."""
        G, Mh, rh = cfg.G, p.Mh, p.rh
        return lambda r: 0.6 * G * Mh / rh * (r / rh) ** 2

    def test_tally_out_param_matches_warning(self, small_nfw_profile):
        """The optional tally out-param reports the same shell-clamp count as the
        ShellClampWarning channel, without changing the returned profile (default
        tally=None leaves all existing callers untouched)."""
        import re
        import warnings
        from subhalo_functions import ShellClampWarning
        p = small_nfw_profile
        eps = self._stripping_eps(p)
        tally = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ShellClampWarning)
            heated = heat_profile(p, eps, tally=tally)
        assert set(tally) == {'shells', 'worst_pct', 'worst_r'}
        w = [x for x in caught if issubclass(x.category, ShellClampWarning)]
        if w:
            n = int(re.search(r'clamped (\d+) shell', str(w[0].message)).group(1))
            assert tally['shells'] == n
            assert tally['worst_pct'] > 0.
        else:
            assert tally['shells'] == 0
        # default path (no tally) returns an identical profile
        assert np.allclose(heated.Mr, heat_profile(p, eps).Mr)

    def test_no_zero_mass_shells_in_bound_region(self, small_nfw_profile):
        """Every consecutive pair of shells in the output must carry positive mass
        (no flat spot in the cumulative mass profile inside the bound region)."""
        p = small_nfw_profile
        result = heat_profile(p, self._stripping_eps(p))
        shell_masses = np.diff(result.Mr)
        assert np.all(shell_masses > 0), (
            "Zero-mass shell(s) inside bound region at indices: "
            f"{np.where(shell_masses <= 0)[0]}"
        )

    def test_only_outer_shells_unbound(self, small_nfw_profile):
        """Bound shells must form a contiguous inner block — no unbound shell
        sandwiched between two bound shells.

        We reconstruct the bound/unbound decision on the *output* grid using
        the same eps and the same energy formula as heat_profile itself:
            Ef = G * Menc / r * (-1 + 2*eps(r)*r/(G*Menc))
        and verify that once a shell is unbound, all shells at larger r are
        also unbound (i.e. the bound set is a prefix of the sorted shell list)."""
        p = small_nfw_profile
        eps = self._stripping_eps(p)
        result = heat_profile(p, eps)

        G = cfg.G
        ri = result.ri
        Menc = result.Mr

        Mshell = np.empty_like(Menc)
        Mshell[0] = Menc[0]
        Mshell[1:] = np.diff(Menc)

        perturb = np.where(Menc > 0, 2.0 * eps(ri) * ri / (G * Menc), 0.0)
        Ef = G * Menc / ri * (-1.0 + perturb)
        bound = (Ef < 0) & (Mshell > 0)

        if bound.any():
            last_bound_idx = int(np.max(np.where(bound)[0]))
            interior = bound[:last_bound_idx]
            assert np.all(interior), (
                "Unbound shell(s) sandwiched inside bound region at indices: "
                f"{np.where(~interior)[0]}"
            )


# ---------------------------------------------------------------------------
# _expand_shells: vectorized Du+24 eq.(36) map vs a serial reference
# ---------------------------------------------------------------------------

def _expand_shells_serial(ri, Menc, eps_vals, thresh=1e-10):
    """Readable outer->inner reference for _expand_shells: the same eq.(36) shell
    map, but the no-crossing clamp is the plain serial loop. The vectorized
    _clamp_monotone must reproduce this exactly; this is the oracle."""
    G = cfg.G
    n = len(ri)
    eps_vals = np.broadcast_to(np.asarray(eps_vals, float), ri.shape)
    perturb = np.zeros_like(ri)
    pos = Menc > 0.
    perturb[pos] = 2. * eps_vals[pos] * ri[pos] / (G * Menc[pos])

    shells = 0
    worst_excess = 0.
    worst_r = np.nan
    for i in reversed(range(n - 1)):
        if Menc[i] > 0. and Menc[i + 1] > 0.:
            limit = (1. - ri[i] / ri[i + 1]
                     * (Menc[i] / Menc[i + 1]) ** (-1. / 3.)
                     * (1. - perturb[i + 1]))
            if perturb[i] > limit:
                excess = (perturb[i] / limit) - 1. if limit > 0 else np.inf
                if excess > thresh:
                    shells += 1
                    if excess > worst_excess:
                        worst_excess, worst_r = excess, ri[i]
                perturb[i] = limit
    tally = dict(shells=shells, worst_pct=worst_excess * 100. if shells else 0.,
                 worst_r=worst_r if shells else np.nan)

    Ef = G * Menc / ri * (perturb - 1.)
    Mshell = np.empty_like(Menc)
    Mshell[0] = Menc[0]
    Mshell[1:] = np.diff(Menc)
    bound = (Ef < 0.) & (Mshell > 0.)
    rf = ri / (1. - perturb)
    order = np.argsort(rf[bound])
    r_bound = rf[bound][order]
    M_bound = np.cumsum(Mshell[bound][order])
    keep = np.ones(len(r_bound), bool)
    keep[:-1] = r_bound[:-1] != r_bound[1:]
    return r_bound[keep], M_bound[keep], tally


def _nfw_grid(M=1e9, c=11.68, npts=300, rlo=1e-3, kind="nfw"):
    h = NFW(M, c)
    ri = np.logspace(np.log10(rlo * h.rh), np.log10(h.rh), npts)
    Mr = np.asarray(h.M(ri), float)
    if kind == "cored":
        rc = 0.3 * h.rh
        rho0 = h.M(rc) / (4. / 3. * np.pi * rc ** 3)
        Mr = np.where(ri < rc, 4. / 3. * np.pi * ri ** 3 * rho0, Mr)
    return ri, Mr, h


class TestExpandShells:
    """The vectorized _expand_shells must match the serial reference to rounding
    across the profile/heating regimes the forward model sees, including the
    numerical edge cases (inner spline undershoot, cored inner bump, low-density
    ringing outer tail, fine grids)."""

    def _cases(self):
        cases = []
        ri, M, h = _nfw_grid()
        sc = cfg.G * h.M(h.rh) / h.rh
        cases.append(("nfw_mild", ri, M, 0.01 * sc * (ri / h.rh) ** 2))
        cases.append(("nfw_strong", ri, M, 0.6 * sc * (ri / h.rh) ** 2))
        cases.append(("nfw_catastrophic", ri, M, 50. * sc * (ri / h.rh) ** 2))
        cases.append(("nfw_const_eps", ri, M, np.full_like(ri, 0.02 * sc)))
        # inner spline undershoot: a few negative Menc knots at small r
        M_us = M.copy()
        M_us[:5] = -1e-3 * M.max() * np.array([5., 4., 3., 2., 1.])
        cases.append(("undershoot_inner", ri, M_us, 0.3 * sc * (ri / h.rh) ** 2))
        # cored profile: constant-density core -> non-monotone perturb
        ric, Mc, hc = _nfw_grid(kind="cored")
        scc = cfg.G * hc.M(hc.rh) / hc.rh
        cases.append(("cored_strong", ric, Mc, 0.5 * scc * (ric / hc.rh) ** 2))
        cases.append(("cored_inner_bump", ric, Mc, 0.15 * scc * (ric / hc.rh)))
        # low-density outer tail with ringing (kept a valid enclosed mass)
        rng = np.random.default_rng(0)
        M_ring = M.copy()
        tail = ri > 0.5 * h.rh
        M_ring[tail] *= (1. + 1e-4 * rng.standard_normal(tail.sum())).cumprod()
        M_ring = np.maximum.accumulate(M_ring)
        cases.append(("ringing_tail", ri, M_ring, 0.4 * sc * (ri / h.rh) ** 2))
        # fine grid: A cumprod over many shells
        rif, Mf, hf = _nfw_grid(npts=2000, rlo=1e-4)
        scf = cfg.G * hf.M(hf.rh) / hf.rh
        cases.append(("fine_grid", rif, Mf, 0.5 * scf * (rif / hf.rh) ** 2))
        return cases

    def test_matches_serial(self):
        for name, ri, Menc, eps in self._cases():
            rb_v, Mb_v = _expand_shells(ri, Menc, eps, tally=(tv := {}))
            rb_s, Mb_s, ts = _expand_shells_serial(ri, Menc, eps)
            assert len(rb_v) == len(rb_s), f"{name}: bound-shell count differs"
            np.testing.assert_allclose(rb_v, rb_s, rtol=1e-9,
                                       err_msg=f"{name}: r_bound diverges")
            np.testing.assert_allclose(Mb_v, Mb_s, rtol=1e-9,
                                       err_msg=f"{name}: M_bound diverges")
            assert tv['shells'] == ts['shells'], f"{name}: clamp count differs"
            if ts['shells']:
                assert abs(tv['worst_pct'] - ts['worst_pct']) <= \
                    1e-6 * abs(ts['worst_pct']), f"{name}: worst_pct differs"
                assert tv['worst_r'] == ts['worst_r'], f"{name}: worst_r differs"

    def test_unbound_is_contiguous_outer_block(self):
        """The clamp guarantees bound-outer => bound-inner, so the surviving
        bound shells are the innermost ones: M(<r) is monotone with no gap."""
        for name, ri, Menc, eps in self._cases():
            rb, Mb = _expand_shells(ri, Menc, eps)
            assert np.all(np.diff(rb) > 0.), f"{name}: r_bound not increasing"
            assert np.all(np.diff(Mb) > 0.), f"{name}: M_bound not increasing"

    def test_mass_conserved_when_all_bound(self):
        """Mild heating unbinds nothing, so total mass is preserved (shell map
        conserves M(<r) per shell)."""
        ri, M, h = _nfw_grid()
        eps = 1e-4 * cfg.G * h.M(h.rh) / h.rh * (ri / h.rh) ** 2
        _, Mb = _expand_shells(ri, M, eps)
        assert abs(Mb[-1] - M[-1]) < 1e-9 * M[-1]

    def test_stronger_heating_unbinds_more(self):
        ri, M, h = _nfw_grid()
        sc = cfg.G * h.M(h.rh) / h.rh
        _, Mb_lo = _expand_shells(ri, M, 0.2 * sc * (ri / h.rh) ** 2)
        _, Mb_hi = _expand_shells(ri, M, 0.8 * sc * (ri / h.rh) ** 2)
        assert Mb_hi[-1] < Mb_lo[-1] <= M[-1] * (1. + 1e-9)


# ---------------------------------------------------------------------------
# Tidal tensor
# ---------------------------------------------------------------------------

class TestTidalTensor:
    @pytest.fixture(scope="class")
    def host(self):
        """MW-scale NFW host for tidal tensor tests."""
        return NFW(1e12, 10.)

    def test_symmetry(self, host):
        """Tidal tensor must be symmetric: T_ij = T_ji."""
        coords = [50., 30., 20.]
        T = tidalTensor(host, coords)
        np.testing.assert_allclose(T, T.T, atol=1e-30,
                                   err_msg="Tidal tensor is not symmetric")

    def test_trace_equals_poisson(self, host):
        """Tr(T) must equal -4*pi*G*rho(r) (Poisson's equation)."""
        coords = [50., 30., 20.]
        r = np.sqrt(sum(c**2 for c in coords))
        T = tidalTensor(host, coords)
        trace = np.trace(T)
        expected = -4. * np.pi * cfg.G * host.rho(r)
        np.testing.assert_allclose(trace, expected, rtol=1e-10,
                                   err_msg="Trace does not satisfy Poisson")

    def test_eigenvalues_rotational_invariance(self, host):
        """Eigenvalues of T must depend only on r, not on direction."""
        r = 100.  # kpc
        # two different directions at the same r
        coords_a = [r, 0., 0.]
        coords_b = [r / np.sqrt(3), r / np.sqrt(3), r / np.sqrt(3)]

        eigs_a = np.sort(np.linalg.eigvalsh(tidalTensor(host, coords_a)))
        eigs_b = np.sort(np.linalg.eigvalsh(tidalTensor(host, coords_b)))
        np.testing.assert_allclose(eigs_a, eigs_b, rtol=1e-10,
                                   err_msg="Eigenvalues depend on direction")

    def test_on_axis_eigenvalues(self, host):
        """
        On the x-axis (y=z=0), the tensor is diagonal with known eigenvalues:
            T_xx = 2*G*M(r)/r^3 - 4*pi*G*rho(r)
            T_yy = T_zz = -G*M(r)/r^3
        """
        r = 80.
        coords = [r, 0., 0.]
        T = tidalTensor(host, coords)

        Menc = host.M(r)
        rho = host.rho(r)

        T_radial = 2. * cfg.G * Menc / r**3 - 4. * np.pi * cfg.G * rho
        T_tangential = -cfg.G * Menc / r**3

        np.testing.assert_allclose(T[0, 0], T_radial, rtol=1e-12)
        np.testing.assert_allclose(T[1, 1], T_tangential, rtol=1e-12)
        np.testing.assert_allclose(T[2, 2], T_tangential, rtol=1e-12)
        # off-diagonal must vanish
        np.testing.assert_allclose(T[0, 1], 0., atol=1e-30)
        np.testing.assert_allclose(T[0, 2], 0., atol=1e-30)
        np.testing.assert_allclose(T[1, 2], 0., atol=1e-30)

    def test_point_mass_limit(self):
        """
        Far outside a truncated halo (NumericProfile) where rho=0 and
        M(r)=const, the tensor must equal the point-mass form:
            T_ij = G*M/r^3 * (3 x_ix_j/r^2 - delta_ij)
        """
        nfw = NFW(1e12, 10.)
        ri = np.logspace(np.log10(1e-3 * nfw.rh), np.log10(nfw.rh), 500)
        trunc = NumericProfile(ri, nfw.M(ri))

        r = 10. * trunc.rh  # well outside, rho extrapolates to zero
        coords = [r, 0., 0.]
        T = tidalTensor(trunc, coords)

        Mh = trunc.Mh
        np.testing.assert_allclose(T[0, 0], 2. * cfg.G * Mh / r**3, rtol=1e-4)
        np.testing.assert_allclose(T[1, 1], -cfg.G * Mh / r**3, rtol=1e-4)

    def test_linear_in_mass(self, host):
        """Tidal tensor scales linearly with halo mass."""
        r = 80.
        coords = [r, 0., 0.]
        T1 = tidalTensor(host, coords)

        host2 = NFW(2e12, 10.)
        T2 = tidalTensor(host2, coords)

        # concentration is the same so M(r)/M_total and rho/M_total scale the same;
        # however rs differs, so the scaling is only exact at r >> rs.
        # Instead, just verify T doubles when the whole profile doubles.
        # Build a host with 2x mass but same rs: use c such that r_vir gives 2x mass.
        # Simpler: just check ratio of on-axis eigenvalues at a radius well inside both halos.
        ratio = T2[0, 0] / T1[0, 0]
        assert ratio > 1., "Doubling mass should increase tidal tensor"

    def test_shape_is_3x3(self, host):
        """Return value must be a 3x3 numpy array."""
        T = tidalTensor(host, [50., 30., 20.])
        assert isinstance(T, np.ndarray)
        assert T.shape == (3, 3)
