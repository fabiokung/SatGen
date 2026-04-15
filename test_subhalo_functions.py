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
from subhalo_functions import NumericProfile, heat_profile


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
