"""
Unit and smoke tests for evolve_heating() in stripping_common.py
and for NumericProfile / SIS in subhalo_functions.py.

Run with:
    python -m pytest test_stripping.py -v
"""

import numpy as np
import pytest

import config as cfg
from profiles import NFW, Vcirc as standalone_Vcirc, tdyn as standalone_tdyn
import stripping_common as sc
from stripping_common import EvolutionResult
from subhalo_functions import NumericProfile, SIS, heat_profile, _log_pchip


# ---------------------------------------------------------------------------
# Module-level fixtures (config init is slow; do it once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def host():
    cfg.Mres = 1e3
    return NFW(3.7e12, 11.68)


@pytest.fixture(scope='module')
def xv0_eccentric(host):
    R0 = 0.7 * host.rh
    xv, _ = sc.make_orbit(host, R0=R0, z0=0., eta=0.21)
    return xv


@pytest.fixture(scope='module')
def xv0_circular(host):
    R0 = 0.7 * host.rh
    xv, _ = sc.make_orbit(host, R0=R0, z0=0., eta=1.0)
    return xv


@pytest.fixture(scope='module')
def p0(host):
    sat_nfw = NFW(1e6, 9.5)
    rvals = np.logspace(np.log10(cfg.Rres), np.log10(sat_nfw.rh), 100)
    return NumericProfile(rvals, sat_nfw.M(rvals))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _finite_positive(arr):
    return np.all(np.isfinite(arr)) and np.all(arr >= 0.)


# ---------------------------------------------------------------------------
# Smoke: first-order only
# ---------------------------------------------------------------------------

class TestFirstOrder:
    def test_smoke(self, host, p0, xv0_eccentric):
        """run completes, returns EvolutionResult with sane arrays"""
        res = sc.evolve_heating(host, p0, xv0_eccentric,
                                tmax=5., Nstep=200, second_order=False)
        assert isinstance(res, EvolutionResult)
        assert res.rmax0 > 0.
        assert res.vmax0 > 0.
        assert res.m[-1] > 0.
        assert _finite_positive(res.m)
        assert _finite_positive(res.r)
        assert _finite_positive(res.rmax)
        assert _finite_positive(res.vmax)
        assert np.all(np.isfinite(res.lt))

    def test_deterministic(self, host, p0, xv0_eccentric):
        """two identical runs must produce bit-identical output"""
        r1 = sc.evolve_heating(host, p0, xv0_eccentric,
                               tmax=5., Nstep=200, second_order=False)
        r2 = sc.evolve_heating(host, p0, xv0_eccentric,
                               tmax=5., Nstep=200, second_order=False)
        np.testing.assert_array_equal(r1.m, r2.m)
        np.testing.assert_array_equal(r1.rmax, r2.rmax)

    def test_explicit_false_kwarg(self, host, p0, xv0_eccentric):
        """explicit second_order=False label kwarg must not crash"""
        res = sc.evolve_heating(host, p0, xv0_eccentric,
                                tmax=5., Nstep=200, second_order=False,
                                label='test-first-order')
        assert res.label == 'test-first-order'


# ---------------------------------------------------------------------------
# Smoke: second-order
# ---------------------------------------------------------------------------

class TestSecondOrder:
    def test_smoke(self, host, p0, xv0_eccentric):
        """second_order=True completes with sane arrays"""
        res = sc.evolve_heating(host, p0, xv0_eccentric,
                                tmax=5., Nstep=200, second_order=True)
        assert isinstance(res, EvolutionResult)
        assert res.m[-1] > 0.
        assert _finite_positive(res.m)
        assert _finite_positive(res.rmax)
        assert np.all(np.isfinite(res.lt))

    def test_more_aggressive_than_first_order(self, host, p0, xv0_eccentric):
        """second-order heating should strip at least as much as first-order"""
        r1 = sc.evolve_heating(host, p0, xv0_eccentric,
                               tmax=5., Nstep=200, second_order=False)
        r2 = sc.evolve_heating(host, p0, xv0_eccentric,
                               tmax=5., Nstep=200, second_order=True)
        # final bound mass: second-order <= first-order (or very close)
        assert r2.m[-1] <= r1.m[-1] * 1.01


# ---------------------------------------------------------------------------
# Circular orbit
# ---------------------------------------------------------------------------

class TestCircularOrbit:
    def test_first_order_circular(self, host, p0, xv0_circular):
        res = sc.evolve_heating(host, p0, xv0_circular,
                                tmax=5., Nstep=200, second_order=False)
        assert res.m[-1] > 0.
        assert _finite_positive(res.m)

    def test_second_order_circular(self, host, p0, xv0_circular):
        res = sc.evolve_heating(host, p0, xv0_circular,
                                tmax=5., Nstep=200, second_order=True)
        assert res.m[-1] > 0.
        assert _finite_positive(res.m)


# ---------------------------------------------------------------------------
# Smoothness: no catastrophic jumps in rmax track
# ---------------------------------------------------------------------------

class TestSmoothness:
    def test_rmax_track_smooth_between_kicks(self, host, p0, xv0_eccentric):
        """Between pericentre kicks the track should be flat. The 95th-percentile
        log-step is a robust check that tolerates the (physical) big drops at
        each peri shock without being fooled by per-step noise.
        """
        res = sc.evolve_heating(host, p0, xv0_eccentric,
                                tmax=50., Nstep=2000, second_order=True)
        rmax0 = res.rmax0
        track = res.rmax / rmax0
        mask = track > 0.
        if mask.sum() < 2:
            pytest.skip("too few nonzero rmax points to check smoothness")
        log_track = np.log(track[mask])
        jumps = np.abs(np.diff(log_track))
        p95 = np.percentile(jumps, 95)
        assert p95 < 0.5, f"95th-pct rmax log-step too large: {p95:.3f}"


# ---------------------------------------------------------------------------
# Timestep constraint
# ---------------------------------------------------------------------------

class TestTimestep:
    def test_timestep_too_large(self, host, p0, xv0_eccentric):
        """too few steps → dt > 0.5*t_dyn at pericentre → ValueError"""
        with pytest.raises(ValueError, match="dt/t_dyn"):
            sc.evolve_heating(host, p0, xv0_eccentric,
                              tmax=10., Nstep=50, second_order=False)

    def test_timestep_ok(self, host, p0, xv0_eccentric):
        """adequate Nstep must not raise"""
        res = sc.evolve_heating(host, p0, xv0_eccentric,
                                tmax=5., Nstep=200, second_order=False)
        assert res.m[-1] > 0.


# ---------------------------------------------------------------------------
# SIS profile unit tests
# ---------------------------------------------------------------------------

class TestSIS:
    @pytest.fixture(scope='class')
    def h(self):
        cfg.Mres = 1e3
        return SIS(3.7e12)

    def test_flat_rotation_curve(self, h):
        """Vcirc should be constant inside rh."""
        radii = np.logspace(0., np.log10(h.rh * 0.99), 20)
        vcs = np.array([float(h.Vcirc(r)) for r in radii])
        assert np.allclose(vcs, h.Vc, rtol=1e-10)

    def test_vcirc_drops_outside(self, h):
        """Vcirc should fall below Vc beyond the virial radius."""
        assert h.Vcirc(2. * h.rh) < h.Vc

    def test_density_slope(self, h):
        """rho(2r)/rho(r) should equal 1/4 (rho proportional to r^{-2})."""
        for r in [10., 50., 100.]:
            assert abs(h.rho(2.*r) / h.rho(r) - 0.25) < 1e-10

    def test_density_zero_outside(self, h):
        assert h.rho(h.rh * 1.01) == 0.

    def test_mass_linear(self, h):
        """M(<2r)/M(<r) should equal 2 well inside rh (M proportional to r)."""
        for r in [10., 50., 100.]:
            assert abs(h.M(2.*r) / h.M(r) - 2.) < 1e-10

    def test_mass_capped_at_mh(self, h):
        assert h.M(2. * h.rh) == h.Mh

    def test_phi_at_rh(self, h):
        """Phi(rh) should equal 2*sigma_v2*(0 - 1) = -Vc^2."""
        assert abs(h.Phi(h.rh) - (-h.Vc2)) < 1e-8 * abs(h.Phi(h.rh))

    def test_fgrav_matches_flat_curve(self, h):
        """Radial force inside rh: |fgrav_R| * r = Vc^2."""
        for r in [20., 100., 200.]:
            fR, _, fz = h.fgrav(r, 0.)
            assert abs(abs(fR) * r - h.Vc2) < 1e-10 * h.Vc2

    def test_sigma_at_center(self, h):
        """sigma(0+) = sigma_v (Jeans solution for SIS)."""
        assert abs(h.sigma(1e-3) - np.sqrt(h.sigma_v2)) / np.sqrt(h.sigma_v2) < 1e-4

    def test_sigma_at_rh(self, h):
        """sigma(rh) = 0 (no pressure support at the truncation edge)."""
        assert h.sigma(h.rh) == 0.

    def test_standalone_vcirc(self, h):
        """The profiles.py standalone Vcirc must agree with the method."""
        for r in [50., 150., 280.]:
            # standalone uses finite differencing (eps=0.001), so ~0.1% error expected
            assert abs(float(standalone_Vcirc(h, r)) / float(h.Vcirc(r)) - 1.) < 1e-3

    def test_standalone_tdyn(self, h):
        """The profiles.py standalone tdyn must return a positive finite value."""
        t = float(standalone_tdyn(h, 100.))
        assert np.isfinite(t) and t > 0.


# ---------------------------------------------------------------------------
# NumericProfile: spline and density invariants
# ---------------------------------------------------------------------------

class TestNumericProfileSplines:
    """rho and rhovals must stay non-negative; cubic spline can overshoot
    between knots so we clamp on query."""
    @pytest.fixture(scope='class')
    def nfw_prof(self):
        cfg.Mres = 1e1
        sat = NFW(3.89e6, 14.75)
        rvals = np.logspace(np.log10(cfg.Rres), np.log10(10. * sat.rs), 100)
        return NumericProfile(rvals, sat.M(rvals))

    def test_rho_nonneg_on_fine_grid(self, nfw_prof):
        rr = np.logspace(np.log10(nfw_prof.ri[0]), np.log10(nfw_prof.rh), 500)
        rho = nfw_prof.rho(rr)
        assert (rho >= 0.).all(), f"got {(rho < 0.).sum()} negative rho values"

    def test_rhovals_nonneg(self, nfw_prof):
        assert (nfw_prof.rhovals >= 0.).all()

    def test_heat_profile_conserves_Mh_at_zero_eps(self):
        """Mass conservation in heat_profile under zero energy injection.
        The cubic spline of M(r) inside NumericProfile can overshoot
        max(Mr) just inside the outer boundary on profiles with a steep
        ramp + plateau (typical post heat/strip shape). heat_profile
        samples the spline on its own grid that extends past rh, so the
        sampled Menc can exceed input Mh; without capping, those
        overshot values leak bound mass into the new profile each call.
        Per-call +0.01% compounds to many-fold growth over tens of
        thousands of timesteps."""
        cfg.Mres = 1e1
        ri = np.logspace(-3, np.log10(0.065), 80)
        Mh = 15096.
        x = ri / 0.04
        Mr = Mh * (1. - 1. / (1. + 5. * x**2.5))
        Mr[-3:] = Mh
        Mr = np.maximum.accumulate(Mr)
        prof = NumericProfile(ri, Mr)
        def eps_zero(r): return 0. * r**2
        heated = heat_profile(prof, eps_zero)
        assert heated.Mh <= prof.Mh * (1.0 + 1e-8), (
            f"heat_profile with eps=0 inflated Mh: input {prof.Mh:.6e}, "
            f"output {heated.Mh:.6e} (ratio {heated.Mh/prof.Mh:.8f})"
        )

    def test_second_order_no_constant_M_inner_oscillation(self):
        """Late-stage 2nd-order snapshots must show smooth, monotonically
        decreasing rho(r) at small r. If the strip rebuild samples below
        newProfile.ri[0], the spline's ext='const' creates a constant-M
        plateau in the new profile and the cubic-spline derivative there
        produces alternating zero / spurious-large rho values."""
        cfg.Mres = 1e1
        from scipy.optimize import brentq
        M_in_rmax = 1e6
        rs_target = 0.22
        mu_at_rmax = np.log(1. + 2.163) - 2.163 / (1. + 2.163)
        def _err(c):
            mu_c = np.log(1. + c) - c / (1. + c)
            return NFW(M_in_rmax * mu_c / mu_at_rmax, c).rs - rs_target
        cNFW = brentq(_err, 1., 100.)
        mu_c = np.log(1. + cNFW) - cNFW / (1. + cNFW)
        sat = NFW(M_in_rmax * mu_c / mu_at_rmax, cNFW)
        rvals = np.logspace(np.log10(cfg.Rres), np.log10(10. * sat.rs), 100)

        hSIS = SIS(3.7e12)
        eta = (50. / 1.022) / hSIS.Vc
        xv0, _ = sc.make_orbit(hSIS, R0=200., z0=0., eta=eta)
        res = sc.evolve_heating(
            hSIS, NumericProfile(rvals, sat.M(rvals)), xv0,
            tmax=30., Nstep=10000, epsh=3., gamma=1.5,
            second_order=True, n_snapshots=10,
        )
        # check inner-region monotonicity in late snapshots (where the
        # plateau-driven oscillations were worst)
        for i in range(res.rho_snapshots.shape[0] // 2, res.rho_snapshots.shape[0]):
            rho = res.rho_snapshots[i]
            if rho.max() == 0.:
                continue
            inner = rho[:30]
            log_rho = np.log10(np.maximum(inner, 1e-30))
            dlog = np.diff(log_rho)
            sign_changes = int(((dlog[:-1] * dlog[1:]) < 0).sum())
            assert sign_changes < 3, (
                f"snap {i}: {sign_changes} sign changes in d(log rho)/d(log r) "
                f"on inner 30 points (oscillating density)"
            )

    def test_second_order_evolution_no_density_spike(self):
        """During second-order tidal heating, snapshot density profiles
        must not have spurious peaks at the outer edge. Without the PCHIP
        rebin in heat_profile, the cubic spline in NumericProfile rings on
        the irregular shell positions returned after expansion and the
        outer rho ramps back up by orders of magnitude."""
        cfg.Mres = 1e1
        from scipy.optimize import brentq
        M_in_rmax = 1e6
        rs_target = 0.22
        mu_at_rmax = np.log(1. + 2.163) - 2.163 / (1. + 2.163)
        def _err(c):
            mu_c = np.log(1. + c) - c / (1. + c)
            return NFW(M_in_rmax * mu_c / mu_at_rmax, c).rs - rs_target
        cNFW = brentq(_err, 1., 100.)
        mu_c = np.log(1. + cNFW) - cNFW / (1. + cNFW)
        sat = NFW(M_in_rmax * mu_c / mu_at_rmax, cNFW)
        rvals = np.logspace(np.log10(cfg.Rres), np.log10(10. * sat.rs), 100)

        hSIS = SIS(3.7e12)
        eta = (50. / 1.022) / hSIS.Vc
        xv0, _ = sc.make_orbit(hSIS, R0=200., z0=0., eta=eta)

        res = sc.evolve_heating(
            hSIS, NumericProfile(rvals, sat.M(rvals)), xv0,
            tmax=10., Nstep=5000, epsh=3., gamma=1.5,
            second_order=True, n_snapshots=10,
        )

        for i in range(res.rho_snapshots.shape[0]):
            rho = res.rho_snapshots[i]
            if rho.max() == 0.:
                continue
            n_outer = max(len(rho) // 4, 5)
            outer = rho[-n_outer:]
            if outer[0] <= 0.:
                continue
            ratio = outer.max() / outer[0]
            assert ratio < 5., (
                f"snap {i}: outer rho rises by {ratio:.2e}x "
                f"(max={outer.max():.3e}, first={outer[0]:.3e})"
            )


# ---------------------------------------------------------------------------
# NumericProfile: vmax/rmax location
# ---------------------------------------------------------------------------

class TestNumericProfileVmaxRmax:
    @pytest.fixture(scope='class')
    def nfw_prof(self):
        cfg.Mres = 1e1
        sat = NFW(3.89e6, 14.75)
        rvals = np.logspace(np.log10(cfg.Rres), np.log10(10. * sat.rs), 100)
        return NumericProfile(rvals, sat.M(rvals))

    def test_rmax_in_interior(self, nfw_prof):
        """rmax must lie strictly inside (ri[0], rh) for an NFW profile."""
        assert nfw_prof.rmax > 1.5 * nfw_prof.ri[0]
        assert nfw_prof.rmax < 0.999 * nfw_prof.rh

    def test_rmax_matches_NFW_analytic(self, nfw_prof):
        """For a fresh NFW profile r_max = 2.163 r_s."""
        rs = 0.22
        assert abs(nfw_prof.rmax / rs - 2.163) < 0.05

    def test_omega_p_matches_M_at_r_half(self, nfw_prof):
        """omega_p = sqrt(G * M(<r_half) / r_half^3); must hold even when
        r_half is clamped to a grid boundary."""
        M_at_rhalf = float(nfw_prof.MInt(nfw_prof.r_half))
        expected = np.sqrt(cfg.G * M_at_rhalf / nfw_prof.r_half**3)
        assert abs(nfw_prof.omega_p / expected - 1.) < 1e-6

    def test_omega_p_equals_Vc_over_r_at_half(self, nfw_prof):
        Vc_at_rhalf = float(nfw_prof.Vcirc(nfw_prof.r_half))
        assert abs(nfw_prof.omega_p / (Vc_at_rhalf / nfw_prof.r_half) - 1.) < 1e-6

    def test_rmax_nan_when_vc_monotone(self):
        """For a profile with rho proportional to r^-2.5, M proportional to r^0.5
        and Vcirc = sqrt(GM/r) decreases monotonically — no interior peak.
        rmax must be NaN so the tidal-track mask drops it (otherwise argmax
        pegs rmax at the inner grid floor)."""
        rvals = np.logspace(-3, -1, 100)
        Mr = rvals**0.5
        p = NumericProfile(rvals, Mr)
        assert np.isnan(p.rmax)
        assert p.Vmax > 0.

    def test_rmax_nan_excluded_by_track_mask(self):
        rvals = np.logspace(-3, -1, 100)
        p = NumericProfile(rvals, rvals**0.5)
        rmax_arr = np.array([0.5, p.rmax, 0.3])
        vmax_arr = np.array([1.0, p.Vmax, 0.8])
        mask = (rmax_arr > 0.) & (vmax_arr > 0.)
        assert mask.sum() == 2
        assert not mask[1]


# ---------------------------------------------------------------------------
# Snapshot grid
# ---------------------------------------------------------------------------

class TestSnapshotGrid:
    def test_r_grid_per_snapshot(self, host, p0, xv0_eccentric):
        """r_grid is (n_snapshots, n_radii) and each row covers the bound
        profile at that snapshot."""
        res = sc.evolve_heating(host, p0, xv0_eccentric,
                                tmax=5., Nstep=200, second_order=False,
                                n_snapshots=10)
        assert res.r_grid.ndim == 2
        assert res.r_grid.shape == (10, 100)
        snap0_outer = res.r_grid[0, -1]
        for i in range(1, res.r_grid.shape[0]):
            if res.M_snapshots[i].max() == 0.:
                continue
            assert res.r_grid[i, -1] <= snap0_outer * 5.

    def test_M_snapshots_no_flat_regions(self, host, p0, xv0_eccentric):
        """M(<r) on each snapshot grid must not contain large flat segments
        (which would indicate boundary extrapolation)."""
        res = sc.evolve_heating(host, p0, xv0_eccentric,
                                tmax=5., Nstep=200, second_order=True,
                                n_snapshots=10)
        for i in range(res.M_snapshots.shape[0]):
            M = res.M_snapshots[i]
            if M.max() == 0.:
                continue
            n_flat = (np.diff(M) == 0.).sum()
            assert n_flat < 5, f"snap {i}: {n_flat} exactly-flat segments"


# ---------------------------------------------------------------------------
# Interpolator edge cases (PCHIP everywhere; FD-derived rho)
# ---------------------------------------------------------------------------

class TestInterpolatorEdgeCases:
    """Pin invariants the cubic-spline pipeline used to violate. PCHIP is
    monotone-preserving on monotone input and stays in [Mr.min(), Mr.max()];
    FD on the knots is exact on plateaus and consistent at the boundaries.
    Each test below corresponds to a real artefact that broke a notebook
    plot or leaked mass during long runs."""

    def test_M_no_overshoot_inside(self):
        """M(r) must never exceed Mh anywhere inside [ri[0], ri[-1]]. The
        old InterpolatedUnivariateSpline overshot ~0.01% above Mh just
        inside rh on a steep ramp + plateau, which leaked mass into
        heat_profile and compounded over many steps."""
        ri = np.logspace(-3, np.log10(0.065), 80)
        Mh = 1.0
        x = ri / 0.04
        Mr = Mh * (1. - 1. / (1. + 5. * x**2.5))
        Mr[-3:] = Mh
        Mr = np.maximum.accumulate(Mr)
        p = NumericProfile(ri, Mr)
        rr = np.linspace(p.ri[0], p.ri[-1], 5000)
        assert p.M(rr).max() <= p.Mh + 1e-12

    def test_M_clamp_on_extrapolation(self):
        """M(r) must return Mh for r > rh (not NaN from extrapolate=False).
        Downstream callers like ev.ltidal hand in radii that can sit just
        past rh due to float-roundoff; NaN there silently propagates into
        the tidal-radius solver."""
        ri = np.logspace(-3, np.log10(0.1), 50)
        Mr = ri**0.5
        p = NumericProfile(ri, Mr)
        m_outside = float(p.M(p.rh * 2.))
        assert np.isfinite(m_outside)
        assert abs(m_outside - p.Mh) < 1e-12
        m_inside_floor = float(p.M(p.ri[0] * 0.1))
        assert np.isfinite(m_inside_floor)

    def test_rho_zero_on_extrapolation(self):
        """rho(r) outside [ri[0], ri[-1]] must be 0, not NaN. PCHIP with
        extrapolate=False returns NaN; the override in NumericProfile.rho
        translates that to 0 so np.maximum and np.where downstream work."""
        ri = np.logspace(-3, np.log10(0.1), 50)
        Mr = ri**0.5
        p = NumericProfile(ri, Mr)
        assert float(p.rho(p.rh * 2.)) == 0.
        assert not np.isnan(float(p.rho(p.rh * 2.)))

    def test_rho_zero_on_M_plateau(self):
        """A flat M(r) region produces rho = 0 there (FD is exact on
        plateaus). Cubic-spline derivative used to ring through near-zero
        and small-positive values across consecutive knots, drawing
        zigzag artefacts on log-rho plots."""
        ri = np.logspace(-3, -1, 100)
        # interior plateau between r=2e-3 and r=5e-3
        Mr = np.where(ri < 2e-3, ri,
                      np.where(ri < 5e-3, 2e-3, ri - 3e-3))
        Mr = np.maximum.accumulate(Mr)  # ensure non-decreasing
        p = NumericProfile(ri, Mr)
        plateau_mask = (ri > 2.2e-3) & (ri < 4.5e-3)
        rho_plateau = p.rho(ri[plateau_mask])
        # outside the plateau, rho is order 1/(4 pi r^2) ~ 1e3-1e6 in these units;
        # demand at least 12 decades suppression in the plateau
        rho_outside = float(p.rho(1.5e-3))
        assert rho_plateau.max() < rho_outside * 1e-12, (
            f"plateau region rho not suppressed: max={rho_plateau.max():.3e}, "
            f"outside={rho_outside:.3e}"
        )

    def test_rho_nonneg_dense_grid(self):
        """rho must stay non-negative on a fine query grid. Monotonic
        PCHIP gives non-negative rhovals, and PCHIP query interpolation
        between non-negative knots can't dip below zero."""
        ri = np.logspace(-3, np.log10(2.), 100)
        sat = NFW(1e6, 10.)
        Mr = sat.M(ri)
        p = NumericProfile(ri, Mr)
        rr = np.logspace(np.log10(p.ri[0]), np.log10(p.rh), 5000)
        rho = p.rho(rr)
        assert np.all(rho >= 0.)
        assert np.all(np.isfinite(rho))

    def test_heat_profile_does_not_overshoot_input_Mh(self):
        """heat_profile output Mh must be <= input Mh under any heating.
        PCHIP doesn't overshoot, so the old np.minimum cap and
        np.maximum.accumulate guards are unnecessary for correctness — a
        regression here means an interpolator was switched back to a
        non-monotone scheme."""
        ri = np.logspace(-3, np.log10(0.065), 80)
        Mh = 15096.
        x = ri / 0.04
        Mr = Mh * (1. - 1. / (1. + 5. * x**2.5))
        Mr[-3:] = Mh
        Mr = np.maximum.accumulate(Mr)
        p = NumericProfile(ri, Mr)
        # mild heating: positive eps that doesn't unbind the halo
        def eps(r): return 0.01 * r**2
        h = heat_profile(p, eps)
        assert h.Mh <= p.Mh + 1e-9

    def test_heat_profile_Mr_strictly_monotone(self):
        """heat_profile output Mr must be strictly non-decreasing. A
        non-monotonic Mr would break ltidal's bisect (no sign change in a
        monotonic bracket) and cause spurious negative rho via FD."""
        ri = np.logspace(-3, np.log10(0.5), 100)
        sat = NFW(1e6, 10.)
        p = NumericProfile(ri, sat.M(ri))
        def eps(r): return 0.5 * r**2
        h = heat_profile(p, eps)
        diffs = np.diff(h.Mr)
        assert np.all(diffs >= -1e-9), (
            f"non-monotonic Mr: min diff = {diffs.min():.3e}"
        )

    def test_rmax_picks_interior_peak_not_outer_plateau(self):
        """A profile with an interior cusp + an outer M-plateau (mass all
        below the plateau) must report rmax at the cusp, not at rh. The
        old argmax-of-Vc finder pegged at the boundary when Vc was nearly
        flat across the post-bulk plateau, drawing diagonal jumps in the
        (rmax, Vmax) track."""
        ri = np.logspace(-3, -1, 200)
        sat = NFW(1e6, 10.)
        Mr = sat.M(ri)
        # extend a flat plateau out to 10x rh by appending knots at constant Mh
        ri_ext = np.concatenate([ri, np.logspace(-1, 0, 50)[1:]])
        Mr_ext = np.concatenate([Mr, np.full(49, Mr[-1])])
        p = NumericProfile(ri_ext, Mr_ext)
        # rmax should sit at the NFW cusp, near r_s * 2.163, not anywhere
        # in the post-bulk plateau
        assert np.isfinite(p.rmax)
        assert p.rmax < 0.5 * ri_ext[-1]
        assert p.rmax > ri[0]


# ---------------------------------------------------------------------------
# log-log PCHIP behaviour (cusp accuracy + boundary contracts)
# ---------------------------------------------------------------------------

class TestLogPCHIP:
    """Power laws are straight lines in log-log, so log-PCHIP is exact for
    them between knots. Linear-axis PCHIP carries O(h^2) inter-knot error
    on the same data. These tests pin that distinction down on the
    quantities we actually care about (M cusps, rho cusps), the boundary
    contract for queries outside the knot range, and the zero-tail
    handling rho needs after stripping."""

    def test_M_power_law_exactness(self):
        """M = r^2 on 20 log-knots: log-PCHIP rel error < 1e-12 at log
        midpoints; linear PCHIP > 1e-3 (regression detector that flips if
        someone reverts the wrap)."""
        from scipy.interpolate import PchipInterpolator
        ri = np.logspace(-3, 1, 20)
        Mr = ri**2
        f_log = _log_pchip(ri, Mr)
        f_lin = PchipInterpolator(ri, Mr, extrapolate=False)
        r_mid = np.sqrt(ri[:-1] * ri[1:])
        truth = r_mid**2
        err_log = np.max(np.abs(f_log(r_mid) - truth) / truth)
        err_lin = np.max(np.abs(f_lin(r_mid) - truth) / truth)
        assert err_log < 1e-12, f"log-PCHIP rel error {err_log:.3e} too large"
        assert err_lin > 1e-3, f"linear PCHIP rel error {err_lin:.3e} unexpectedly small"

    def test_rho_cusp_reproduction(self):
        """rho ~ r^-1.5 on 30 log-knots: log-PCHIP rel error < 1e-10 at
        log midpoints. Linear PCHIP visibly worse (>= 1e-2 on the steepest
        decade), so the cusp slope at the innermost knots is the practical
        win."""
        from scipy.interpolate import PchipInterpolator
        ri = np.logspace(-3, 1, 30)
        rho = ri**(-1.5)
        f_log = _log_pchip(ri, rho)
        f_lin = PchipInterpolator(ri, rho, extrapolate=False)
        r_mid = np.sqrt(ri[:-1] * ri[1:])
        truth = r_mid**(-1.5)
        err_log = np.max(np.abs(f_log(r_mid) - truth) / truth)
        err_lin_inner = np.max(np.abs(f_lin(r_mid[:5]) - truth[:5]) / truth[:5])
        assert err_log < 1e-10, f"log-PCHIP rel error {err_log:.3e}"
        assert err_lin_inner > 1e-2, (
            f"linear PCHIP cusp error {err_lin_inner:.3e} unexpectedly small"
        )

    def test_M_boundary_clamps_preserved(self):
        """M(0.5*ri[0]) returns M(ri[0]); M(2*ri[-1]) returns Mh. With the
        wrap helper handling the clamp, NumericProfile.M no longer needs
        np.clip — confirm the contract still holds end-to-end."""
        ri = np.logspace(-3, np.log10(0.1), 50)
        Mr = ri**0.5
        p = NumericProfile(ri, Mr)
        m_below = float(p.M(0.5 * p.ri[0]))
        m_above = float(p.M(2.0 * p.ri[-1]))
        assert np.isfinite(m_below) and np.isfinite(m_above)
        # below: returns the first masked y (typically Mr[0] when all knots positive)
        assert abs(m_below - float(p.MInt(p.ri[0]))) < 1e-12
        # above: returns Mh
        assert abs(m_above - p.Mh) < 1e-9 * p.Mh

    def test_rho_boundary_returns_zero(self):
        """rho query outside [ri[0], ri[-1]] returns 0 — clamp_below='zero'
        and clamp_above='zero' carry through. NaN here would propagate
        into ev.ltidal and the rmax root-finder."""
        ri = np.logspace(-3, np.log10(0.1), 50)
        sat = NFW(1e6, 10.)
        Mr = sat.M(ri)
        p = NumericProfile(ri, Mr)
        assert float(p.rho(p.ri[-1] * 2.)) == 0.
        assert float(p.rho(p.ri[0] * 0.5)) == 0.

    def test_rho_zero_tail_preserved(self):
        """A profile with rho == 0 on the outer 30% of knots must read
        zero in the tail. leading_only=True restricts the log-PCHIP to the
        leading positive run; everything past the first zero clamps to 0."""
        ri = np.logspace(-3, -1, 100)
        # cusp + sharp truncation at 70% of the knot range
        cut = int(0.7 * len(ri))
        Mr = np.zeros_like(ri)
        Mr[:cut] = ri[:cut]**2
        Mr[cut:] = Mr[cut - 1]
        p = NumericProfile(ri, Mr)
        rho_tail = p.rho(ri[cut + 5:])
        assert np.all(rho_tail == 0.), f"rho tail not zero, max={rho_tail.max():.3e}"
        # cusp interior still gets log-PCHIP accuracy
        rho_inner = p.rho(ri[10:20])
        assert np.all(rho_inner > 0.)

    def test_M_convergence_rate_flat(self):
        """M = r^2.5 on N ∈ {10, 20, 40, 80} log-knots. log-PCHIP error is
        roundoff-flat (~1e-13); linear PCHIP error scales ~h^2. The flatness
        is the strongest single argument for the change."""
        from scipy.interpolate import PchipInterpolator
        Ns = [10, 20, 40, 80]
        errs_log, errs_lin = [], []
        for N in Ns:
            ri = np.logspace(-3, 1, N)
            Mr = ri**2.5
            r_mid = np.sqrt(ri[:-1] * ri[1:])
            truth = r_mid**2.5
            f_log = _log_pchip(ri, Mr)
            f_lin = PchipInterpolator(ri, Mr, extrapolate=False)
            errs_log.append(np.max(np.abs(f_log(r_mid) - truth) / truth))
            errs_lin.append(np.max(np.abs(f_lin(r_mid) - truth) / truth))
        # log: flat at machine precision across all N
        assert max(errs_log) < 1e-10, f"log-PCHIP errs not flat: {errs_log}"
        # linear: error halves (or better) when N doubles — confirms ~h^2 scaling
        assert errs_lin[0] / errs_lin[-1] > 16., (
            f"linear PCHIP did not converge as expected: {errs_lin}"
        )

    def test_heat_profile_first_shell_no_log_blowup(self):
        """heat_profile's final rebin uses log-PCHIP on (r_bound, M_bound)
        from cumsum. If the first shell carries a tiny mass relative to the
        rest, log(M_bound[0]) is very negative — the rebin grid starts at
        r_bound[0] so the helper never extrapolates. Pin this behaviour
        with a profile that exposes it (NFW with extreme inner resolution)."""
        ri = np.logspace(-5, np.log10(0.5), 200)  # innermost shell << rest
        sat = NFW(1e6, 10.)
        p = NumericProfile(ri, sat.M(ri))
        def eps(r): return 0.1 * r**2
        h = heat_profile(p, eps)
        assert np.all(np.isfinite(h.Mr))
        assert np.all(h.Mr >= 0.)
        assert np.all(np.diff(h.Mr) >= -1e-9 * h.Mh)
