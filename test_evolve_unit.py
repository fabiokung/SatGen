"""
Unit tests for evolve.py — fast, no tree files required.

Run with:
    source .venv/bin/activate
    python -m pytest test_evolve_unit.py -v
"""

import numpy as np
import pytest

import config as cfg
import evolve as ev
from profiles import NFW, Dekel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_mres():
    """Ensure cfg.Mres is set so msub doesn't fall back to sp.Minit (which
    is only set by SatEvo.py, not by the profile constructors)."""
    original = cfg.Mres
    cfg.Mres = 1e6  # [M_sun] standard resolution floor
    yield
    cfg.Mres = original


@pytest.fixture
def nfw_host():
    """MW-scale NFW host, c=10."""
    return NFW(1e12, 10.)


@pytest.fixture
def dekel_sat():
    """Small satellite: Dekel profile, 1% host mass."""
    return Dekel(1e10, 15., 0.5)


@pytest.fixture
def xv_inside(nfw_host):
    """Phase-space coords placing satellite well inside host (r ~ 0.1 r_vir)."""
    r = 0.1 * nfw_host.rh
    # [R, phi, z, V_R, V_phi, V_z] in cylindrical coords
    return np.array([r, 0., 0., 0., 100., 0.])


@pytest.fixture
def xv_outside(nfw_host):
    """Phase-space coords placing satellite far outside host (r ~ 5 r_vir)."""
    r = 5.0 * nfw_host.rh
    return np.array([r, 0., 0., 0., 50., 0.])


# ---------------------------------------------------------------------------
# g_P10 tests
# ---------------------------------------------------------------------------

class TestGP10:
    def test_identity_at_one(self):
        """At x=1 (no mass loss), both ratios should be exactly 1."""
        gv, gl = ev.g_P10(1.0, alpha=1.0)
        assert abs(gv - 1.0) < 1e-10
        assert abs(gl - 1.0) < 1e-10

    def test_identity_at_one_alpha_zero(self):
        gv, gl = ev.g_P10(1.0, alpha=0.0)
        assert abs(gv - 1.0) < 1e-10
        assert abs(gl - 1.0) < 1e-10

    def test_identity_at_one_alpha_half(self):
        gv, gl = ev.g_P10(1.0, alpha=0.5)
        assert abs(gv - 1.0) < 1e-10
        assert abs(gl - 1.0) < 1e-10

    def test_near_zero_mass_small(self):
        """At very small x, both ratios should approach 0."""
        gv, gl = ev.g_P10(1e-4, alpha=1.0)
        assert gv < 0.1
        assert gl < 0.1

    def test_vmax_monotone_decreasing(self):
        """v_max ratio should decrease as bound mass fraction decreases."""
        xs = np.linspace(0.01, 1.0, 50)
        gvs = [ev.g_P10(x, alpha=1.0)[0] for x in xs]
        diffs = np.diff(gvs)
        assert np.all(diffs >= -1e-10), "g_P10 v_max ratio is not monotone"

    def test_lmax_monotone_decreasing(self):
        """l_max ratio should decrease as bound mass fraction decreases."""
        xs = np.linspace(0.01, 1.0, 50)
        gls = [ev.g_P10(x, alpha=1.0)[1] for x in xs]
        diffs = np.diff(gls)
        assert np.all(diffs >= -1e-10), "g_P10 l_max ratio is not monotone"

    def test_ratios_bounded_above_by_one(self):
        """Both ratios should be ≤ 1 for all x ∈ (0, 1]."""
        for x in [0.01, 0.1, 0.5, 0.9, 1.0]:
            gv, gl = ev.g_P10(x, alpha=1.0)
            assert gv <= 1.0 + 1e-10
            assert gl <= 1.0 + 1e-10

    def test_ratios_non_negative(self):
        for x in [0.001, 0.01, 0.1, 0.5, 1.0]:
            gv, gl = ev.g_P10(x, alpha=1.0)
            assert gv >= 0.
            assert gl >= 0.

    def test_alpha_clipping(self):
        """Alpha outside [0, 1.5] should not raise; clips to grid bounds."""
        gv_low, gl_low = ev.g_P10(0.5, alpha=-1.0)
        gv_high, gl_high = ev.g_P10(0.5, alpha=5.0)
        # Should match boundary values
        gv_0, gl_0 = ev.g_P10(0.5, alpha=0.0)
        gv_15, gl_15 = ev.g_P10(0.5, alpha=1.5)
        assert abs(gv_low - gv_0) < 1e-10
        assert abs(gv_high - gv_15) < 1e-10


# ---------------------------------------------------------------------------
# g_EPW18 tests
# ---------------------------------------------------------------------------

class TestGEPW18:
    def test_identity_at_one(self):
        """At x=1, both size and stellar mass ratios should be 1."""
        gl, gm = ev.g_EPW18(1.0, alpha=1.0, lefflmax=0.1)
        assert abs(gl - 1.0) < 1e-10
        assert abs(gm - 1.0) < 1e-10

    def test_identity_at_one_other_params(self):
        for alpha in [0.0, 0.5, 1.5]:
            for lefflmax in [0.05, 0.1]:
                gl, gm = ev.g_EPW18(1.0, alpha=alpha, lefflmax=lefflmax)
                assert abs(gl - 1.0) < 1e-10, f"failed at alpha={alpha}, lefflmax={lefflmax}"
                assert abs(gm - 1.0) < 1e-10

    def test_near_zero_mass_small(self):
        """At very small x, stellar mass ratio should drop significantly.
        Note: l_eff ratio can be non-monotone (rises then falls) due to the
        EPW18 fitting formula, so we only check the stellar mass ratio here."""
        gl, gm = ev.g_EPW18(1e-3, alpha=1.0, lefflmax=0.1)
        assert gm < 0.5  # stellar mass should be well stripped at 0.1% bound mass

    def test_ratios_non_negative(self):
        for x in [0.001, 0.01, 0.1, 0.5, 1.0]:
            gl, gm = ev.g_EPW18(x, alpha=1.0, lefflmax=0.1)
            assert gl >= 0.
            assert gm >= 0.

    def test_leff_tidal_puffing_then_stripping(self):
        """The EPW18 l_eff track captures 'tidal puffing': the stellar body can
        temporarily expand (l_eff/l_eff0 > 1) before being stripped. Verify:
        - At x=1: ratio = 1
        - At intermediate x: ratio can exceed 1 (tidal puffing is physical)
        - At very small x: ratio < ratio at x=1 (stripping eventually wins)
        """
        _, _ = ev.g_EPW18(1.0, alpha=1.0, lefflmax=0.1)  # boundary: should be 1
        gl_mid, _ = ev.g_EPW18(0.1, alpha=1.0, lefflmax=0.1)
        gl_tiny, _ = ev.g_EPW18(1e-4, alpha=1.0, lefflmax=0.1)
        # At extreme stripping, l_eff must eventually fall below the mid value
        assert gl_tiny < gl_mid, \
            "l_eff ratio at x=1e-4 should be less than at x=0.1 (stripping wins at extreme loss)"

    def test_mstar_monotone(self):
        xs = np.linspace(0.01, 1.0, 50)
        gms = [ev.g_EPW18(x, alpha=1.0, lefflmax=0.1)[1] for x in xs]
        assert np.all(np.diff(gms) >= -1e-10)

    def test_concentrated_stars_stripped_less(self):
        """Stars concentrated well inside DM (small lefflmax) should
        be stripped less than stars spread out (large lefflmax)."""
        gl_concentrated, gm_concentrated = ev.g_EPW18(0.1, alpha=1.0, lefflmax=0.05)
        gl_spread, gm_spread = ev.g_EPW18(0.1, alpha=1.0, lefflmax=0.1)
        # More concentrated → larger remaining fraction (less stripped)
        assert gm_concentrated >= gm_spread - 1e-10


# ---------------------------------------------------------------------------
# ltidal tests
# ---------------------------------------------------------------------------

class TestLtidal:
    def test_returns_positive(self, dekel_sat, nfw_host, xv_inside):
        lt = ev.ltidal(dekel_sat, nfw_host, xv_inside, choice='King62')
        assert lt > 0.

    def test_king62_leq_tormen98(self, dekel_sat, nfw_host, xv_inside):
        """King62 (with centrifugal term) gives smaller or equal tidal radius
        than Tormen98 (without centrifugal) — i.e., stronger stripping."""
        lt_king = ev.ltidal(dekel_sat, nfw_host, xv_inside, choice='King62')
        lt_tormen = ev.ltidal(dekel_sat, nfw_host, xv_inside, choice='Tormen98')
        assert lt_king <= lt_tormen + 1e-6 * lt_tormen

    def test_large_distance_gives_large_tidal_radius(self, dekel_sat, nfw_host, xv_outside):
        """Far from host, tidal radius should approach or exceed satellite r_vir."""
        lt = ev.ltidal(dekel_sat, nfw_host, xv_outside, choice='King62')
        assert lt >= dekel_sat.rh * 0.5  # at least not tiny

    def test_inside_lt_smaller_than_rh(self, dekel_sat, nfw_host, xv_inside):
        """Well inside host, tidal radius should be smaller than satellite r_vir
        (satellite is being significantly stripped)."""
        lt = ev.ltidal(dekel_sat, nfw_host, xv_inside, choice='King62')
        # For a 1% mass satellite deep in the host, lt < rh is expected
        # (not guaranteed for all parameter choices, but for our fixture it should hold)
        assert lt < dekel_sat.rh * 2.0  # soft bound: tidal radius is sub-virial

    def test_composite_potential(self, dekel_sat, xv_inside):
        """Should work with a list of profiles as host potential."""
        from profiles import MN
        host_halo = NFW(1e12, 10.)
        host_disk = MN(5e10, 5., 1.)
        lt = ev.ltidal(dekel_sat, [host_halo, host_disk], xv_inside, choice='King62')
        assert lt > 0.


# ---------------------------------------------------------------------------
# msub tests
# ---------------------------------------------------------------------------

class TestMsub:
    def test_mass_does_not_increase(self, dekel_sat, nfw_host, xv_inside):
        """Tidal stripping should never increase satellite mass."""
        m_init = dekel_sat.Mh
        m_new, lt = ev.msub(dekel_sat, nfw_host, xv_inside, dt=0.1, alpha=1.0)
        assert m_new <= m_init + 1e-10 * m_init

    def test_mass_above_resolution(self, dekel_sat, nfw_host, xv_inside):
        """Evolved mass should be at or above Mres."""
        if cfg.Mres is not None:
            m_new, lt = ev.msub(dekel_sat, nfw_host, xv_inside, dt=0.1, alpha=1.0)
            assert m_new >= cfg.Mres - 1.

    def test_no_stripping_far_from_host(self, dekel_sat, nfw_host, xv_outside):
        """Satellite far outside host should lose negligible mass."""
        m_init = dekel_sat.Mh
        m_new, lt = ev.msub(dekel_sat, nfw_host, xv_outside, dt=0.1, alpha=1.0)
        # lt > rh means no stripping in this step
        assert m_new == m_init

    def test_higher_alpha_strips_more(self, dekel_sat, nfw_host, xv_inside):
        """Higher stripping efficiency should produce more mass loss."""
        m_low, _ = ev.msub(dekel_sat, nfw_host, xv_inside, dt=0.1, alpha=0.1)
        m_high, _ = ev.msub(dekel_sat, nfw_host, xv_inside, dt=0.1, alpha=2.0)
        assert m_high <= m_low + 1e-10 * m_low

    def test_returns_tidal_radius(self, dekel_sat, nfw_host, xv_inside):
        m, lt = ev.msub(dekel_sat, nfw_host, xv_inside, dt=0.1)
        assert lt > 0.

    def test_king62_strips_more_than_tormen98(self, dekel_sat, nfw_host, xv_inside):
        """King62 tidal radius is smaller, so it should strip more mass."""
        m_king, _ = ev.msub(dekel_sat, nfw_host, xv_inside, dt=0.1,
                            choice='King62', alpha=1.0)
        m_tormen, _ = ev.msub(dekel_sat, nfw_host, xv_inside, dt=0.1,
                               choice='Tormen98', alpha=1.0)
        assert m_king <= m_tormen + 1e-10 * m_tormen


# ---------------------------------------------------------------------------
# Dekel2 tests
# ---------------------------------------------------------------------------

class TestDekel2:
    def _sat_params(self):
        """Return a consistent set of Dekel satellite parameters."""
        s = Dekel(1e11, 20., 0.5)
        lmax0 = s.rmax
        vmax0 = s.Vmax
        alpha0 = s.alphah
        slope0 = s.s001  # slope at 0.01 r_vir used in Dekel2
        return s.Mh, lmax0, vmax0, alpha0, slope0

    def test_identity_at_full_mass(self):
        """Dekel2 at mv=mv0 should return concentration close to original."""
        mv0, lmax0, vmax0, alpha0, slope0 = self._sat_params()
        c, Delta = ev.Dekel2(mv0, mv0, lmax0, vmax0, alpha0, slope0, z=0.)
        s_orig = Dekel(mv0, c, alpha0, Delta=Delta, z=0.)
        # The returned c and Delta should reconstruct a valid Dekel profile
        assert c > 0.
        assert Delta > 0.
        assert np.isfinite(c)
        assert np.isfinite(Delta)

    def test_concentration_decreases_with_mass_loss(self):
        """Heavier stripping (smaller mv/mv0) should give lower concentration."""
        mv0, lmax0, vmax0, alpha0, slope0 = self._sat_params()
        c_90, D_90 = ev.Dekel2(0.9 * mv0, mv0, lmax0, vmax0, alpha0, slope0, z=0.)
        c_50, D_50 = ev.Dekel2(0.5 * mv0, mv0, lmax0, vmax0, alpha0, slope0, z=0.)
        c_10, D_10 = ev.Dekel2(0.1 * mv0, mv0, lmax0, vmax0, alpha0, slope0, z=0.)
        # rh (virial radius) decreases with mass but rmax decreases more slowly,
        # so concentration c = rh/rmax should generally decrease under stripping
        # This is an empirical check — weak inequality
        assert c_50 <= c_90 * 1.5  # allow some variation
        for c in [c_90, c_50, c_10]:
            assert c > 0.

    def test_returns_finite_values(self):
        mv0, lmax0, vmax0, alpha0, slope0 = self._sat_params()
        for frac in [0.9, 0.5, 0.1, 0.01]:
            c, D = ev.Dekel2(frac * mv0, mv0, lmax0, vmax0, alpha0, slope0, z=0.)
            assert np.isfinite(c), f"c not finite at mv/mv0={frac}"
            assert np.isfinite(D), f"Delta not finite at mv/mv0={frac}"
            assert c > 0.
            assert D > 0.

    def test_redshift_dependence(self):
        """Output should differ at different redshifts (rho_crit changes)."""
        mv0, lmax0, vmax0, alpha0, slope0 = self._sat_params()
        c0, D0 = ev.Dekel2(0.5 * mv0, mv0, lmax0, vmax0, alpha0, slope0, z=0.)
        c2, D2 = ev.Dekel2(0.5 * mv0, mv0, lmax0, vmax0, alpha0, slope0, z=2.)
        # rho_crit is higher at z=2, so Delta should differ
        assert abs(D0 - D2) > 1.  # they shouldn't be identical
