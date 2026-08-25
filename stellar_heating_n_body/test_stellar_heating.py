"""Tests for the N-body stellar-heating kernels."""
import numpy as np
import pytest

from stellar_heating import (NFW, _G_NB, _accel_cart_clumps_nb, _accel_cart_tabM_nb,
                             _dekel_M_enc_grid, _bound_mask, Plummer, init_plummer,
                             _PLUMMER_R0_FROM_RHALF)

D_GRID = np.linspace(np.log(1e-7), np.log(3.0), 96)


def host_grid(host, n=1200):
    r = np.logspace(-6, 2, n)
    return np.log(r), np.asarray(host.M(r))


@pytest.fixture
def host():
    return NFW(M=1e9, c=24.58)


def test_tabM_matches_the_dekel_kernel(host):
    """Fed the Dekel M(<d) it evaluates analytically, the tabulated kernel must return
    the same acceleration -- the table is a change of representation, not of physics."""
    rng = np.random.default_rng(0)
    log_r, M_enc = host_grid(host)
    n_sub = 12
    pos = rng.normal(0., 0.03, (200, 3))
    mc = rng.normal(0., 0.5, (n_sub, 3))
    Mh = np.full(n_sub, 1e5)
    rs = np.full(n_sub, 5e-4)
    c = np.full(n_sub, 20.)
    al = np.full(n_sub, 0.5)

    a_dekel = _accel_cart_clumps_nb(pos, mc, Mh, rs, c, al, log_r, M_enc)
    tab = np.stack([_dekel_M_enc_grid(np.exp(D_GRID), Mh[j], rs[j], c[j], al[j])
                    for j in range(n_sub)])
    a_tab = _accel_cart_tabM_nb(pos, mc, tab, D_GRID, log_r, M_enc)

    # the table is log-linear between knots, so it tracks the closed form to the grid's
    # own resolution rather than to machine precision
    assert np.allclose(a_tab, a_dekel, rtol=2e-3)


def test_tabM_is_point_mass_at_large_separation(host):
    """Past its outer knot a clump carries all its mass, so the force is G M / d^2."""
    log_r, M_enc = host_grid(host)
    M_clump = 1e5
    pos = np.array([[0.5, 0., 0.]])
    mc = np.array([[-0.5, 0., 0.]])
    tab = np.full((1, D_GRID.size), M_clump)

    a = _accel_cart_tabM_nb(pos, mc, tab, D_GRID, log_r, M_enc)
    a_host = _accel_cart_tabM_nb(pos, mc, np.zeros((1, D_GRID.size)), D_GRID,
                                 log_r, M_enc)
    assert a[0, 0] - a_host[0, 0] == pytest.approx(-_G_NB * M_clump / 1.0**2, rel=1e-10)


def test_tabM_force_vanishes_at_the_clump_centre(host):
    """M(<d) -> 0 as d -> 0 for any physical clump, so the extended force softens on the
    clump's own scale and no softening length is needed."""
    log_r, M_enc = host_grid(host)
    d = np.exp(D_GRID)
    tab = (1e5 * np.minimum(d / 1e-3, 1.) ** 2)[None, :]
    sep = np.array([1e-6, 1e-5, 1e-4])
    a = np.array([
        _accel_cart_tabM_nb(np.array([[s, 0., 0.]]), np.zeros((1, 3)), tab, D_GRID,
                            log_r, np.zeros_like(M_enc))[0, 0]
        for s in sep])
    # M ~ d^2 against a d^-2 force law leaves |a| constant, never divergent. The log
    # grid interpolates a convex M(<d) from above, so the level sits a per cent over the
    # closed form -- flatness across 100x in d is the property under test.
    assert np.all(np.isfinite(a))
    assert np.ptp(np.abs(a)) / np.abs(a).mean() < 0.02
    assert np.abs(a).mean() == pytest.approx(_G_NB * 1e5 / 1e-3**2, rel=0.02)


def test_bound_mask_calls_a_deep_cold_star_bound_and_a_fast_one_free(host):
    r = np.logspace(-6, 2, 1200)
    log_r, Phi = np.log(r), np.asarray(host.Phi(r))
    pos = np.array([[0.01, 0., 0.], [0.01, 0., 0.]])
    vesc = np.sqrt(-2. * np.interp(np.log(0.01), log_r, Phi))
    vel = np.array([[0., 0.1 * vesc, 0.], [0., 1.5 * vesc, 0.]])
    assert _bound_mask(pos, vel, log_r, Phi).tolist() == [True, False]


def test_plummer_half_mass_radius_matches_the_scale_radius_conversion(host):
    """The R_0* <-> r_1/2 conversion the drivers report through."""
    R0 = 2e-3
    pos, _ = init_plummer(1e3, R0, host, m_star=1.0, seed=3)
    r_half = np.percentile(np.linalg.norm(pos, axis=1), 50)
    assert r_half * _PLUMMER_R0_FROM_RHALF == pytest.approx(R0, rel=0.05)


def test_plummer_density_integrates_to_its_total_mass():
    R0, M = 2e-3, 1e3
    r = np.logspace(-8, 1, 40000)
    m = np.trapezoid(4. * np.pi * r**2 * Plummer(r, M, R0), r)
    assert m == pytest.approx(M, rel=1e-4)


def test_kernels_survive_a_non_finite_star(host):
    """A star flung to a non-finite radius must not index out of bounds.

    t = (log r - log r_min) * inv_dlr is NaN there, and NaN fails BOTH `t < 0` and
    `t > hi`, so a negatively-phrased clamp falls through to int(NaN) and reads garbage
    -- a segfault, not an exception. Reproduces as SIGSEGV before the positive guard.
    """
    log_r = np.log(np.logspace(-6, 2, 1200))
    M_enc = np.asarray(host.M(np.logspace(-6, 2, 1200)))
    mc = np.array([[0.5, 0., 0.]])
    tab = np.full((1, D_GRID.size), 1e5)
    pos = np.array([[0.01, 0., 0.], [np.nan, 0., 0.], [np.inf, 0., 0.]])

    a = _accel_cart_tabM_nb(pos, mc, tab, D_GRID, log_r, M_enc)
    assert np.isfinite(a[0]).all()          # the healthy star is unaffected


def test_menc_from_profiles_takes_any_object_exposing_M():
    """The point of the tabulated kernel: a profile with no closed form goes straight in.

    A nopython kernel cannot call a scipy spline behind a Python object, so tabulating at
    the boundary is what makes the njit force possible at all -- and it is why a
    Dekel-specific kernel is not needed.
    """
    from stellar_heating import menc_from_profiles

    class Duck:                      # anything with .M(r) qualifies
        def M(self, r):
            return 1e5 * (r / (r + 1e-3))**2

    prof = Duck()
    tab, log_d = menc_from_profiles([prof], n_d=256, d_lo=1e-6, d_hi=1e-2)
    assert np.allclose(tab[0], prof.M(np.exp(log_d)), rtol=1e-12)

    # and it truncates at the bound mass, the calibration's 'hard' convention
    cut, _ = menc_from_profiles([prof], log_d_grid=log_d, m_bound=np.array([1e3]))
    assert cut[0].max() == pytest.approx(1e3, rel=1e-12)


def test_menc_from_profiles_takes_a_satgen_numeric_profile():
    """The case that motivated it: an evolved remnant, tabulated straight off the spline.

    config.py loads etc/*.npy relative to the working directory, so the import has to run
    from the repo root.
    """
    import os, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cwd = os.getcwd()
    sys.path.insert(0, root)
    os.chdir(root)
    try:
        from subhalo_functions import NumericProfile
        from stellar_heating import menc_from_profiles, NFW
        r = np.logspace(-6, -2, 200)
        num = NumericProfile(r, np.asarray(NFW(M=1e5, c=20.).M(r)))
        tab, log_d = menc_from_profiles([num], n_d=256, d_lo=1e-6, d_hi=1e-2)
        d = np.exp(log_d)
        inside = (d > r[0]) & (d < r[-1])
        assert np.allclose(tab[0][inside], np.asarray(num.M(d[inside])), rtol=1e-10)
    finally:
        os.chdir(cwd)


def test_menc_from_profiles_zeroes_a_disrupted_clump():
    from stellar_heating import menc_from_profiles
    tab, _ = menc_from_profiles([None], n_d=32)
    assert tab.shape == (1, 32) and not tab.any()
