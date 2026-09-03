"""Tests for the GPU tabulated-M(<d) kernels and their CPU-side helpers.

The helpers run anywhere. The device tests need nvcc and a GPU and skip otherwise; what
they hold is that the device force is the CPU force -- the tabulated kernel and the
original point-mass and Dekel ones -- and that the device stepping is the original
drivers' stepping, to rounding where the inputs are the same numbers.
"""
import os
import shutil
import sys

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'stellar_heating_n_body'))
import tabm_gpu as tg
from stellar_heating import (NFW, _G_NB, _PLUMMER_R0_FROM_RHALF, _accel_cart_tabM_nb,
                             _accel_cart_nb, _accel_cart_clumps_nb, _dekel_M_enc_grid,
                             init_plummer, stellar_simulation_nb, stellar_simulation_clumps_nb)

D_GRID = np.linspace(np.log(1e-7), np.log(3.0), 96)
D_FINE = np.linspace(np.log(1e-7), np.log(30.0), 4096)
GPU = shutil.which('nvcc') is not None


def _orig_host_grid(host):
    """The 1000-point M(<r) lookup stellar_simulation_nb and stellar_simulation_clumps_nb
    build, so a comparison against them is down to arithmetic and not to the host grid."""
    r = np.logspace(-5, 2, 1000)
    return np.log(r), np.asarray(host.M(r))


def _circles(rng, n, t_knots, r_lo=0.005, r_hi=0.5):
    """n MACHOs on circular orbits, as the cylindrical (n, n_t, 6) array the drivers take."""
    R = np.exp(rng.uniform(np.log(r_lo), np.log(r_hi), n))
    Om = 20. / np.sqrt(R)
    ph0 = rng.uniform(0., 2 * np.pi, n)
    phi = ph0[:, None] + Om[:, None] * t_knots[None, :]
    z = 0.3 * R[:, None] * np.sin(0.5 * phi)
    out = np.zeros((n, t_knots.size, 6))
    out[:, :, 0] = R[:, None]
    out[:, :, 1] = phi
    out[:, :, 2] = z
    out[:, :, 4] = (R * Om)[:, None]
    out[:, :, 5] = 0.15 * (R * Om)[:, None] * np.cos(0.5 * phi)
    return out


def _cart(xv_i):
    """Cartesian positions of one snapshot of a cylindrical (n, 6) array."""
    return np.stack([xv_i[:, 0] * np.cos(xv_i[:, 1]), xv_i[:, 0] * np.sin(xv_i[:, 1]),
                     xv_i[:, 2]], axis=1)


def _r_half_origin(pos):
    return float(np.percentile(np.linalg.norm(pos, axis=1), 50))


def _ident(n):
    return np.arange(n, dtype=np.int32)


def test_g_matches_the_cpu_kernel():
    assert tg.G_NB == _G_NB


def test_clump_rows_flat_tail():
    d = np.exp(D_GRID)
    rising = 1e5 * (d / (d + 1e-3))**2
    trunc = np.minimum(rising, 2e4)
    m_out, rt2 = tg.clump_rows(np.stack([np.full(96, 5.), rising, trunc]), D_GRID)
    assert m_out.tolist() == [5., rising[-1], 2e4]
    assert rt2[0] == d[1] * d[1]  # a point mass outside the first knot, extrapolated inside
    assert rt2[1] == d[-1] * d[-1]  # held at the last knot beyond it
    assert rt2[2] == d[np.argmax(trunc >= 2e4)]**2


def test_substeps_match_the_cpu_rule():
    host = NFW(M=1e9, c=24.58)
    dt = 12. / 24000
    omega = np.sqrt(_G_NB * host.Mh / host.rs**3)
    assert tg.yoshida_substeps(host, dt, 200.) == max(1, int(np.ceil(dt * omega * 200.)))


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_advance_evaluates_copies_as_the_host_would():
    """Position by a cubic Hermite through the knots at the phase-wrapped time, held at the
    last knot past it, rotated into the copy's plane; the M(<d) row as the time blend of
    the bound-fraction rows, hard-truncated at the mass, zero once dead."""
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = tg.host_grid(host)
    rng = np.random.default_rng(4)
    d = np.exp(D_GRID)
    t = np.linspace(0., 12., 300)
    p = np.stack([0.02 * np.cos(3. * t), 0.02 * np.sin(3. * t), 0.005 * t], axis=1)
    v = np.stack([-0.06 * np.sin(3. * t), 0.06 * np.cos(3. * t), np.full_like(t, 0.005)], axis=1)
    m = 1e4 - 500. * t
    M0 = 1e4 * (d / (d + 5e-4))**2
    MT = 4e3 * (d / (d + 3e-4))**2
    bank = dict(kt=t, kp=p, kv=v, km=m, koff=np.array([0, t.size, t.size]),
                M0=np.stack([M0, M0]), MT=np.stack([MT, MT]), hasT=np.array([1, 0]),
                m_init=np.array([1e4, 1e4]), m_final=np.array([4e3, 1e4]),
                spos=np.array([[0., 0., 0.], [0.01, -0.02, 0.03]]), smass=np.array([0., 1e4]))
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    rot = np.array([[1 - 2 * (q[2]**2 + q[3]**2), 2 * (q[1] * q[2] - q[3] * q[0]),
                     2 * (q[1] * q[3] + q[2] * q[0])],
                    [2 * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * (q[1]**2 + q[3]**2),
                     2 * (q[2] * q[3] - q[1] * q[0])],
                    [2 * (q[1] * q[3] - q[2] * q[0]), 2 * (q[2] * q[3] + q[1] * q[0]),
                     1 - 2 * (q[1]**2 + q[2]**2)]])
    orbit = np.array([0, 0, 1, 0])
    phase = np.array([0., 4.3, 0., 0.])
    t_alive = np.array([np.inf, np.inf, np.inf, 5.])
    g = tg.TabMGpu(4, 4, 4, D_GRID, log_r, M_enc)
    g.set_bank(**bank)
    g.set_copies(orbit, phase, np.stack([np.eye(3), rot, rot, np.eye(3)]), t_alive)
    sp = CubicHermiteSpline(t, p, v, axis=0)

    def row(mq, hasT):
        w = np.clip((mq - 1e4) / (4e3 - 1e4), 0., 1.) if hasT else 0.
        return np.minimum((1. - w) * M0 + w * MT, mq)

    for t_pos, tA, tB, wM, t_end in ((2.0, 0., 3., 2.5 / 3., 2.5), (13., 12., 12., 0., 13.)):
        g.advance(t_pos, tA, tB, wM, t_end)
        xyz, m_out, rt2, rows = g.clumps()
        want = [sp(min(t_pos, 12.)), rot @ sp((t_pos + 4.3) % 12.), rot @ bank['spos'][1],
                sp(min(t_pos, 12.))]
        assert np.allclose(xyz, want, rtol=1e-12, atol=1e-16)
        mA, mB = np.interp(min(tA, 12.), t, m), np.interp(min(tB, 12.), t, m)
        live = (1. - wM) * row(mA, 1) + wM * row(mB, 1)
        want_rows = [live,
                     (1. - wM) * row(np.interp((tA + 4.3) % 12., t, m), 1)
                     + wM * row(np.interp((tB + 4.3) % 12., t, m), 1),
                     np.minimum(M0, 1e4), np.zeros(96) if t_alive[3] <= t_end else live]
        assert np.allclose(rows, want_rows, rtol=1e-12, atol=0.)
        assert np.array_equal(m_out, rows[:, -1])
        assert np.array_equal(rt2, tg.clump_rows(rows, D_GRID)[1])
    assert (rows[3] == 0.).all()  # dead by the second case
    g.close()


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_gpu_force_is_the_cpu_force():
    """Against _accel_cart_tabM_nb on rows with a spread of truncation radii, clumps close
    enough that many pairs read the table, both arms, both precisions."""
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = tg.host_grid(host)
    rng = np.random.default_rng(1)
    pos, vel = init_plummer(1e3, 2e-3, host, m_star=1.0, seed=3)
    n = 500
    d = np.exp(D_GRID)
    table = np.stack([np.minimum(1e4 * (d / (d + 2e-3))**2, m) for m in
                      rng.uniform(2e2, 1e4, 40)])
    orb = rng.integers(0, 40, n).astype(np.int32)
    xyz = rng.normal(0., 0.02, (n, 3))
    m_out, rt2 = tg.clump_rows(table, D_GRID)
    a_cpu = _accel_cart_tabM_nb(pos, xyz, table[orb], D_GRID, log_r, M_enc, 1.0)
    g = tg.TabMGpu(pos.shape[0], n, 40, D_GRID, log_r, M_enc)
    g.set_stars(pos, vel)
    g.set_table(table)
    g.set_clumps(xyz, m_out[orb], rt2[orb], orb)
    for fp64, tol in ((True, 1e-12), (False, 1e-5)):
        a = g.accel(m_star_sg=1.0, fp64=fp64)
        assert np.abs(a - a_cpu).max() <= tol * np.abs(a_cpu).max()
    g.close()


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_inner_extrapolation_is_the_cpu_extrapolation():
    """Inside the first knot the CPU reads M = row[0] (d / d_min)^3; the device must,
    whatever the row's flat-tail radius says."""
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = tg.host_grid(host)
    pos, vel = init_plummer(1e3, 2e-3, host, m_star=1.0, seed=3)
    d = np.exp(D_GRID)
    table = np.stack([np.minimum(1e4 * (d / (d + 2e-3))**2, 3e3), np.full(96, 5e3)])
    n = 6
    xyz = pos[:n] + np.array([[1e-9, 0, 0], [0, 3e-8, 0], [0, 0, -8e-8],
                              [5e-9, 5e-9, 0], [0, -2e-8, 2e-8], [9.9e-8, 0, 0]])
    orb = np.array([0, 1, 0, 1, 0, 1], np.int32)
    m_out, rt2 = tg.clump_rows(table, D_GRID)
    a_cpu = _accel_cart_tabM_nb(pos, xyz, table[orb], D_GRID, log_r, M_enc, 0.)
    a_host = _accel_cart_tabM_nb(pos, xyz, np.zeros((n, 96)), D_GRID, log_r, M_enc, 0.)
    g = tg.TabMGpu(pos.shape[0], n, 2, D_GRID, log_r, M_enc)
    g.set_stars(pos, vel)
    g.set_table(table)
    g.set_clumps(xyz, m_out[orb], rt2[orb], orb)
    a = g.accel(fp64=True)
    pair = np.abs(a_cpu - a_host).max()
    assert pair > 0.
    assert np.abs(a - a_cpu).max() <= 1e-12 * pair
    g.close()


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_force_is_reproducible_across_split_blocks():
    """The clump range is split over blocks whose partials are summed in a fixed order,
    so two evaluations of the same state are the same bits."""
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = tg.host_grid(host)
    rng = np.random.default_rng(6)
    pos, vel = init_plummer(1e3, 2e-3, host, m_star=1.0, seed=3)
    n = 5000
    xyz = rng.normal(0., 0.05, (n, 3))
    g = tg.TabMGpu(pos.shape[0], n, 1, D_GRID, log_r, M_enc)
    g.set_stars(pos, vel)
    g.set_table(np.full((1, 96), 3e3))
    g.set_clumps(xyz, np.full(n, 3e3), np.zeros(n), np.zeros(n, np.int32))
    for fp64 in (True, False):
        a1, a2 = g.accel(fp64=fp64), g.accel(fp64=fp64)
        assert np.array_equal(a1, a2)
    g.step(1e-4, 3)
    p1, _ = g.stars()
    g.set_stars(pos, vel)
    g.step(1e-4, 3)
    assert np.array_equal(g.stars()[0], p1)
    g.close()


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_a_non_finite_star_does_not_touch_the_others():
    """A star flung to a non-finite radius reads NaN itself and no more, as on the CPU."""
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = tg.host_grid(host)
    pos, vel = init_plummer(1e3, 2e-3, host, m_star=1.0, seed=3)
    xyz = np.array([[0.005, 0., 0.], [0., -0.004, 0.001]])
    g = tg.TabMGpu(pos.shape[0], 2, 1, D_GRID, log_r, M_enc)
    g.set_table(np.full((1, 96), 2e3))
    g.set_clumps(xyz, np.full(2, 2e3), np.zeros(2), np.zeros(2, np.int32))
    g.set_stars(pos, vel)
    a_ok = g.accel(fp64=True)
    bad = pos.copy()
    bad[7] = [np.nan, 0., 0.]
    bad[8] = [np.inf, 0., 0.]
    g.set_stars(bad, vel)
    a = g.accel(fp64=True)
    keep = np.ones(pos.shape[0], bool)
    keep[[7, 8]] = False
    assert np.array_equal(a[keep], a_ok[keep])
    assert not np.isfinite(a[7]).all()
    g.close()


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_capacity_and_shape_errors_raise():
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = tg.host_grid(host)
    g = tg.TabMGpu(8, 4, 2, D_GRID, log_r, M_enc)
    with pytest.raises(ValueError):
        g.set_table(np.zeros((2, 50)))
    with pytest.raises(RuntimeError):
        g.set_table(np.zeros((3, 96)))
    with pytest.raises(ValueError):
        g.set_clumps(np.zeros((3, 3)), np.zeros(2), np.zeros(3), np.zeros(3, np.int32))
    with pytest.raises(RuntimeError):
        g.set_clumps(np.zeros((5, 3)), np.zeros(5), np.zeros(5), np.zeros(5, np.int32))
    with pytest.raises(ValueError):
        g.set_bank(np.zeros(3), np.zeros((2, 3)), np.zeros((3, 3)), np.zeros(3), [0, 3],
                   np.zeros((1, 96)), np.zeros((1, 96)), [0], [1.], [1.], np.zeros((1, 3)), [0.])
    with pytest.raises(RuntimeError):  # a single knot, refused as a spline would refuse it
        g.set_bank(np.zeros(1), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros(1), [0, 1],
                   np.zeros((1, 96)), np.zeros((1, 96)), [0], [1.], [1.], np.zeros((1, 3)), [0.])
    g.set_bank(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), [0, 0],
               np.zeros((1, 96)), np.zeros((1, 96)), [0], [1.], [1.], np.zeros((1, 3)), [0.])
    with pytest.raises(RuntimeError):  # reads an orbit the bank does not have
        g.set_copies([1], [0.], np.eye(3)[None], [np.inf])
    with pytest.raises(RuntimeError):  # more copies than rows
        g.set_copies([0, 0, 0], [0., 0., 0.], np.stack([np.eye(3)] * 3), [np.inf] * 3)
    g.close()


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_point_mass_force_is_the_original_kernels():
    """A row read at its last knot everywhere is a point mass, so against the
    softening-free _accel_cart_nb the device must agree to rounding. That kernel has no
    guard at zero separation, so the test keeps clear of it."""
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = _orig_host_grid(host)
    rng = np.random.default_rng(2)
    pos, vel = init_plummer(1e3, 2e-3, host, m_star=1.0, seed=3)
    n = 300
    mc = rng.normal(0., 0.05, (n, 3))
    a_max = _accel_cart_nb(pos, mc, 2e4, log_r, M_enc, 0.0)
    g = tg.TabMGpu(pos.shape[0], n, n, D_GRID, log_r, M_enc)
    g.set_stars(pos, vel)
    g.set_table(np.full((n, D_GRID.size), 2e4))
    g.set_clumps(mc, np.full(n, 2e4), np.zeros(n), _ident(n))
    a = g.accel(fp64=True)
    assert np.abs(a - a_max).max() <= 1e-12 * np.abs(a_max).max()
    a = g.accel(fp64=False)
    assert np.abs(a - a_max).max() <= 1e-5 * np.abs(a_max).max()
    g.close()


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_dekel_force_is_the_original_kernel_to_the_table_resolution():
    """_accel_cart_clumps_nb evaluates the Dekel M(<d) in closed form; on a fine table
    the device tracks it to the interpolation error of that table, not to rounding."""
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = _orig_host_grid(host)
    rng = np.random.default_rng(5)
    pos, vel = init_plummer(1e3, 2e-3, host, m_star=1.0, seed=3)
    n = 40
    mc = rng.normal(0., 0.004, (n, 3))
    Mh, rs, c, al = np.full(n, 1e5), np.full(n, 5e-4), np.full(n, 20.), np.full(n, 0.5)
    a_max = _accel_cart_clumps_nb(pos, mc, Mh, rs, c, al, log_r, M_enc)
    a_host = _accel_cart_clumps_nb(pos, mc, np.zeros(n), rs, c, al, log_r, M_enc)
    tab = np.stack([_dekel_M_enc_grid(np.exp(D_FINE), Mh[j], rs[j], c[j], al[j])
                    for j in range(n)])
    g = tg.TabMGpu(pos.shape[0], n, n, D_FINE, log_r, M_enc)
    g.set_stars(pos, vel)
    g.set_table(tab)
    g.set_clumps(mc, tab[:, -1], np.full(n, np.inf), _ident(n))
    a = g.accel(fp64=True)
    pair = np.abs(a_max - a_host).max()
    assert np.abs(a - a_max).max() <= 1e-4 * pair
    g.close()


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_device_stepping_is_the_original_point_mass_driver():
    """stellar_simulation_nb against the device fed the same MACHO positions step by
    step, 360 sub-steps with the clumps a few percent of the host force -- enough to
    move the median radius at O(dt) if a coefficient or the kick/drift order were wrong,
    weak enough that the dynamics do not amplify the two arithmetics' rounding within
    the run. Clumps carrying most of the force turn chaotic in tens of sub-steps."""
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = _orig_host_grid(host)
    rng = np.random.default_rng(8)
    pos, vel = init_plummer(1e3, 2e-3, host, m_star=1.0, seed=3)
    n_step, dt, n, mMA = 40, 12. / 24000, 20, 3e2
    t_knots = np.arange(n_step + 1) * dt
    xvM = _circles(rng, n, t_knots, r_lo=0.02, r_hi=0.5)
    n_sub = 9
    R0, HL, _, _, _ = stellar_simulation_nb(xvM, pos, vel, mMA, host, t_knots[1:], 2e-3,
                                            n_substeps=n_sub, earlyStop=False)
    _, HL_h, _, _, _ = stellar_simulation_nb(xvM, pos, vel, 0., host, t_knots[1:], 2e-3,
                                             n_substeps=n_sub, earlyStop=False)
    assert np.abs(HL / HL_h - 1.).max() > 1e-4  # the clumps do move the median radius
    g = tg.TabMGpu(pos.shape[0], n, n, D_GRID, log_r, M_enc)
    g.set_stars(pos, vel)
    g.set_table(np.full((n, D_GRID.size), mMA))
    got = [_r_half_origin(pos)]
    for i in range(n_step):
        g.set_clumps(_cart(xvM[:, i, :]), np.full(n, mMA), np.zeros(n), _ident(n))
        g.step(dt / n_sub, n_sub, fp64=True)
        got.append(_r_half_origin(g.stars()[0]))
    rel = np.abs(np.array(got) / HL - 1.).max()
    assert rel < 1e-9, rel
    assert np.allclose(np.array(got[1:]) * _PLUMMER_R0_FROM_RHALF, R0[1:], rtol=1e-9)
    g.close()


@pytest.mark.skipif(not GPU, reason='no nvcc')
def test_device_stepping_is_the_original_dekel_driver():
    """stellar_simulation_clumps_nb with time-varying Dekel clumps against the device
    reading a fine per-step table of the same profiles. The table's interpolation error
    (~1e-5 of the pair force) is what bounds the agreement here, not the arithmetic."""
    host = NFW(M=1e9, c=24.58)
    log_r, M_enc = _orig_host_grid(host)
    rng = np.random.default_rng(9)
    pos, vel = init_plummer(1e3, 2e-3, host, m_star=1.0, seed=3)
    n_step, dt, n = 40, 12. / 24000, 12
    t_knots = np.arange(n_step + 1) * dt
    xvM = _circles(rng, n, t_knots, r_lo=0.01, r_hi=0.05)
    M = np.linspace(2e3, 1.2e3, n_step + 1)[None, :] * np.ones((n, 1))
    c = np.full((n, n_step + 1), 20.)
    al = np.full((n, n_step + 1), 0.5)
    rh = np.full((n, n_step + 1), 1e-2)
    n_sub = 9
    _, HL, _, _, _ = stellar_simulation_clumps_nb(xvM, pos, vel, M, c, al, rh, host,
                                                  t_knots[1:], 2e-3, n_substeps=n_sub,
                                                  earlyStop=False)
    _, HL_h, _, _, _ = stellar_simulation_clumps_nb(xvM, pos, vel, 0. * M, c, al, rh, host,
                                                    t_knots[1:], 2e-3, n_substeps=n_sub,
                                                    earlyStop=False)
    assert np.abs(HL / HL_h - 1.).max() > 1e-4
    d = np.exp(D_FINE)
    g = tg.TabMGpu(pos.shape[0], n, n, D_FINE, log_r, M_enc)
    g.set_stars(pos, vel)
    got = [_r_half_origin(pos)]
    for i in range(n_step):
        tab = np.stack([_dekel_M_enc_grid(d, M[j, i], rh[j, i] / c[j, i], c[j, i], al[j, i])
                        for j in range(n)])
        g.set_table(tab)
        g.set_clumps(_cart(xvM[:, i, :]), tab[:, -1], np.full(n, np.inf), _ident(n))
        g.step(dt / n_sub, n_sub, fp64=True)
        got.append(_r_half_origin(g.stars()[0]))
    rel = np.abs(np.array(got) / HL - 1.).max()
    assert rel < 1e-6, rel
    g.close()
