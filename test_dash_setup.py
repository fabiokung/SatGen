"""DASH (Ogiya+19) IC matching: assert dash_nfw_setup reproduces the DASH
simulation host/subhalo NFW params and orbit exactly.

Paper prescription (arXiv:1901.08601 sec. 2.1.1): NFW host + subhalo at virial
overdensity Delta = 200*rho_crit; subhalo hard-truncated at r_vir,s (total mass
M_vir,s); mass ratio M_vir,h/M_vir,s = 1000 -> V_vir,h = M^(1/3) = 10 V_vir,s;
orbit launched at apocenter X=(r_a,0,0), V=(0, L/r_a, 0); snapshots 0.12 Gyr
apart at h=0.678 so the radial period is T_r = nTr * 0.12 Gyr.
"""
import os
import pickle

import numpy as np
import pytest

import stripping_common as sc

REPO = os.path.dirname(os.path.abspath(__file__))
IDX = pickle.load(open(os.path.join(REPO, 'etc/dash/dash_index.pkl'), 'rb'))
DASH_H, DASH_DT_SNAP = 0.678, 0.12          # paper sec. 2.1.1 / 2.1.2

# the 4 Option-A training cells + the held-out radial one
CELLS = [(12.5, 1.15, 0.7), (19.9, 1.15, 0.7), (25.0, 1.15, 0.7),
         (19.9, 0.87, 0.9), (19.9, 1.15, 0.2)]


def cell(cs, xc, eta, ch=10.0):
    for r in IDX:
        if all(abs(r[k] - v) < 1e-6 for k, v in
               (('ch', ch), ('cs', cs), ('xc', xc), ('eta', eta))):
            r = dict(r)
            r['file'] = os.path.join(REPO, r['file'])
            return r
    raise KeyError((ch, cs, xc, eta))


@pytest.mark.parametrize("cs,xc,eta", CELLS)
def test_host_subhalo_nfw_params(cs, xc, eta):
    c = cell(cs, xc, eta)
    host, sat, rvals, M_sub, _xv0, _Tr = sc.dash_nfw_setup(c)
    # concentrations are the DASH cell values, exactly
    assert host.ch == pytest.approx(c['ch'])
    assert sat.ch == pytest.approx(c['cs'])
    # both at Delta = 200 * rho_crit (DASH and SatGen default convention)
    assert host.Deltah == pytest.approx(200.)
    assert sat.Deltah == pytest.approx(200.)
    # mass ratio 1000 -> radius ratio (1/1000)^(1/3) = 0.1, V_vir ratio 10
    assert sat.rh / host.rh == pytest.approx(0.1, rel=1e-6)
    vvir_h = host.Vcirc(host.rh)
    vvir_s = sat.Vcirc(sat.rh)
    assert vvir_h / vvir_s == pytest.approx(1000.**(1. / 3.), rel=1e-6)


@pytest.mark.parametrize("cs,xc,eta", CELLS)
def test_subhalo_hard_truncated_at_rvir(cs, xc, eta):
    c = cell(cs, xc, eta)
    _host, sat, rvals, M_sub, _xv0, _Tr = sc.dash_nfw_setup(c)
    # IC is the pure NFW cut at r_vir,s: outermost knot is r_vir,s, and m_0 there
    # is M_vir,s = M_NFW(<r_vir,s). No mass sampled beyond (DASH hard truncation).
    assert rvals[-1] == pytest.approx(sat.rh, rel=1e-9)
    assert M_sub[-1] == pytest.approx(sat.M(sat.rh), rel=1e-9)
    assert M_sub[-1] == pytest.approx(sat.Mh, rel=1e-3)   # == M_vir,s


@pytest.mark.parametrize("cs,xc,eta", CELLS)
def test_orbit_matches_dash_index(cs, xc, eta):
    c = cell(cs, xc, eta)
    host, _sat, _rvals, _M_sub, xv0, _Tr = sc.dash_nfw_setup(c)
    rvir_h, vvir_h = host.rh, host.Vcirc(host.rh)
    R, _phi, z, VR, Vphi, Vz = xv0
    # launched at apocenter on the x-y plane: |r| = r_apo, v is tangential
    r_orb = np.hypot(R, z)
    assert r_orb / rvir_h == pytest.approx(c['rapo'], abs=2e-3)
    assert abs(Vphi) / vvir_h == pytest.approx(c['vtan'], abs=2e-3)
    assert abs(VR) / vvir_h < 5e-3            # radial velocity ~0 at apo
    assert abs(Vz) / vvir_h < 5e-3


@pytest.mark.parametrize("cs,xc,eta", CELLS)
def test_orbit_pericenter_and_period(cs, xc, eta):
    c = cell(cs, xc, eta)
    host, _sat, _rvals, _M_sub, xv0, T_r = sc.dash_nfw_setup(c)
    rvir_h = host.rh
    # integrate and read peri/apo against the index
    from orbit import orbit
    o = orbit(xv0)
    t = np.linspace(0., 3. * T_r, 6000)
    o.integrate(t, [host])
    r = np.hypot(o.xvArray[:, 0], o.xvArray[:, 2]) / rvir_h
    assert r.min() / r.max() == pytest.approx(c['peri_apo'], rel=0.05)
    # SatGen radial period vs DASH (nTr * 0.12 Gyr), rescaled by the h ratio
    # (T_r ~ 1/H_0 ~ 1/h; DASH h=0.678, SatGen cfg.h=0.7)
    import config as cfg
    T_r_dash = c['nTr'] * DASH_DT_SNAP * (DASH_H / cfg.h)
    assert T_r == pytest.approx(T_r_dash, rel=0.05)


def test_dash_radprof_is_truncated():
    """Data-level truth we match: DASH snap-0 enclosed mass is flat beyond
    r_vir,s (hard truncation), not the still-rising pure-NFW M(<r)."""
    c = cell(19.9, 1.15, 0.7)
    d = os.path.dirname(c['file'])
    M = np.loadtxt(os.path.join(d, 'radprof_m.txt'))
    rbins, M0 = M[0, 1:], M[1, 1:]            # r in r_vir,s, M in M_vir,s
    inside = M0[np.argmin(np.abs(rbins - 0.95))]
    beyond = M0[np.argmin(np.abs(rbins - 1.6))]
    assert beyond == pytest.approx(inside, rel=1e-3)   # flat -> truncated
    assert beyond == pytest.approx(1.0, abs=1e-3)      # total = M_vir,s
