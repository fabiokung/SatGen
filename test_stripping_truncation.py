"""
Integration regression: the Kazantzidis l_t tail (with the second-order term)
prevents the disruption that the hard cut produces on Du+24's radial 1/20
orbit.

The King62 rate strips on the subhalo dynamical time T_dyn,sub(l_t) while the
tidal-tensor cumulant accumulates on the host orbital time (Du+24 eqs. 35,
38-39). Under that split the hard cut still collapses the radial orbit to a
bare stub, but at ~6-8 Gyr rather than the ~4 Gyr of the older
all-subhalo-time rate.

Slower than the Tier-1 unit tests (~30 s) -- runs two short evolve_heating
integrations. Run with:
    source .venv/bin/activate
    python -m pytest test_stripping_truncation.py -v
"""

import numpy as np
import pytest

import config as cfg
from profiles import NFW
from subhalo_functions import NumericProfile
import stripping_common as sc


@pytest.fixture
def du24_setup():
    """Du+24 Section II.A setup on the radial Rp/Ra = 1/20 orbit."""
    original = cfg.Mres
    cfg.Mres = 1e3
    hNFW = NFW(1.0e12, 263.2 / 23.69)
    sat = NFW(1.0e9, 26.32 / 1.279)
    rvals = np.logspace(np.log10(cfg.Rres), np.log10(sat.rh), 200)
    M_sub = sat.M(rvals)
    xv0_20, _ = sc.make_orbit(hNFW, R0=0.7 * hNFW.rh, z0=0., eta=0.131)
    yield hNFW, rvals, M_sub, xv0_20
    cfg.Mres = original


def _run(setup, truncation, second_order=False):
    hNFW, rvals, M_sub, xv0_20 = setup
    return sc.evolve_heating(
        hNFW, NumericProfile(rvals, M_sub), xv0_20,
        tmax=12., Nstep=12000, epsh=0.0741, gamma=0., beta_h=0.278,
        alpha=3.93, t_dyn_mode='sub_lt', second_order=second_order,
        truncation=truncation)


def test_hard_cut_disrupts_on_radial_orbit(du24_setup):
    """The hard cut collapses the subhalo to the mass floor on the 1/20
    orbit. The split timescale slows this to ~6-8 Gyr; the older
    all-subhalo-time rate disrupted at ~4 Gyr."""
    res = _run(du24_setup, 'hard')
    m = res.m[np.isfinite(res.m)]
    t = res.t[np.isfinite(res.m)]
    floored = np.where(m <= cfg.Mres)[0]
    assert len(floored) > 0, "hard cut should disrupt on the 1/20 orbit"
    assert 5. < t[floored[0]] < 10.  # driver: ~6.5 Gyr (first order)


def test_kazantzidis_tail_survives(du24_setup):
    """The Kazantzidis l_t tail with the second-order term keeps the subhalo
    bound, with a finite Vmax, past the time the hard cut disrupts it."""
    res = _run(du24_setup, 'kazantzidis', second_order=True)
    m = res.m[np.isfinite(res.m)]
    assert np.all(m > cfg.Mres), "kazantzidis run should not hit the mass floor"
    late = np.isfinite(res.t) & (res.t > 9.)
    assert np.sum(late) > 0
    assert np.all(np.isfinite(res.vmax[late])), \
        "the Kazantzidis profile keeps a Vmax peak past the hard-cut disruption"
