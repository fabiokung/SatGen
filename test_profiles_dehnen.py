"""
Unit tests for the Dehnen (1993) profile.

The gamma = 1 member is Hernquist (1990), which profiles.py implements
independently, so that comparison exercises the general-gamma algebra end to
end. Everything else is checked against quadrature or against exact identities
(Poisson, the total-mass limit, the analytic central dispersion).

Run with:
    python -m pytest test_profiles_dehnen.py -v
"""

import numpy as np
import pytest
from scipy.integrate import quad

import config as cfg
from profiles import Dehnen, Hernquist


# UFD-scale cored host: the Segue-I setup, total mass 1e9 at a = 445.5 pc.
UFD = dict(M=1.0e9, a=0.4455)

RADII = np.logspace(-4, 2, 25)


@pytest.fixture(scope="module")
def cored():
    return Dehnen(UFD['M'], UFD['a'])


@pytest.mark.parametrize("field", ['rho', 'M', 'Phi', 'Vcirc', 'sigma'])
def test_gamma_one_is_hernquist(field):
    dh = Dehnen(1e9, 0.5, gamma=1.)
    hq = Hernquist(1e9, 0.5)
    assert getattr(dh, field)(RADII) == pytest.approx(
        getattr(hq, field)(RADII), rel=1e-12)


def test_hernquist_sigma_is_reachable():
    # regression: Hernquist.sigma must read self.GMb, not a cfg attribute
    assert np.isfinite(Hernquist(1e9, 0.5).sigma(0.5))


def test_mass_matches_density_quadrature(cored):
    for r in (1e-3, 0.05, 0.5, 5.):
        num = quad(lambda s: cfg.FourPi * s**2 * cored.rho(s), 0., r,
                   limit=200)[0]
        assert num == pytest.approx(cored.M(r), rel=1e-8)


def test_total_mass_is_finite_and_normalized(cored):
    assert cored.M(1e12) == pytest.approx(UFD['M'], rel=1e-9)
    # the r^-4 tail keeps 6% of the mass outside the overdensity radius
    assert 0.9 < cored.M(cored.rh) / UFD['M'] < 0.95


def test_poisson_identity(cored):
    # d2Phi/dr2 = 4 pi G rho - 2 G M / r^3 follows from the spherical
    # Laplacian with dPhi/dr = G M / r^2, independent of the closed form
    for r in (1e-3, 0.05, 0.5, 5., 50.):
        want = cfg.FourPiG * cored.rho(r) - 2. * cfg.G * cored.M(r) / r**3
        assert cored.d2Phidr2(r) == pytest.approx(want, rel=1e-10)


def test_fgrav_and_vcirc_agree(cored):
    for r in (1e-3, 0.05, 0.5, 5.):
        assert np.sqrt(-r * cored.fgrav(r, 0.)[0]) == pytest.approx(
            cored.Vcirc(r), rel=1e-12)


def test_potential_is_finite_at_centre_and_vanishes_far(cored):
    assert cored.Phi(1e-12) == pytest.approx(
        -cfg.G * UFD['M'] / (2. * UFD['a']), rel=1e-10)
    assert cored.Phi(1e9) == pytest.approx(0., abs=1e-6 * abs(cored.Phi(1.)))


def test_slope_runs_from_gamma_to_four():
    for g in (0., 0.5, 1., 1.5):
        h = Dehnen(1e9, 0.5, gamma=g)
        assert h.s(1e-8) == pytest.approx(g, abs=1e-6)
        assert h.s(1e8) == pytest.approx(4., abs=1e-6)


@pytest.mark.parametrize('gamma', [0., 1.])
def test_sigma_matches_jeans_quadrature(gamma):
    # both closed forms against the integral they solve. gamma = 1 is the only
    # independent check on that branch: Dehnen and Hernquist transcribe the same
    # expression, so comparing them to each other cannot catch an error in it
    h = Dehnen(UFD['M'], UFD['a'], gamma=gamma)
    for r in (1e-3, 0.05, 0.5, 5.):
        num = quad(lambda s: h.rho(s) * cfg.G * h.M(s) / s**2,
                   r, np.inf, limit=400)[0]
        assert np.sqrt(num / h.rho(r)) == pytest.approx(h.sigma(r), rel=1e-8)


def test_central_dispersion_is_GM_over_30a(cored):
    # sigma_r^2(0) = G M / (30 a) for gamma = 0
    assert cored.sigma(1e-9)**2 == pytest.approx(
        cfg.G * UFD['M'] / (30. * UFD['a']), rel=1e-8)


def test_vmax_is_at_two_a(cored):
    assert cored.rmax == pytest.approx(2. * UFD['a'], rel=1e-12)
    assert cored.Vmax == pytest.approx(cored.Vcirc(cored.rmax), rel=1e-12)
    r = np.logspace(np.log10(0.05 * UFD['a']), np.log10(50. * UFD['a']), 4001)
    assert cored.Vcirc(r).max() <= cored.Vmax * (1. + 1e-12)


def test_rh_encloses_the_overdensity(cored):
    assert cored.rhobar(cored.rh) == pytest.approx(cored.rhoh, rel=1e-9)
    # gamma = 0 has a closed form: rhobar(r) = 3M / (4 pi (r+a)^3)
    want = (3. * UFD['M'] / (cfg.FourPi * cored.rhoh))**(1. / 3.) - UFD['a']
    assert cored.rh == pytest.approx(want, rel=1e-9)


def test_rh_raises_when_no_such_radius_exists():
    # a core diffuse enough to sit below Delta rhoc everywhere
    with pytest.raises(ValueError, match="no radius encloses"):
        Dehnen(1e9, 500.)


def test_gamma_outside_range_raises():
    for g in (-0.1, 2., 3.):
        with pytest.raises(ValueError, match="gamma must be"):
            Dehnen(1e9, 0.5, gamma=g)


def test_sigma_unimplemented_gamma_raises():
    with pytest.raises(NotImplementedError, match="gamma = 0 and 1"):
        Dehnen(1e9, 0.5, gamma=0.5).sigma(0.5)


def test_arrays_and_scalars_agree(cored):
    for field in ('rho', 's', 'M', 'rhobar', 'tdyn', 'Phi', 'Vcirc', 'sigma',
                  'd2Phidr2'):
        vec = getattr(cored, field)(RADII)
        one = np.array([getattr(cored, field)(r) for r in RADII])
        assert vec == pytest.approx(one, rel=1e-14)


def test_z_coordinate_is_spherical_radius(cored):
    for field in ('rho', 'M', 'Phi', 'Vcirc', 'sigma', 'd2Phidr2'):
        f = getattr(cored, field)
        assert f(0.3, 0.4) == pytest.approx(f(0.5), rel=1e-14)
