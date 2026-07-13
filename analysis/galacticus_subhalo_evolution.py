"""Compare SatGen stripping prescriptions to a Galacticus full-evolution run.

The Galacticus run carries M_bound as a scalar ledger: the subhalo NFW (Mvir, rs)
is frozen at infall and only the bound mass evolves (King62/Zentner+2005 tidal
stripping, no tidal heating). So m_bound(t) and orbital r(t) are the direct
Galacticus truth; there is no vmax/rmax track. A reference track is derived by
mapping the bound-mass history through the Penarrubia+2010 tidal-track relation.

Per representative subhalo we hold the host static at the subhalo's infall-redshift
host values (Mvir, rs) and evolve from infall to z=0 under:
  green -- Green+19 DASH transfer function (evolve_satgen_green, alpha='conc')
  hard  -- Du+24 shell heating + hard tidal-radius truncation on the apocenter
           re-virialization engine (evolve_heating_revirial, truncation='hard',
           second order, lt='Tormen98'), at a fixed calibrated parameter set.

Units: Galacticus positions Mpc, velocities km/s -> SatGen kpc, kpc/Gyr.
Cosmic time comes straight from the snapshot outputTime attribute (Gyr); infall
time from basicTimeLastIsolated.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import h5py
import numpy as np

import config as cfg
import cosmo as co
import stripping_common as sc
from model_params import CHI_V
from profiles import NFW
from subhalo_functions import NumericProfile

HDF5 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'galacticus_subhalos.hdf5')

# fields we pull per node per snapshot
_FIELDS = ['nodeIndex', 'nodeIsIsolated', 'basicMass', 'satelliteBoundMass',
           'darkMatterProfileScale', 'darkMatterOnlyRadiusVirial',
           'basicTimeLastIsolated',
           'positionOrbitalX', 'positionOrbitalY', 'positionOrbitalZ',
           'velocityOrbitalX', 'velocityOrbitalY', 'velocityOrbitalZ',
           'satellitePositionX', 'satellitePositionY', 'satellitePositionZ']


class Galacticus:
    """Snapshots ordered z-ascending (snaps[0] = z=0). t/z per snapshot in Gyr."""

    def __init__(self, path=HDF5):
        with h5py.File(path, 'r') as f:
            outs = f['Outputs']
            order = sorted(outs.keys(),
                           key=lambda k: outs[k]['nodeData']['redshift'][0])
            self.t = np.array([outs[k].attrs['outputTime'] for k in order])
            self.z = np.array([float(outs[k]['nodeData']['redshift'][0])
                               for k in order])
            self.snaps = [{d: outs[k]['nodeData'][d][()] for d in _FIELDS}
                          for k in order]
            self.t0 = float(self.t[0])  # cosmic time at z=0

    def host(self, si):
        """(Mvir, rs[kpc], rvir[kpc]) of the most massive isolated node in si."""
        s = self.snaps[si]
        iso = s['nodeIsIsolated']
        hi = int(np.argmax(np.where(iso == 1, s['basicMass'], -1.)))
        return (float(s['basicMass'][hi]),
                float(s['darkMatterProfileScale'][hi] * 1e3),
                float(s['darkMatterOnlyRadiusVirial'][hi] * 1e3))

    def first_satellite_snapshot(self, nid):
        """Index of the earliest (highest-z) snapshot where node nid is a
        satellite (nodeIsIsolated==0). None if it never is."""
        for si in range(len(self.snaps) - 1, -1, -1):
            s = self.snaps[si]
            pos = np.where(s['nodeIndex'] == nid)[0]
            if len(pos) and s['nodeIsIsolated'][pos[0]] == 0:
                return si
        return None

    def node_track(self, nid, satellite_only=True):
        """Time series for node nid: t[Gyr], z, m_bound, m_basic, r[kpc], rvir_host[kpc].

        r is the satellite-frame distance (satellitePosition), which stays continuous
        across host-node switches -- positionOrbital re-references to the immediate host
        and jumps when a progenitor merges up the tree (identical to positionOrbital for
        clean infallers). rvir_host is the current host's virial radius, for r/r_vir.
        """
        t, z, mb, mv, r, rvh = [], [], [], [], [], []
        for si, (tt, zz) in enumerate(zip(self.t, self.z)):
            s = self.snaps[si]
            pos = np.where(s['nodeIndex'] == nid)[0]
            if not len(pos):
                continue
            j = pos[0]
            if satellite_only and s['nodeIsIsolated'][j] != 0:
                continue
            t.append(tt); z.append(zz)
            mb.append(float(s['satelliteBoundMass'][j]))
            mv.append(float(s['basicMass'][j]))
            rr = np.sqrt(sum(float(s['satellitePosition%s' % a][j])**2 for a in 'XYZ'))
            r.append(rr * 1e3)
            iso = s['nodeIsIsolated']
            hi = int(np.argmax(np.where(iso == 1, s['basicMass'], -1.)))
            rvh.append(float(s['darkMatterOnlyRadiusVirial'][hi]) * 1e3)
        o = np.argsort(t)  # ascending time
        return {k: np.asarray(v)[o] for k, v in
                dict(t=t, z=z, m_bound=mb, m_basic=mv, r=r, rvir_host=rvh).items()}


def _nfw_from_data(Mvir, rs, rvir, z):
    """NFW for a Galacticus (Mvir, rs, rvir) triple, all in data units (kpc).

    Concentration c = rvir/rs is taken straight from the output
    (darkMatterOnlyRadiusVirial / darkMatterProfileScale). Delta is back-solved
    so the profile's virial radius equals rvir, which reproduces rs and rho_s
    exactly; rhoc(z) only sets that bookkeeping Delta and cancels out of the
    resulting M(r).
    """
    rhoc = co.rhoc(z, cfg.h, cfg.Om, cfg.OL)
    Delta = 3. * Mvir / (cfg.FourPi * rvir**3 * rhoc)
    return NFW(Mvir, rvir / rs, Delta=Delta, z=z)


def infall_ics(G, nid):
    """Static-host ICs at node nid's infall.

    Time axis is anchored at the true infall time (basicTimeLastIsolated, when
    m_bound==Mvir by definition); the orbit xv0 is read from the first satellite
    snapshot (earliest output with orbital coords), within ~0.5 Gyr of infall.
    """
    si = G.first_satellite_snapshot(nid)
    if si is None:
        raise ValueError(f"node {nid} is never a satellite")
    s = G.snaps[si]
    j = int(np.where(s['nodeIndex'] == nid)[0][0])
    z = float(G.z[si])
    t_infall = float(s['basicTimeLastIsolated'][j])
    Mvir = float(s['basicMass'][j])
    rs = float(s['darkMatterProfileScale'][j] * 1e3)
    rvir = float(s['darkMatterOnlyRadiusVirial'][j] * 1e3)

    x, y, zc = (float(s['positionOrbital%s' % a][j]) * 1e3 for a in 'XYZ')
    vx, vy, vz = (float(s['velocityOrbital%s' % a][j]) * cfg.kms2kpcGyr for a in 'XYZ')
    R = np.hypot(x, y)
    phi = np.arctan2(y, x)
    VR = (x * vx + y * vy) / R
    Vphi = (x * vy - y * vx) / R
    xv0 = np.array([R, phi, zc, VR, Vphi, vz])

    sub = _nfw_from_data(Mvir, rs, rvir, z)
    hMv, hrs, hrvir = G.host(si)
    host = _nfw_from_data(hMv, hrs, hrvir, z)
    return dict(si=si, z=z, t_infall=t_infall, tmax=G.t0 - t_infall,
                Mvir=Mvir, rs=rs, c_sub=sub.ch, sub=sub,
                host=host, host_Mvir=hMv, host_rs=hrs, host_c=host.ch,
                xv0=xv0, r0=float(np.hypot(R, zc)), rvir_host=host.rh)


def run_node(G, nid, params, n_snapshots=40, step_frac=0.01):
    """Evolve node nid under green + hard-truncation; return the two
    EvolutionResults, the Galacticus reference track, and the ICs.

    params: hard-truncation heating params (eps_h, gamma_h, alpha_s, beta_h, f_2)
    for the second-order shell heating with a hard tidal-radius truncation
    (lt='Tormen98'). Both engines are step_frac-adaptive; dynamical friction is on
    so the orbit is comparable to Galacticus'.
    """
    ic = infall_ics(G, nid)
    host, xv0, tmax = ic['host'], ic['xv0'], ic['tmax']

    green = sc.evolve_satgen_green(
        host, ic['Mvir'], ic['c_sub'], xv0, tmax=tmax, step_frac=step_frac,
        alpha='conc', n_snapshots=n_snapshots, label='SatGen Green/DASH')

    rvals = np.logspace(np.log10(cfg.Rres), np.log10(ic['sub'].rh), 200)
    hard = sc.evolve_heating_revirial(
        host, NumericProfile(rvals, ic['sub'].M(rvals)), xv0,
        tmax=tmax, step_frac=step_frac, truncation='hard', second_order=True,
        t_dyn_mode='sub_lt', chi_v=CHI_V, lt_choice='Tormen98',
        epsh=params['eps_h'], gamma=params['gamma_h'], alpha=params['alpha_s'],
        beta_h=params['beta_h'], f2=params['f_2'], early_terminate=True,
        label='SatGen Du+24 hard')

    gal = G.node_track(nid)
    gal['t_rel'] = gal['t'] - ic['t_infall']
    gal['m_over_m0'] = gal['m_bound'] / ic['Mvir']
    # Galacticus' first OUTPUT snapshot is already past the first pericentre (for
    # late, plunging infallers it can be tens of percent stripped). Its bound mass
    # equals Mvir at infall by definition (the bound-mass initializor sets it at
    # basicTimeLastIsolated), so prepend that exact (t_rel=0, m/m0=1) point to start
    # the mass-loss/track curves at the same place as the SatGen runs.
    gal['t_anchor'] = np.concatenate([[0.], gal['t_rel']])
    gal['m_anchor'] = np.concatenate([[1.], gal['m_over_m0']])
    # anchor the orbit at t_rel=0 too: the SatGen runs start there from xv0, so
    # prepend Galacticus' own first-snapshot radius (r0 = |xv0|, ~0.1 Gyr after
    # infall)
    gal['t_orb'] = np.concatenate([[0.], gal['t_rel']])
    gal['r_orb'] = np.concatenate([[ic['r0']], gal['r']])
    # host virial radius along the orbit (anchored at the infall-host rvir), for the
    # r/r_vir panel: it grows z_inf -> 0, mostly virial pseudo-evolution (rho_crit drop)
    gal['rvir_orb'] = np.concatenate([[ic['rvir_host']], gal['rvir_host']])
    # Galacticus' own bound-mass -> (rmax,vmax) map (Penarrubia2010 tidal track)
    gal['track_R'], gal['track_V'] = penarrubia2010_track(gal['m_anchor'])
    return dict(nid=nid, ic=ic, green=green, hard=hard, gal=gal)


def du24_nfw_track(x, mu_V=0.6175, eta_V=0.2895, mu_R=0.5529, eta_R=0.4675):
    """Du+24 eq. 19 tidal track (NFW sub on NFW host): (R/R0, V/V0) vs x=m/m0."""
    V = 2**mu_V * x**eta_V / (1.0 + x)**mu_V
    R = 2**mu_R * x**eta_R / (1.0 + x)**mu_R
    return R, V


def green_dash_track(c, M, n=120, fb_min=1e-5):
    """Green+19 DASH structural tidal track at concentration c: sweep the
    transfer function over bound fraction (no orbit) and read
    (r_max/r_max,0, V_max/V_max,0), down to the DASH grid floor.

    M is the infall halo mass. The track is nominally scale-free in M, but
    _vmax_rmax floors its root search at cfg.Rres (a fixed 0.001 kpc), so for
    small haloes r_max approaches the floor under deep stripping and the deep
    end of the normalized track shifts -- pass the actual subhalo mass, not a
    1e9 stand-in."""
    from profiles import Green
    s = Green(M, c)
    v0, r0 = sc._vmax_rmax(s)
    fb = np.logspace(0, np.log10(fb_min), n)
    R, V = np.empty(n), np.empty(n)
    for i, f in enumerate(fb):
        s.update_mass(f * M)
        v, r = sc._vmax_rmax(s)
        V[i], R[i] = v / v0, r / r0
    return fb, R, V


def penarrubia2010_track(x, mu_V=0.40, eta_V=0.30, mu_R=-0.30, eta_R=0.40):
    """Penarrubia+2010 tidal-track g(x) = 2^mu x^eta / (1+x)^mu for (R/R0, V/V0)
    vs bound fraction x = m_bound/m_infall, with the canonical NFW (gamma=1)
    exponents.

    This is the map Galacticus applies to its scalar bound-mass ledger to get a
    satellite's (r_max, V_max) when its Penarrubia2010 structural profile is
    enabled. Same family as du24_nfw_track, a re-fit of these exponents to
    NFW-on-NFW N-body.

    NB: this dataset does NOT enable that profile -- it uses a plain (frozen) NFW
    with the bound mass evolved only as a decoupled scalar (King62/Zentner+2005 tidal
    stripping). So this is how Galacticus *would* place its bound-mass history on the
    track, not structure it actually output here."""
    V = 2.0**mu_V * x**eta_V / (1.0 + x)**mu_V
    R = 2.0**mu_R * x**eta_R / (1.0 + x)**mu_R
    return R, V
