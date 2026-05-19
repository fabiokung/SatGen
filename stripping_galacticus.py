# Decoupled-ledger tidal evolution (Mode B) -- a one-off experiment kept
# separate from stripping_common.evolve_heating.
#
# Unlike evolve_heating, here heating is the *sole* reshaper of the density
# profile: heat_profile (Du+24 monotonic shell expansion) runs every step and
# the profile is never hard-cut or rebuilt from the bound mass. The King62 /
# Zentner+05 bound mass `m` is a separate scalar ODE -- it feeds dynamical
# friction and the tidal-track x-axis but never reshapes the profile. The
# infall subhalo carries a Kazantzidis+06 exponential tail at r_vir.
#
# Decoupling the bound-mass ledger from the profile structure follows the
# Du+24 modelling approach (and is inspired by Galacticus).

import numpy as np
from scipy.optimize import brentq

import config as cfg
from profiles import tdyn
from orbit import orbit
import evolve as ev
from subhalo_functions import heat_profile, tidalTensor
from stripping_common import _HeatingStepper, EvolutionResult


def _bound_omega_p(profile, m_bound, r_vir):
    """Angular frequency at the *bound* half-mass radius, for the adiabatic
    heating correction.

    The half-mass is taken of min(boundMass, M_profile(<r_vir)) -- the heated
    profile's own half-mass radius would be wrong here because the profile
    mass and the King62 bound mass are decoupled. f_DM = 1: the SatGen
    subhalo is pure DM.
    """
    M_within_rvir = float(profile.M(min(r_vir, profile.rh)))
    m_half = 0.5 * min(m_bound, M_within_rvir)
    M_inner = float(profile.M(profile.ri[0]))
    if m_half <= M_inner:
        r_half = float(profile.ri[0])
    elif m_half >= float(profile.Mh):
        r_half = float(profile.rh)
    else:
        r_half = brentq(lambda r: float(profile.M(r)) - m_half,
                        profile.ri[0], profile.rh, xtol=1e-8)
    return np.sqrt(cfg.G * float(profile.M(r_half)) / max(r_half, profile.ri[0])**3)


def evolve_galacticus(host, numProfile0, xv0, r_vir, tmax=37., Nstep=37000,
                      epsh=0.0741, gamma=0., alpha=3.93, beta_h=0.278,
                      second_order=False, f2=0.547, chi_v=-0.333,
                      t_dyn_mode='sub_lt', n_snapshots=10, label=None):
    """Decoupled-ledger evolution: heating reshapes the profile, the King62
    bound mass is a separate scalar that never reshapes it.

    numProfile0 is the infall subhalo profile -- an NFW with a Kazantzidis+06
    exponential tail at r_vir (build it with subhalo_functions.truncate_-
    kazantzidis(nfw, r_vir, r_decay=r_vir, slope=...)). r_vir is the infall
    virial radius: the bound mass starts at M(<r_vir) (the tail mass beyond it
    is unbound bookkeeping) and r_vir caps the adiabatic half-mass.

    Defaults are Du+24 Table IV (NFW host, gamma=1). second_order=True adds the
    Benson+Du22 second-order term. Returns an EvolutionResult; the tidal-track
    x-axis is m / M(<r_vir).

    t_dyn_mode sets only the King62 strip timescale T (Du+24 eq. 35):
    'host' -> T = tdyn(host, r_sub); 'sub_lt' -> T = T_dyn,sub(lt). The
    tidal-tensor cumulant decay (eq. 39) always uses the host orbital time.
    """
    if label is None:
        label = 'decoupled 1st+2nd' if second_order else 'decoupled 1st-order'
    assert cfg.Mres is not None, "cfg.Mres must be set before calling evolve_galacticus"
    potential = host
    timesteps = np.linspace(0., tmax, Nstep + 1)[1:]

    numProfile = numProfile0
    rmax0 = numProfile.rmax
    vmax0 = numProfile.Vmax

    snap_npts = 100
    track_steps = np.round(np.linspace(0, Nstep - 1, n_snapshots)).astype(int)

    t_arr = np.full(Nstep, np.nan)
    r_arr = np.full(Nstep, np.nan)
    m_arr = np.full(Nstep, np.nan)
    vmax_arr = np.full(Nstep, np.nan)
    rmax_arr = np.full(Nstep, np.nan)
    lt_arr = np.full(Nstep, np.nan)
    r_grids = np.zeros((n_snapshots, snap_npts))
    rho_snaps = np.zeros((n_snapshots, snap_npts))
    M_snaps = np.zeros((n_snapshots, snap_npts))

    o = orbit(xv0)
    r = np.sqrt(xv0[0]**2 + xv0[2]**2)
    # bound mass starts at M(<r_vir) = M_vir; the Kazantzidis tail beyond r_vir
    # is finite-mass regularization, not part of the bound ledger
    m = float(numProfile.M(min(r_vir, numProfile.rh)))
    lt = cfg.Rres
    tprevious = 0.
    tt_int = np.zeros((3, 3))

    heater = _HeatingStepper(numProfile, second_order, f2=f2, chi_v=chi_v)

    for i, t in enumerate(timesteps):
        dt = t - tprevious
        if r <= cfg.Rres:
            tprevious = t
            continue
        o.integrate(t, potential, m)
        xv = o.xv
        r = np.sqrt(xv[0]**2 + xv[2]**2)
        V = np.sqrt(xv[3]**2 + xv[4]**2 + xv[5]**2)
        x = xv[0] * np.cos(xv[1])
        y = xv[0] * np.sin(xv[1])
        z_c = xv[2]
        t_orb = tdyn(potential, r)
        if dt > 0.5 * t_orb:
            raise ValueError(
                f"dt/t_dyn = {dt/t_orb:.2f} > 0.5 at r={r:.1f} kpc, t={t:.2f} Gyr; increase Nstep"
            )

        tt_cur = tidalTensor(potential, [x, y, z_c])
        # tidal-tensor cumulant always decays on the host orbital time
        # (Du+24 eq. 39); t_dyn_mode only sets the King62 strip timescale.
        tt_int += (tt_cur - beta_h * tt_int / t_orb) * dt
        T_shock = r / V if V > 0. else 1e10
        # adiabatic omega_p from the bound half-mass radius, not
        # numProfile.omega_p -- the profile mass and m are decoupled here
        omega_p = _bound_omega_p(numProfile, m, r_vir)
        adiabatic = (1. + (omega_p * T_shock)**2)**(-gamma)
        tidalHR = (epsh / 3) * adiabatic * np.sum(tt_cur * tt_int)

        eps_r, should_reset = heater.step(dt, tidalHR, r, t, t_orb)

        if m <= cfg.Mres:
            m, lt = cfg.Mres, cfg.Rres
        else:
            try:
                # heating is the sole profile reshaper -- no hard cut
                numProfile = heat_profile(numProfile, eps_r)
            except RuntimeError:
                # heat_profile raises with <=2 bound shells: heating alone has
                # unbound the subhalo. Treat as destruction.
                m, lt = cfg.Mres, cfg.Rres
            else:
                lt = float(ev.ltidal(numProfile, potential, xv, 'King62'))  # type: ignore[arg-type]
                if lt < numProfile.rh:
                    # Zentner+05 King62 rate: dm/dt = alpha (m - M(<lt)) / T.
                    # The numerator mixes the King62 scalar m with the heated
                    # profile's M(<lt) -- a hybrid of the two channels.
                    M_at_lt = min(float(numProfile.M(lt)), float(numProfile.Mh))
                    T_strip = t_orb
                    if t_dyn_mode == 'sub_lt':
                        T_strip = (np.pi / 2.) * (
                            np.sqrt(lt**3 / (cfg.G * M_at_lt)) if M_at_lt > 0.
                            else numProfile.rmax / numProfile.Vmax
                        )
                    dm = max(alpha * (m - M_at_lt) * dt / T_strip, 0.)
                    m = max(m - dm, cfg.Mres)

        if should_reset:
            heater.reset(numProfile, t)

        t_arr[i] = t
        r_arr[i] = r
        m_arr[i] = m
        vmax_arr[i] = numProfile.Vmax
        rmax_arr[i] = numProfile.rmax
        lt_arr[i] = lt

        slot = np.searchsorted(track_steps, i)
        if slot < n_snapshots and track_steps[slot] == i:
            rg = np.logspace(np.log10(numProfile.ri[0]),
                             np.log10(numProfile.rh), snap_npts)
            r_grids[slot] = rg
            rho_snaps[slot] = numProfile.rho(rg)
            M_snaps[slot] = np.maximum.accumulate(numProfile.M(rg))

        tprevious = t

    return EvolutionResult(
        t=t_arr, r=r_arr, m=m_arr, vmax=vmax_arr, rmax=rmax_arr, lt=lt_arr,
        r_grid=r_grids, rho_snapshots=rho_snaps, M_snapshots=M_snaps,
        snapshot_steps=track_steps, rmax0=rmax0, vmax0=vmax0, label=label,
    )
