"""
Plot the subhalo mass function from SatEvo output and compare to N-body
reference slope (~-1.9 from Springel+08 / Garrison-Kimmel+14).

Usage:
    python scripts/plot_shmf.py <sat_output.npz> [--save]
"""

import sys
import os
import numpy as np
import matplotlib
if '--save' in sys.argv or os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_z0_subhalos(path):
    """Return masses and host mass for all surviving subhalos at z=0.

    SatGen stores snapshots with iz=0 at z=0 (lowest redshift first).
    In raw TreeGen output, order=-99 means not yet accreted; order>0 means
    an active satellite. In SatEvo output, order>0 means survived to that snapshot.
    """
    f = np.load(path, allow_pickle=False)
    mass = f['mass']
    order = f['order']
    # iz=0 is z=0 (SatGen convention: low-z first)
    iz0 = 0
    host_mass = mass[0, iz0]
    # Subhalos: branches with order >= 1, excluding the main branch (id=0)
    ids = np.arange(1, mass.shape[0])
    sub_masses = mass[ids, iz0]
    sub_order = order[ids, iz0]
    # Keep only survived (order > 0) and resolved (mass > 0) subhalos
    survived = (sub_order > 0) & (sub_masses > 0.)
    return sub_masses[survived], host_mass


def fit_power_law(mu, N, mu_min=-4., mu_max=-1.):
    """Fit dN/d ln(mu) = A * mu^slope in a given range using linear regression."""
    mask = (mu >= 10**mu_min) & (mu <= 10**mu_max) & (N > 0)
    if mask.sum() < 3:
        return None, None
    log_mu = np.log10(mu[mask])
    log_N = np.log10(N[mask])
    coeffs = np.polyfit(log_mu, log_N, 1)
    return coeffs[0], coeffs[1]  # slope, intercept


def load_infall_subhalos(path):
    """Fallback: use infall masses for all branches (from raw TreeGen output).
    Returns (infall_masses, host_mass_at_z0).
    """
    f = np.load(path, allow_pickle=False)
    mass = f['mass']
    host_mass = mass[0, 0]  # main halo at z=0
    # Infall mass = max mass across all snapshots for each non-main branch
    sub_infall = mass[1:].max(axis=1)
    sub_infall = sub_infall[sub_infall > 0.]
    return sub_infall, host_mass


def main(path):
    print(f"\nSubhalo mass function: {path}\n")
    sub_masses, host_mass = load_z0_subhalos(path)

    z0_label = 'z=0 (evolved)'
    if len(sub_masses) < 10:
        print(f"  No evolved subhalos at z=0 (raw TreeGen input?) — using infall masses")
        sub_masses, host_mass = load_infall_subhalos(path)
        z0_label = 'at infall (unevolved)'

    print(f"  Host mass at z=0: {host_mass:.2e} M_sun")
    print(f"  N subhalos: {len(sub_masses)} ({z0_label})")

    if len(sub_masses) < 10:
        print("  Too few subhalos to fit SHMF. Run with a larger tree.")
        sys.exit(0)

    mu = sub_masses / host_mass

    # Bin in log(mu)
    bins = np.logspace(-5, 0, 30)
    counts, edges = np.histogram(mu, bins=bins)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
    dln_mu = np.log(edges[1:] / edges[:-1])
    dN_dlnmu = counts / dln_mu

    # Fit power law
    slope, intercept = fit_power_law(bin_centers, dN_dlnmu)

    if slope is not None:
        print(f"  Fitted SHMF slope: {slope:.2f}  (expected ~ -1.9)")
        if -2.2 <= slope <= -1.6:
            print(f"  PASS  SHMF slope within acceptable range [-2.2, -1.6]")
        else:
            print(f"  WARN  SHMF slope outside expected range [-2.2, -1.6]")

    # Reference slope
    mu_ref = np.logspace(-5, -0.5, 100)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(bin_centers[counts > 0], dN_dlnmu[counts > 0],
              'ko', ms=6, label='SatGen output')
    if slope is not None and intercept is not None:
        ax.loglog(mu_ref, 10**intercept * mu_ref**slope,
                  'b--', label=f'Fit: slope = {slope:.2f}', lw=2)
        ax.loglog(mu_ref, 10**intercept * mu_ref**(-1.9),
                  'r:', label='Reference slope −1.9', lw=2)
    else:
        # Normalize reference to first data point
        mask0 = counts > 0
        norm = dN_dlnmu[mask0][0] / bin_centers[mask0][0]**(-1.9)
        ax.loglog(mu_ref, norm * mu_ref**(-1.9), 'r:', label='Reference slope −1.9', lw=2)
    ax.set_xlabel('m / M_host')
    ax.set_ylabel('dN / d ln(m/M_host)')
    ax.set_title(f'Subhalo mass function ({z0_label})\n{os.path.basename(path)}')
    ax.legend()
    ax.set_xlim(1e-5, 1)
    plt.tight_layout()

    if '--save' in sys.argv:
        os.makedirs('FIGURE', exist_ok=True)
        out = 'FIGURE/shmf.png'
        plt.savefig(out, dpi=150)
        print(f"  Saved to {out}")
    else:
        plt.show()


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) < 2 or sys.argv[1].startswith('--'):
        print("Usage: python scripts/plot_shmf.py <sat_output.npz> [--save]")
        sys.exit(1)
    main(sys.argv[1])
