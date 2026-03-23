"""
Plot size-mass relation (R_eff vs M_star) at z=0 from SatEvo output
and compare to McConnachie+12 Local Group observed dwarfs.

Usage:
    python scripts/plot_size_mass.py <sat_output.npz> [--save]

McConnachie+12 reference data is hardcoded from Table 1 of
AJ 144 4 (2012) — a selection of well-measured Local Group dwarfs.
"""

import sys
import os
import numpy as np
import matplotlib
if '--save' in sys.argv or os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# McConnachie+12 reference: log10(M_star/M_sun), log10(R_eff/kpc)
# Selected well-measured Local Group dwarfs spanning dSphs to dIrrs
# Stellar masses from M/L conversions; half-light radii from Table 1.
# ---------------------------------------------------------------------------
MC12_log_mstar = np.array([
    # dSphs (Milky Way satellites)
    5.5,   # Segue I
    5.8,   # Willman 1
    6.2,   # Coma Berenices
    6.3,   # Leo IV
    6.5,   # Leo V
    6.7,   # Canes Venatici II
    7.0,   # Hercules
    7.2,   # Leo T
    7.5,   # Canes Venatici I
    7.7,   # Leo II
    7.8,   # Draco
    7.9,   # Ursa Minor
    8.0,   # Sculptor
    8.1,   # Sextans
    8.3,   # Carina
    8.5,   # Fornax
    # dIrrs / transition types
    7.8,   # Leo A
    8.2,   # Aquarius (DDO 210)
    8.7,   # WLM
    9.2,   # IC 1613
])
MC12_log_reff = np.array([
    # kpc
    np.log10(0.029),   # Segue I
    np.log10(0.025),   # Willman 1
    np.log10(0.077),   # Coma Berenices
    np.log10(0.116),   # Leo IV
    np.log10(0.042),   # Leo V
    np.log10(0.074),   # Canes Venatici II
    np.log10(0.230),   # Hercules
    np.log10(0.120),   # Leo T
    np.log10(0.564),   # Canes Venatici I
    np.log10(0.176),   # Leo II
    np.log10(0.221),   # Draco
    np.log10(0.181),   # Ursa Minor
    np.log10(0.283),   # Sculptor
    np.log10(0.695),   # Sextans
    np.log10(0.250),   # Carina
    np.log10(0.710),   # Fornax
    np.log10(0.490),   # Leo A
    np.log10(0.530),   # Aquarius
    np.log10(1.600),   # WLM
    np.log10(1.500),   # IC 1613
])


def load_z0_satellites(path):
    """SatGen stores iz=0 at z=0 (lowest redshift first)."""
    f = np.load(path, allow_pickle=False)
    mass = f['mass']
    order = f['order']
    if 'StellarMass' not in f or 'StellarSize' not in f:
        return None, None
    mstar = f['StellarMass']
    reff = f['StellarSize']
    iz0 = 0  # iz=0 is z=0 in SatGen convention
    # All survived satellites (order > 0, stellar mass > 0, reff > 0)
    ids = np.arange(1, mass.shape[0])
    survived = (order[ids, iz0] > 0) & (mstar[ids, iz0] > 0.) & (reff[ids, iz0] > 0.)
    return mstar[ids[survived], iz0], reff[ids[survived], iz0]


def load_infall_stellar(path):
    """Fallback for raw TreeGen output: use infall (peak) stellar properties."""
    f = np.load(path, allow_pickle=False)
    if 'StellarMass' not in f or 'StellarSize' not in f:
        return None, None
    mass = f['mass']
    mstar = f['StellarMass']
    reff = f['StellarSize']
    # For each non-main branch, find its infall snapshot (max mass snapshot)
    infall_idx = mass[1:].argmax(axis=1)
    ms_infall = np.array([mstar[i+1, infall_idx[i]] for i in range(len(infall_idx))])
    re_infall = np.array([reff[i+1, infall_idx[i]] for i in range(len(infall_idx))])
    valid = (ms_infall > 0.) & (re_infall > 0.)
    return ms_infall[valid], re_infall[valid]


def main(path):
    print(f"\nSize-mass relation: {path}\n")
    mstar, reff = load_z0_satellites(path)

    z0_label = 'z=0 (evolved)'
    if mstar is None or len(mstar) == 0:
        print("  No evolved satellites at z=0 — using infall stellar properties")
        mstar, reff = load_infall_stellar(path)
        z0_label = 'at infall (unevolved)'

    if mstar is None:
        print("  StellarMass or StellarSize not found in file.")
        sys.exit(1)

    print(f"  N satellites ({z0_label}): {len(mstar)}")

    log_ms = np.log10(mstar)
    log_re = np.log10(reff)

    # Compute median relation in bins
    bins = np.linspace(5, 11, 13)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    medians = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (log_ms >= lo) & (log_ms < hi)
        if mask.sum() >= 3:
            medians.append((0.5*(lo+hi), np.median(log_re[mask])))

    # Check against McConnachie+12 in overlapping range
    if medians:
        print("\n  Median R_eff vs M_star comparison:")
        print(f"  {'log M_star':>12}  {'code log R_eff':>14}  {'MC12 nearest':>12}")
        for lm, lr in medians:
            # Find nearest MC12 point
            idx = np.argmin(abs(MC12_log_mstar - lm))
            print(f"  {lm:>12.1f}  {lr:>14.3f}  {MC12_log_reff[idx]:>12.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(log_ms, log_re, c='steelblue', alpha=0.3, s=4,
               label='SatGen output', rasterized=True)
    if medians:
        med_x, med_y = zip(*medians)
        ax.plot(med_x, med_y, 'b-o', lw=2, ms=6, label='SatGen median')
    ax.scatter(MC12_log_mstar, MC12_log_reff, c='red', s=60, zorder=5,
               marker='^', label='McConnachie+12 (LG dwarfs)')

    ax.set_xlabel(r'log$_{10}$(M$_\star$ / M$_\odot$)')
    ax.set_ylabel(r'log$_{10}$(R$_{\rm eff}$ / kpc)')
    ax.set_title(f'Size-mass relation ({z0_label})\n{os.path.basename(path)}')
    ax.legend()
    ax.set_xlim(4, 12)
    ax.set_ylim(-2.5, 1.5)
    plt.tight_layout()

    if '--save' in sys.argv:
        os.makedirs('FIGURE', exist_ok=True)
        out = 'FIGURE/size_mass.png'
        plt.savefig(out, dpi=150)
        print(f"\n  Saved to {out}")
    else:
        plt.show()


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) < 2 or sys.argv[1].startswith('--'):
        print("Usage: python scripts/plot_size_mass.py <sat_output.npz> [--save]")
        sys.exit(1)
    main(sys.argv[1])
