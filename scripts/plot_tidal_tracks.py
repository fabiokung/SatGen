"""
Compare evolve.py tidal track implementations against reference values
from Penarrubia+10 and Errani, Penarrubia & Walker 2018.

Usage:
    python scripts/plot_tidal_tracks.py [--save]

Without --save: shows interactive plot (requires display).
With --save: writes FIGURE/tidal_tracks.png (creates FIGURE/ if needed).
"""

import sys
import os
import numpy as np
import matplotlib
if '--save' in sys.argv or os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent dir to path so we can import SatGen modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evolve as ev


# ---------------------------------------------------------------------------
# Reference values for Penarrubia+10, Fig 5
# (Vmax/Vmax0 and rmax/rmax0 vs m/m0 for NFW alpha=1 and cuspy alpha=1.5)
#
# Computed analytically from the P10 formula (Eq. 3-4 with Table 3 params):
#   g_v(x) = (2/(1+x))^mu_v * x^eta_v
#   g_r(x) = (2/(1+x))^mu_r * x^eta_r
# NFW (alpha=1):   mu_v=0.4, eta_v=0.3, mu_r=-0.3, eta_r=0.4
# Cuspy (alpha=1.5): mu_v=0.4, eta_v=0.24, mu_r=0.0,  eta_r=0.48
# ---------------------------------------------------------------------------

# NFW (alpha=1) reference points
P10_NFW_mfrac = np.array([1.0, 0.5, 0.1, 0.05, 0.01])
P10_NFW_gvmax = np.array([1.0000, 0.9113, 0.6366, 0.5268, 0.3301])
P10_NFW_grmax = np.array([1.0000, 0.6954, 0.3328, 0.2488, 0.1291])

# Cuspy (alpha=1.5) reference points
P10_cuspy_mfrac = np.array([1.0, 0.5, 0.1, 0.05, 0.01])
P10_cuspy_gvmax = np.array([1.0000, 0.9499, 0.7309, 0.6311, 0.4352])
P10_cuspy_grmax = np.array([1.0000, 0.7172, 0.3311, 0.2373, 0.1097])

# ---------------------------------------------------------------------------
# Reference values from Errani, Penarrubia & Walker 2018, Fig 3
# (l_eff/l_eff0 and m_star/m_star0 vs m_max/m_max0, alpha=1, lefflmax=0.1)
#
# IMPORTANT: The EPW18 l_eff track exhibits "tidal puffing" — the stellar
# effective radius first EXPANDS (ratio > 1) before being stripped at very
# small bound mass fractions. This is the correct physical behavior per
# the EPW18 paper. Reference values below reflect this.
# ---------------------------------------------------------------------------
EPW18_mmax_frac = np.array([1.0, 0.5, 0.1, 0.01])
# l_eff can exceed 1 ("tidal puffing") — approx from EPW18 Fig 3
EPW18_gleff    = np.array([1.0, 1.05, 1.16, 1.07])
# m_star is always stripped monotonically — approx from EPW18 Fig 3
EPW18_gmstar   = np.array([1.0, 0.97, 0.84, 0.55])


def plot_P10(ax_vmax, ax_rmax):
    xs = np.logspace(-3, 0, 200)

    # Our implementation
    gvs_nfw = np.array([ev.g_P10(x, alpha=1.0)[0] for x in xs])
    gls_nfw = np.array([ev.g_P10(x, alpha=1.0)[1] for x in xs])
    gvs_cuspy = np.array([ev.g_P10(x, alpha=1.5)[0] for x in xs])
    gls_cuspy = np.array([ev.g_P10(x, alpha=1.5)[1] for x in xs])

    ax_vmax.loglog(xs, gvs_nfw, 'b-', label='This code, α=1 (NFW)', lw=2)
    ax_vmax.loglog(xs, gvs_cuspy, 'r-', label='This code, α=1.5', lw=2)
    ax_vmax.loglog(P10_NFW_mfrac, P10_NFW_gvmax, 'bo', ms=8, label='P10 ref, α=1')
    ax_vmax.loglog(P10_cuspy_mfrac, P10_cuspy_gvmax, 'rs', ms=8, label='P10 ref, α=1.5')
    ax_vmax.set_xlabel('m(t)/m(0)')
    ax_vmax.set_ylabel('Vmax(t)/Vmax(0)')
    ax_vmax.set_title('Penarrubia+10 tidal tracks: Vmax')
    ax_vmax.legend(fontsize=9)
    ax_vmax.set_xlim(1e-3, 1.2)
    ax_vmax.set_ylim(0.1, 1.2)

    ax_rmax.loglog(xs, gls_nfw, 'b-', lw=2)
    ax_rmax.loglog(xs, gls_cuspy, 'r-', lw=2)
    ax_rmax.loglog(P10_NFW_mfrac, P10_NFW_grmax, 'bo', ms=8)
    ax_rmax.loglog(P10_cuspy_mfrac, P10_cuspy_grmax, 'rs', ms=8)
    ax_rmax.set_xlabel('m(t)/m(0)')
    ax_rmax.set_ylabel('rmax(t)/rmax(0)')
    ax_rmax.set_title('Penarrubia+10 tidal tracks: rmax')
    ax_rmax.set_xlim(1e-3, 1.2)
    ax_rmax.set_ylim(0.03, 1.2)


def plot_EPW18(ax_leff, ax_mstar):
    xs = np.logspace(-3, 0, 200)

    gls = np.array([ev.g_EPW18(x, alpha=1.0, lefflmax=0.1)[0] for x in xs])
    gms = np.array([ev.g_EPW18(x, alpha=1.0, lefflmax=0.1)[1] for x in xs])

    ax_leff.loglog(xs, gls, 'g-', label='This code, α=1', lw=2)
    ax_leff.loglog(EPW18_mmax_frac, EPW18_gleff, 'g^', ms=8, label='EPW18 ref')
    ax_leff.set_xlabel('m_max(t)/m_max(0)')
    ax_leff.set_ylabel('l_eff(t)/l_eff(0)')
    ax_leff.set_title('EPW18 tidal tracks: stellar size')
    ax_leff.legend(fontsize=9)
    ax_leff.set_xlim(1e-3, 1.2)
    ax_leff.set_ylim(0.05, 1.2)

    ax_mstar.loglog(xs, gms, 'm-', label='This code, α=1', lw=2)
    ax_mstar.loglog(EPW18_mmax_frac, EPW18_gmstar, 'm^', ms=8, label='EPW18 ref')
    ax_mstar.set_xlabel('m_max(t)/m_max(0)')
    ax_mstar.set_ylabel('m_star(t)/m_star(0)')
    ax_mstar.set_title('EPW18 tidal tracks: stellar mass')
    ax_mstar.legend(fontsize=9)
    ax_mstar.set_xlim(1e-3, 1.2)
    ax_mstar.set_ylim(0.1, 1.2)


def check_accuracy():
    """Print numerical comparison between our implementation and reference points."""
    print("\nP10 accuracy check (NFW, alpha=1):")
    print(f"  {'m/m0':>8}  {'code Vmax':>10}  {'ref Vmax':>10}  {'rel err':>8}")
    for x_ref, gv_ref in zip(P10_NFW_mfrac, P10_NFW_gvmax):
        gv, _ = ev.g_P10(x_ref, alpha=1.0)
        err = abs(gv - gv_ref) / gv_ref
        flag = '  <-- >10% deviation' if err > 0.10 else ''
        print(f"  {x_ref:>8.3f}  {gv:>10.4f}  {gv_ref:>10.4f}  {err:>7.1%}{flag}")

    print("\nEPW18 accuracy check (alpha=1, lefflmax=0.1):")
    print(f"  {'m_max':>8}  {'code leff':>10}  {'ref leff':>10}  {'rel err':>8}")
    for x_ref, gl_ref in zip(EPW18_mmax_frac, EPW18_gleff):
        gl, _ = ev.g_EPW18(x_ref, alpha=1.0, lefflmax=0.1)
        err = abs(gl - gl_ref) / gl_ref
        flag = '  <-- >10% deviation' if err > 0.10 else ''
        print(f"  {x_ref:>8.3f}  {gl:>10.4f}  {gl_ref:>10.4f}  {err:>7.1%}{flag}")


def main():
    check_accuracy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Tidal track validation', fontsize=14)

    plot_P10(axes[0, 0], axes[0, 1])
    plot_EPW18(axes[1, 0], axes[1, 1])

    plt.tight_layout()

    if '--save' in sys.argv:
        os.makedirs('FIGURE', exist_ok=True)
        out = 'FIGURE/tidal_tracks.png'
        plt.savefig(out, dpi=150)
        print(f"\nSaved to {out}")
    else:
        plt.show()


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    main()
