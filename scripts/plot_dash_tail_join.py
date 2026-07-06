"""Plot the DASH tail-join distribution (scripts/dash_tail_join.py output).

Where does the stripped subhalo peel off its bound NFW body, in units of the
King62 tidal radius? xi = log10(r_te / l_t), with r_te the Green & van den Bosch
(2019) NFW-departure scale (their DASH-calibrated concentration-dependent
transfer function). The figure is the provenance for the tail_xi prior.
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, '..', 'etc', 'calibration_runs', 'dash_tail_join.pkl')
CHAIN_XI = 0.98          # power-law chain p50 (step 136), for contrast


def figure(recs, chain_xi=CHAIN_XI):
    xi = np.array([r['xi_rte'] for r in recs])
    ts = np.array([r['tail_slope'] for r in recs])
    fb = np.array([r['fb'] for r in recs])
    g = np.isfinite(xi)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    ax = axs[0]
    bins = np.linspace(-1.5, 0.5, 50)
    ax.hist(xi[g], bins=bins, density=True, alpha=0.55, color='C0',
            label=r'DASH $r_{\rm te}$ (Green+19)')
    ax.axvline(np.median(xi[g]), color='C0', ls='-', lw=1.5,
               label=f'DASH median ({np.median(xi[g]):+.2f})')
    ax.axvline(0, color='k', ls=':', lw=1, label=r'$r_{\rm join}=l_t$')
    ax.axvline(chain_xi, color='C3', ls='--', lw=2,
               label=f'MCMC p50 (xi={chain_xi:.2f})')
    # Du+24's own N-body (eq. 30 r_te transfer function) vs l_t, same measure
    du24 = os.path.join(os.path.dirname(PKL), 'du24_tail_join.pkl')
    if os.path.exists(du24):
        xr_du = np.concatenate([pickle.load(open(du24, 'rb'))[o]['xi_rte']
                                for o in ('1/5', '1/20')])
        ax.hist(xr_du, bins=np.linspace(-1.5, 0.5, 22), density=True,
                histtype='step', lw=2, color='C2',
                label=f'Du+24 $r_{{\\rm te}}$ ({np.median(xr_du):+.2f})')
    ax.axvspan(-1.2, 0, color='grey', alpha=0.10, label='prior (-1.2, 0)')
    ax.set_xlabel(r'$\xi = \log_{10}(r_{\rm te}/l_t)$')
    ax.set_ylabel('probability density')
    ax.set_title('DASH tidal-tail join vs King62 $l_t$')
    ax.legend(fontsize=9)

    ax = axs[1]
    hb = ax.hexbin(fb[g], xi[g], gridsize=30, bins='log', cmap='viridis',
                   extent=(0.02, 0.5, -1.5, 0.5))
    ax.axhline(0, color='w', ls=':', lw=1)
    ax.axhline(chain_xi, color='C3', ls='--', lw=2)
    ax.set_xlabel(r'$f_b$ (bound fraction)')
    ax.set_ylabel(r'$\xi = \log_{10}(r_{\rm te}/l_t)$')
    ax.set_title('join vs stripping depth')
    fig.colorbar(hb, ax=ax, label=r'$\log_{10} N$')

    fig.tight_layout()
    return fig, dict(
        n=int(g.sum()),
        med=float(np.median(xi[g])), med_slope=float(np.median(ts[np.isfinite(ts)])),
        p16=float(np.percentile(xi[g], 16)), p84=float(np.percentile(xi[g], 84)),
        p5=float(np.percentile(xi[g], 5)), p95=float(np.percentile(xi[g], 95)))


if __name__ == '__main__':
    recs = pickle.load(open(PKL, 'rb'))
    fig, stats = figure(recs)
    print(stats)
    out = os.path.join(HERE, '..', 'etc', 'calibration_runs', 'dash_tail_join.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('->', out)
