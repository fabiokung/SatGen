"""Plot the DASH tail-join distribution (scripts/dash_tail_join.py output).

Where does the N-body density profile reach the envelope slope, in units of the
King62 tidal radius? xi = log10(r_break / l_t). The figure is the provenance for
the tail_xi prior.
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, '..', 'etc', 'calibration_runs', 'dash_tail_join.pkl')
CHAIN_XI = 0.98          # power-law chain p50 (step 136), for contrast


def figure(recs, chain_xi=CHAIN_XI):
    xi5 = np.array([r['xi5'] for r in recs])
    xi4 = np.array([r['xi4'] for r in recs])
    fb = np.array([r['fb'] for r in recs])
    g5 = np.isfinite(xi5)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    ax = axs[0]
    bins = np.linspace(-3, 1.5, 60)
    ax.hist(xi4[np.isfinite(xi4)], bins=bins, alpha=0.45, color='C0',
            label=r'slope $-4$ (onset)')
    ax.hist(xi5[g5], bins=bins, alpha=0.55, color='C1',
            label=r'slope $-5$ (envelope, $\approx$ tail_n)')
    for x, c, ls in ((np.median(xi5[g5]), 'C1', '-'),
                     (np.median(xi4[np.isfinite(xi4)]), 'C0', '-')):
        ax.axvline(x, color=c, ls=ls, lw=1.5)
    ax.axvline(0, color='k', ls=':', lw=1, label=r'$r_{\rm join}=l_t$')
    ax.axvline(chain_xi, color='C3', ls='--', lw=2,
               label=f'MCMC p50 (xi={chain_xi:.2f})')
    # Du+24's own N-body (eq. 30 r_te transfer function), slope-(-5) break / l_t
    du24 = os.path.join(os.path.dirname(PKL), 'du24_tail_join.pkl')
    if os.path.exists(du24):
        import pickle as _pk
        xb = np.concatenate([_pk.load(open(du24, 'rb'))[o]['xi_brk']
                             for o in ('1/5', '1/20')])
        ax.axvspan(xb.min(), xb.max(), color='C2', alpha=0.18,
                   label=f'Du+24 N-body break ({xb.min():+.2f},{xb.max():+.2f})')
    ax.axvspan(-2, 2, color='grey', alpha=0.08, label='prior (-2, 2)')
    ax.set_xlabel(r'$\xi = \log_{10}(r_{\rm join}/l_t)$')
    ax.set_ylabel('snapshots')
    ax.set_title('DASH tidal-tail join vs King62 $l_t$')
    ax.legend(fontsize=9)

    ax = axs[1]
    hb = ax.hexbin(fb[g5], xi5[g5], gridsize=30, bins='log', cmap='viridis',
                   extent=(0.02, 0.5, -3, 1))
    ax.axhline(0, color='w', ls=':', lw=1)
    ax.axhline(chain_xi, color='C3', ls='--', lw=2)
    ax.set_xlabel(r'$f_b$ (bound fraction)')
    ax.set_ylabel(r'$\xi$ (slope $-5$)')
    ax.set_title('join vs stripping depth')
    fig.colorbar(hb, ax=ax, label=r'$\log_{10} N$')

    fig.tight_layout()
    return fig, dict(
        n=int(g5.sum()),
        med5=float(np.median(xi5[g5])), med4=float(np.median(xi4[np.isfinite(xi4)])),
        p16=float(np.percentile(xi5[g5], 16)), p84=float(np.percentile(xi5[g5], 84)),
        p5=float(np.percentile(xi5[g5], 5)), p95=float(np.percentile(xi5[g5], 95)))


if __name__ == '__main__':
    recs = pickle.load(open(PKL, 'rb'))
    fig, stats = figure(recs)
    print(stats)
    out = os.path.join(HERE, '..', 'etc', 'calibration_runs', 'dash_tail_join.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    print('->', out)
