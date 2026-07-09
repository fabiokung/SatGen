"""Extract Du+24 N-body reference curves from the raw simulation HDF5 files.

Du, Benson, Treu et al. 2024 -- Tidal evolution of cored and cuspy dark matter
halos (arXiv:2403.09597). The raw idealized-N-body outputs ship with Galacticus
(testSuite/data/idealizedSubhaloSimulations/); the two gamma_subhalo=1 (NFW,
alpha=1 beta=3 gamma=1) orbits, R_p/R_a = 1/5 (ratio 0.2) and 1/20 (ratio 0.05),
are copied into etc/du24_nbody/. Each file carries a time series (381 samples)
of boundMass, velocityMaximum and radiusVelocityMaximum with their measurement
errors, plus the apo/peri-centre snapshot indices.

    python scripts/du24_nbody_reference.py

Writes track_{1_5,1_20}.csv (V/V0, R/R0 vs M_bound/M0) and mbound_t_{1_20,1_5}.csv
(M_bound/M0 vs t), all sampled at apocentres where the subhalo is relaxed --
V_max/R_max oscillate between peri and apo, so the clean tidal-track relation
lives at the apocentre plateaus (Du+24 Fig. 12), and the mass-loss plateaus do
too (Fig. 10). Errors are the N-body's own, propagated to the ratios.
"""
import os
import h5py
import numpy as np

DATADIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'etc',
                                       'du24_nbody'))

# ratio (R_p/R_a) -> csv tag
ORBITS = {'0.05': '1_20', '0.2': '1_5'}


def _rel(y, sy, y0, sy0):
    """Relative error of the ratio y/y0, propagating both endpoints."""
    return np.sqrt((sy / y)**2 + (sy0 / y0)**2)


def extract(ratio):
    fn = os.path.join(DATADIR, f'simulationData_xc_0.7_ratio_{ratio}'
                               f'_alpha_1.0_beta_3.0_gamma_1.0.hdf5')
    with h5py.File(fn, 'r') as f:
        apo = f['indexApoCenter'][:]
        t = f['time'][:]
        bm, bmE = f['boundMass'][:], f['boundMassError'][:]
        vm, vmE = f['velocityMaximum'][:], f['velocityMaximumError'][:]
        rm, rmE = f['radiusVelocityMaximum'][:], f['radiusVelocityMaximumError'][:]
    m0, v0, r0 = bm[0], vm[0], rm[0]
    track = np.column_stack([
        bm[apo] / m0, vm[apo] / v0, rm[apo] / r0,
        _rel(vm[apo], vmE[apo], v0, vmE[0]),
        _rel(rm[apo], rmE[apo], r0, rmE[0])])
    # mbound clock is orbit number k = t/T_r (apocentre k is the k-th orbit), the
    # period-normalized axis the model is compared on -- immune to the SatGen-vs-
    # N-body radial-period offset and the N-body's orbital decay. Recorded here so
    # nothing downstream re-derives it. The physical apocentre times live in the HDF5.
    orbit = np.arange(len(apo), dtype=float)
    mbound = np.column_stack([
        orbit, bm[apo] / m0, _rel(bm[apo], bmE[apo], m0, bmE[0])])
    return track, mbound


def main():
    for ratio, tag in ORBITS.items():
        track, mbound = extract(ratio)
        orbit = tag.replace('_', '/')
        np.savetxt(
            os.path.join(DATADIR, f'track_{tag}.csv'), track, delimiter=',',
            fmt=['%.6e', '%.6f', '%.6f', '%.6e', '%.6e'],
            header=(f'Du+24 N-body gamma=1 (alpha=1,beta=3), R_p/R_a={orbit}, '
                    'apocentre samples\n'
                    'V_max/V_max0, R_max/R_max0 vs M_bound/M0; sigmas are the '
                    "N-body's relative errors on the ratios\n"
                    'x,V_over_V0,R_over_R0,sigma_V_rel,sigma_R_rel'))
        np.savetxt(
            os.path.join(DATADIR, f'mbound_t_{tag}.csv'), mbound, delimiter=',',
            fmt=['%.1f', '%.6e', '%.6e'],
            header=(f'Du+24 N-body gamma=1, R_p/R_a={orbit}: M_bound at apocentres '
                    'vs orbit number k = t/T_r\n'
                    'sigma is the relative error on M_bound/M0\n'
                    'orbit,mbound_over_m0,sigma_rel'))
        print(f'{orbit}: {len(track)} apo track pts, {len(mbound)} mbound pts')


if __name__ == '__main__':
    main()
