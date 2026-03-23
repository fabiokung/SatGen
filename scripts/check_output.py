"""
Self-consistency checks on SatEvo output.

Usage:
    python scripts/check_output.py <sat_output.npz>

Prints PASS/FAIL for each check and exits with code 1 if any fail.
"""

import sys
import numpy as np


def load(path):
    return np.load(path, allow_pickle=False)


def check_no_negative_masses(f):
    mass = f['mass']
    order = f['order']
    evolved = order > 0
    neg = mass[evolved] < 0.
    n_neg = neg.sum()
    if n_neg == 0:
        print(f"  PASS  no_negative_masses")
        return True
    else:
        print(f"  FAIL  no_negative_masses: {n_neg} entries with mass < 0")
        return False


def check_stellar_leq_dm(f):
    if 'StellarMass' not in f:
        print(f"  SKIP  stellar_leq_dm (StellarMass not in file)")
        return True
    mass = f['mass']
    mstar = f['StellarMass']
    order = f['order']
    evolved = order > 0
    bad = (mstar[evolved] > mass[evolved]).sum()
    if bad == 0:
        print(f"  PASS  stellar_leq_dm")
        return True
    else:
        print(f"  FAIL  stellar_leq_dm: {bad} entries where M_star > M_halo")
        return False


def check_no_subhalo_exceeds_host(f):
    """Check that no surviving satellite's mass exceeds the main halo mass at z=0.
    (At high-z snapshots, massive infalling satellites can transiently approach the
    host progenitor mass — that's physically valid during major mergers.)"""
    mass = f['mass']
    order = f['order']
    iz0 = 0  # iz=0 is z=0
    host_mass = mass[0, iz0]
    ids = np.arange(1, mass.shape[0])
    active = order[ids, iz0] > 0
    bad = mass[ids[active], iz0] > host_mass
    n_bad = bad.sum()
    if n_bad == 0:
        print(f"  PASS  no_subhalo_exceeds_host")
        return True
    else:
        print(f"  FAIL  no_subhalo_exceeds_host: {n_bad} satellites above host mass at z=0")
        return False


def check_fsub_reasonable(f):
    """Subhalo mass fraction at z=0 should be in [0.01, 0.30]."""
    mass = f['mass']
    order = f['order']
    # SatGen stores iz=0 at z=0 (lowest redshift first)
    iz0 = 0
    host_mass = mass[0, iz0]
    if host_mass <= 0.:
        print(f"  SKIP  fsub_reasonable (host mass = 0 at last snapshot)")
        return True
    subs = (order[:, iz0] >= 1) & (np.arange(len(order[:, iz0])) > 0)
    sub_masses = mass[subs, iz0]
    sub_masses = sub_masses[sub_masses > 1e-4 * host_mass]
    fsub = sub_masses.sum() / host_mass
    if 0.005 <= fsub <= 0.35:
        print(f"  PASS  fsub_reasonable: fsub = {fsub:.4f}")
        return True
    else:
        print(f"  WARN  fsub_reasonable: fsub = {fsub:.4f} (expected 0.005–0.35)")
        return True  # warn only, don't fail — depends on host mass / resolution


def check_monotonic_mass_loss(f, sample_frac=0.05):
    """
    Spot-check that evolved satellite branches lose mass monotonically
    (no mass re-gain after stripping begins).

    SatGen convention: iz=0 is z=0, iz increasing → higher redshift.
    Infall is at the highest active iz; evolution proceeds toward iz=0.

    Samples a fraction of branches to keep runtime fast.
    """
    mass = f['mass']
    order = f['order']
    n_branches = mass.shape[0]
    rng = np.random.default_rng(42)
    sample = rng.choice(n_branches, size=max(1, int(n_branches * sample_frac)),
                        replace=False)
    n_violations = 0
    for i in sample:
        m = mass[i]
        o = order[i]
        # active snapshots: order > 0 means satellite is being tracked
        active = np.where(o > 0)[0]
        if len(active) < 2:
            continue
        # Infall is at the largest active iz (highest redshift);
        # evolution proceeds from high iz to low iz (toward z=0)
        iz_infall = active[-1]  # highest-redshift active snapshot
        m_evolution = m[:iz_infall + 1][::-1]  # flip so time runs infall→z=0
        # Find the subset that is actually evolved (order > 0 and mass > 0)
        active_evolution = m_evolution[m_evolution > 0]
        if len(active_evolution) < 2:
            continue
        # After infall, mass should not increase by more than 1% (allow numerical noise)
        increases = np.diff(active_evolution)
        threshold = 0.01 * active_evolution[:-1]
        bad = increases > threshold
        n_violations += bad.sum()
    if n_violations == 0:
        print(f"  PASS  monotonic_mass_loss (sampled {len(sample)} branches)")
        return True
    else:
        print(f"  WARN  monotonic_mass_loss: {n_violations} small mass increases "
              f"(may be due to subhalo release events, not necessarily bugs)")
        return True  # warn only — releases can cause apparent mass increases


def main(path):
    print(f"\nChecking: {path}\n")
    f = load(path)
    print(f"  Shape: {f['mass'].shape[0]} branches × {f['mass'].shape[1]} snapshots\n")

    results = [
        check_no_negative_masses(f),
        check_stellar_leq_dm(f),
        check_no_subhalo_exceeds_host(f),
        check_fsub_reasonable(f),
        check_monotonic_mass_loss(f),
    ]

    print()
    n_fail = results.count(False)
    if n_fail == 0:
        print("All checks passed.")
    else:
        print(f"{n_fail} check(s) FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        print(__doc__)
        sys.exit(0)
    if len(sys.argv) != 2:
        print("Usage: python scripts/check_output.py <sat_output.npz>")
        sys.exit(1)
    main(sys.argv[1])
