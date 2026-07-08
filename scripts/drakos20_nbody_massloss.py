"""Digitize the Drakos+2020 N-body mass-loss curves into per-orbit CSVs.

Usage:
    python scripts/drakos20_nbody_massloss.py

Drakos, Taylor & Benson 2020 (MNRAS 494, 378; arXiv:2003.09452) publish no raw
N-body data, so we recover it from the vector figure in their arXiv e-print.
Fig. 8 (Plots/MassLossPredictionEta.pdf) plots bound mass M/Msat vs t/torb for
the ten NFW satellite orbits S1..S10; the simulation points are dense open
circles (one colour per panel, ~51 per panel) and the black lines are the three
eta mass-loss models. Each circle is a vector path, so its centre is recovered
essentially exactly as a path-vertex centroid -- this is not raster digitization.

The e-print tarball is fetched on demand into a local cache and is not committed.
Output CSVs (committed) land in etc/drakos20_nbody/.
"""
import os
import tarfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, os.pardir, "etc", "drakos20_nbody")
CACHE = os.path.join(OUTDIR, ".cache")
ARXIV_ID = "2003.09452"
FIG_MEMBER = "Plots/MassLossPredictionEta.pdf"

# Table 1 of Drakos+2020: per-orbit host/orbit parameters. Columns are the
# host/satellite mass ratio, host virial radius Rvir/rs (= 10*ratio**(1/3),
# c_host=10, equal mean density within rvir), apo/peri radii in satellite scale
# radii, tangential velocity at apo in vunit=sqrt(G*Msat/rs), radial period in
# tunit=sqrt(rs**3/G/Msat), circularity L/Lmax, relative energy, and the radius
# of a circular orbit of the same energy over Rvir. Sim 3 = "Fast", Sim 4 = "Slow".
ORBITS = {
    #      ratio  Rvir/rs  ra/rs  rp/rs  va/vunit  torb/tunit  circ   eta_rel  Rc/Rvir
    "S1":  (100,  46.4,    100,   10,    0.34,     206.8,      0.42,  0.85,    1.26),
    "S2":  (100,  46.4,    100,   50,    0.90,     299.4,      0.92,  0.71,    1.63),
    "S3":  (300,  66.9,    100,   10,    0.51,     129.7,      0.40,  1.09,    0.88),
    "S4":  (300,  66.9,    100,   50,    1.42,     185.4,      0.92,  0.92,    1.13),
    "S5":  (100,  46.4,    500,   50,    0.23,     1778.5,     0.47,  0.24,    6.14),
    "S6":  (300,  66.9,    25,    10,    1.50,     31.48,      0.82,  2.29,    0.26),
    "S7":  (50,   36.8,    80,    5,     0.19,     201.6,      0.30,  0.86,    1.24),
    "S8":  (50,   36.8,    90,    15,    0.37,     259.7,      0.58,  0.76,    1.48),
    "S9":  (10,   21.5,    40,    10,    0.30,     196.8,      0.71,  0.88,    1.19),
    "S10": (10,   21.5,    25,    10,    0.42,     123.2,      0.85,  1.14,    0.82),
}
ORBIT_COLS = ("Mhost_over_Msat", "Rvir_over_rs", "ra_over_rs", "rp_over_rs",
              "va_over_vunit", "torb_over_tunit", "circularity", "eta_rel", "Rc_over_Rvir")

# initial-condition particle count (Drakos+2020 Sec. 3), used for the Poisson
# bound-count shot-noise floor 1/sqrt(N0 * m/Msat).
N0 = 1_286_991


def fetch_figure():
    """Return the Fig. 8 PDF bytes, downloading + caching the arXiv e-print."""
    os.makedirs(CACHE, exist_ok=True)
    tarpath = os.path.join(CACHE, ARXIV_ID + ".tar.gz")
    if not os.path.exists(tarpath):
        url = "https://arxiv.org/e-print/" + ARXIV_ID
        print("downloading", url)
        req = urllib.request.Request(url, headers={"User-Agent": "SatGen digitizer"})
        with urllib.request.urlopen(req) as r, open(tarpath, "wb") as f:
            f.write(r.read())
    with tarfile.open(tarpath) as tf:
        member = tf.extractfile(FIG_MEMBER)
        if member is None:
            raise RuntimeError("%s not found in %s e-print" % (FIG_MEMBER, ARXIV_ID))
        return member.read()


def digitize(pdf_bytes):
    """Recover per-orbit (t/torb, M/Msat) from the vector circles in Fig. 8."""
    import fitz  # pymupdf; only needed to regenerate, not to read the CSVs

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pg = doc[0]
    words = pg.get_text("words")
    cx = lambda w: (w[0] + w[2]) / 2
    cy = lambda w: (w[1] + w[3]) / 2

    # x calibration: the '0' and '2' tick labels repeat once per column (5 cols).
    zeros = sorted(cx(w) for w in words if w[4] == "0" and cy(w) > 250)
    twos = sorted(cx(w) for w in words if w[4] == "2" and cy(w) > 250)
    xscale = float(np.mean([(t - z) / 2 for z, t in zip(zeros, twos)]))

    # y calibration: linear fit of the '0.0'..'1.0' tick labels, per row.
    def yfit(row_sel):
        d = {}
        for w in words:
            if w[4] in ("0.0", "0.2", "0.4", "0.6", "0.8", "1.0") and row_sel(cy(w)):
                d.setdefault(w[4], []).append(cy(w))
        ys = np.array([float(k) for k in d])
        px = np.array([np.mean(d[k]) for k in d])
        return np.polyfit(px, ys, 1)

    ytop, ybot = yfit(lambda y: y < 130), yfit(lambda y: 130 < y < 260)

    # panel titles S1..S10 sit above each column; read them to map grid->name.
    titles = [w for w in words if w[4].startswith("S") and w[4][1:].isdigit()]
    grid = {}  # (col, 'top'/'bot') -> name
    for w in titles:
        col = int(np.argmin([abs(cx(w) - z) for z in zeros]))
        row = "top" if cy(w) < 130 else "bot"
        grid[(col, row)] = w[4]

    # simulation circles: stroked (unfilled) coloured paths, one colour per panel.
    groups = {}
    for d in pg.get_drawings():
        if d["type"] != "s" or not d["color"] or d["color"] == (0.0, 0.0, 0.0):
            continue
        pts = [(p.x, p.y) for it in d["items"] for p in it[1:]
               if isinstance(p, fitz.Point)]
        if pts:
            a = np.array(pts)
            groups.setdefault(tuple(round(c, 3) for c in d["color"]), []).append(
                (a[:, 0].mean(), a[:, 1].mean()))

    data = {}
    for cs in groups.values():
        cs = np.array(cs)
        col = int(np.argmin([abs(cs[:, 0].min() - z) for z in zeros]))
        row = "top" if cs[:, 1].mean() < 130 else "bot"
        A = ytop if row == "top" else ybot
        t = (cs[:, 0] - zeros[col]) / xscale
        M = A[0] * cs[:, 1] + A[1]
        o = np.argsort(t)
        data[grid[(col, row)]] = (t[o], M[o])

    # the pre-first-pericentre plateau is the initial mass: anchor it to 1.0 to
    # remove a small (~0.8%) constant offset in the tick-based y calibration.
    offset = np.mean([M[0] for _, M in data.values()]) - 1.0
    return {k: (t, M - offset) for k, (t, M) in data.items()}


def write_csvs(data):
    os.makedirs(OUTDIR, exist_ok=True)
    for name in sorted(data, key=lambda s: int(s[1:])):
        t, M = data[name]
        # Poisson shot noise on the bound-particle count, floored at 2% for the
        # definitional ambiguity in "bound mass" (temporarily-bound outer material,
        # Drakos+2020 Sec. 6.2).
        sig = np.maximum(1.0 / np.sqrt(N0 * np.clip(M, 1e-4, None)), 0.02)
        path = os.path.join(OUTDIR, "mbound_%s.csv" % name)
        with open(path, "w") as f:
            f.write("# Drakos+2020 (arXiv:%s) Fig. 8 orbit %s: bound mass vs time\n" % (ARXIV_ID, name))
            f.write("# digitized from the vector simulation circles; M/Msat anchored to 1 at t=0\n")
            f.write("t_over_torb,mbound_over_msat,sigma_rel\n")
            for ti, mi, si in zip(t, M, sig):
                f.write("%.4f,%.4f,%.6e\n" % (ti, mi, si))
        print("wrote", os.path.relpath(path, os.path.join(HERE, os.pardir)), "(%d pts)" % len(t))

    path = os.path.join(OUTDIR, "orbits.csv")
    with open(path, "w") as f:
        f.write("# Drakos+2020 (arXiv:%s) Table 1: host + orbit parameters (satellite units)\n" % ARXIV_ID)
        f.write("sim," + ",".join(ORBIT_COLS) + "\n")
        for name in sorted(ORBITS, key=lambda s: int(s[1:])):
            f.write(name + "," + ",".join("%g" % v for v in ORBITS[name]) + "\n")
    print("wrote", os.path.relpath(path, os.path.join(HERE, os.pardir)))


if __name__ == "__main__":
    data = digitize(fetch_figure())
    for name in sorted(data, key=lambda s: int(s[1:])):
        t, M = data[name]
        print("%-4s n=%2d  t=[%.2f,%.2f]  M/Msat: %.3f -> %.3f"
              % (name, len(t), t.min(), t.max(), M[0], M[-1]))
    write_csvs(data)
