"""GPU force and Yoshida-4 stepping for the tabulated-M(<d) stellar N-body.

ctypes binding for tabm_cuda.cu. The kernels compile with nvcc into a shared library the
first time they are used on a machine (TABM_GPU_BUILD_DIR, default a temp dir), keyed by
the source hash; that needs nvcc on PATH, a host compiler nvcc accepts, and a device
visible at build time (-arch=native). Nothing is imported at module load that a CPU-only
machine lacks, and nothing is imported from stellar_heating_n_body, so the helpers below
(host_grid, clump_rows, yoshida_substeps) work anywhere.

Conventions follow stellar_simulation_tabM: positions in kpc, velocities in kpc/Gyr, the
host as a 1200-point log-linear M(<r) grid, clumps as rows of M(<d) on a uniform log d
grid, the stars' own enclosed mass through m_star_sg. Clumps enter either directly
(set_table + set_clumps, any rows, any positions) or as copies of a bank of evolved
orbits (set_bank + set_copies + advance), which is what a realization is.
"""
import ctypes
import hashlib
import os
import subprocess
import tempfile

import numpy as np
from numpy.ctypeslib import ndpointer

G_NB = 4.4985e-06        # kpc^3 Gyr^-2 Msun^-1, must match tabm_cuda.cu and stellar_heating

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(_HERE, 'tabm_cuda.cu')


def build(out_dir=None):
    """Compile tabm_cuda.cu for the device present, cached by source hash; returns the
    library's path. The cache key carries no architecture, so a build directory belongs
    to one node class."""
    out_dir = (out_dir or os.environ.get('TABM_GPU_BUILD_DIR')
               or os.path.join(tempfile.gettempdir(), 'tabm_gpu'))
    os.makedirs(out_dir, exist_ok=True)
    with open(SRC, 'rb') as fh:
        h = hashlib.sha256(fh.read()).hexdigest()[:12]
    so = os.path.join(out_dir, 'libtabm_%s.so' % h)
    if os.path.exists(so):
        return so
    tmp = '%s.%d.tmp' % (so, os.getpid())
    subprocess.run(['nvcc', '-O3', '-std=c++14', '-Xcompiler', '-fPIC', '-shared',
                    '-arch=native', '-o', tmp, SRC], check=True)
    os.replace(tmp, so)
    return so


_D = ndpointer(np.float64, flags='C_CONTIGUOUS')
_I = ndpointer(np.int32, flags='C_CONTIGUOUS')
_L = ndpointer(np.int64, flags='C_CONTIGUOUS')
_P = ctypes.c_void_p


def _c(a, dtype, shape=None):
    a = np.ascontiguousarray(a, dtype=dtype)
    return a if shape is None else a.reshape(shape)


def _lib(so):
    lib = ctypes.CDLL(so)
    c_int, c_double = ctypes.c_int, ctypes.c_double
    lib.tabm_last_error.restype = ctypes.c_char_p
    lib.tabm_last_error.argtypes = []
    lib.tabm_device_info.argtypes = [ctypes.c_char_p, c_int, ctypes.POINTER(c_int),
                                     ctypes.POINTER(c_int), ctypes.POINTER(c_int),
                                     ctypes.POINTER(ctypes.c_size_t)]
    lib.tabm_create.restype = _P
    lib.tabm_create.argtypes = [c_int, c_int, c_int, c_int, _D, _D, c_int]
    lib.tabm_destroy.restype = None
    lib.tabm_destroy.argtypes = [_P]
    lib.tabm_set_stars.argtypes = [_P, _D, _D]
    lib.tabm_get_stars.argtypes = [_P, _D, _D]
    lib.tabm_set_clumps.argtypes = [_P, c_int, _D, _D, _D, _I]
    lib.tabm_set_table.argtypes = [_P, c_int, c_int, _D, c_double, c_double]
    lib.tabm_set_bank.argtypes = [_P, c_int, ctypes.c_longlong, _D, _D, _D, _D, _L, _D, _D, _I,
                                  _D, _D, _D, _D, _D, _D]
    lib.tabm_set_copies.argtypes = [_P, c_int, _I, _D, _D, _D]
    lib.tabm_advance.argtypes = [_P, c_double, c_double, c_double, c_double, c_double]
    lib.tabm_get_clumps.argtypes = [_P, _D, _D, _D, _D, c_int]
    lib.tabm_step.argtypes = [_P, c_double, c_int, c_double, c_int]
    lib.tabm_accel.argtypes = [_P, _D, c_double, c_int]
    lib.tabm_time_force.argtypes = [_P, c_int, c_int, ctypes.POINTER(ctypes.c_float)]
    lib.tabm_sync.argtypes = [_P]
    return lib


def host_grid(host, n=1200):
    """The host M(<r) lookup stellar_simulation_tabM builds: (log r, M) on 1e-6..1e2 kpc."""
    r = np.logspace(-6, 2, n)
    return np.log(r), np.asarray(host.M(r), dtype=np.float64)


def yoshida_substeps(host, dt_outer, safety=100.):
    """stellar_simulation_tabM's sub-step count: omega_max dt_sub < 1/safety."""
    omega_max = np.sqrt(G_NB * host.Mh / host.rs**3)
    return max(1, int(np.ceil(dt_outer * omega_max * safety)))


def clump_rows(table, log_d_grid):
    """(m_out, rt2) per row: the flat-tail mass and the squared separation past which the
    row is exactly flat, so the kernel can read the last knot there instead of the table.
    The same scan the kernel runs on the rows it builds: bitwise equality from the end,
    and never below the second knot, because inside the first the CPU extrapolates the
    row and the lookup reproduces that where a point mass would not. A row still rising
    at its last knot is held there beyond the grid, as the CPU holds it."""
    table = np.asarray(table, dtype=np.float64)
    d = np.exp(np.asarray(log_d_grid, dtype=np.float64))
    flat = table == table[:, -1:]
    tail = np.where(flat.all(axis=1), table.shape[1], np.argmin(flat[:, ::-1], axis=1))
    k = np.maximum(table.shape[1] - tail, 1)
    return np.ascontiguousarray(table[:, -1]), np.ascontiguousarray(d[k] * d[k])


def device(lib=None):
    """(name, compute capability, SM count, memory bytes) of the device in use."""
    lib = lib or _lib(build())
    name = ctypes.create_string_buffer(256)
    major, minor, sms = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    mem = ctypes.c_size_t()
    if lib.tabm_device_info(name, 256, major, minor, sms, mem) != 0:
        raise RuntimeError('device_info: ' + lib.tabm_last_error().decode())
    return name.value.decode(), (major.value, minor.value), sms.value, mem.value


class TabMGpu:
    """One stellar system on the device: n_star stars against up to n_clump_cap clumps
    reading up to n_orbit_cap rows of M(<d) on log_d_grid, in the host given by
    (host_log_r, host_M_enc) from host_grid."""

    def __init__(self, n_star, n_clump_cap, n_orbit_cap, log_d_grid, host_log_r, host_M_enc,
                 so=None):
        self.lib = _lib(so or build())
        self.log_d_grid = _c(log_d_grid, np.float64)
        self.n_star, self.n_d = int(n_star), int(self.log_d_grid.size)
        self.n_clump_cap, self.n_orbit_cap = int(n_clump_cap), int(n_orbit_cap)
        log_r, M_enc = _c(host_log_r, np.float64), _c(host_M_enc, np.float64)
        self.h = self.lib.tabm_create(self.n_star, self.n_clump_cap, self.n_orbit_cap,
                                      self.n_d, log_r, M_enc, log_r.size)
        if not self.h:
            raise RuntimeError('tabm_create: ' + self._err())
        self.n_clump = 0
        self._orb = np.zeros(0, np.int32)

    def _err(self):
        return self.lib.tabm_last_error().decode()

    def _ck(self, rc, what):
        if rc != 0:
            raise RuntimeError('%s: %s' % (what, self._err()))

    def close(self):
        if getattr(self, 'h', None):
            self.lib.tabm_destroy(self.h)
            self.h = None

    def __del__(self):
        self.close()

    def set_stars(self, pos, vel):
        self._ck(self.lib.tabm_set_stars(self.h, _c(pos, np.float64, (self.n_star, 3)),
                                         _c(vel, np.float64, (self.n_star, 3))), 'set_stars')

    def stars(self):
        pos, vel = np.empty((self.n_star, 3)), np.empty((self.n_star, 3))
        self._ck(self.lib.tabm_get_stars(self.h, pos, vel), 'get_stars')
        return pos, vel

    def set_table(self, table):
        """M(<d) rows (n_orbit, n_d) on log_d_grid, for clumps set with set_clumps."""
        table = _c(table, np.float64)
        if table.ndim != 2 or table.shape[1] != self.n_d:
            raise ValueError('table %s does not match %d knots' % (table.shape, self.n_d))
        self._ck(self.lib.tabm_set_table(self.h, table.shape[0], self.n_d, table,
                                         float(self.log_d_grid[0]),
                                         float(self.log_d_grid[-1])), 'set_table')

    def set_clumps(self, xyz, m_out, rt2, orbit_of):
        """Clumps for the next steps: Cartesian xyz (n, 3), each one's outer mass and
        flat-tail rt2 (from clump_rows on its row), and the row it reads."""
        xyz = _c(xyz, np.float64, (-1, 3))
        n = xyz.shape[0]
        m_out, rt2, orb = _c(m_out, np.float64), _c(rt2, np.float64), _c(orbit_of, np.int32)
        if not (m_out.size == rt2.size == orb.size == n):
            raise ValueError('clump arrays disagree on n: %d %d %d %d'
                             % (n, m_out.size, rt2.size, orb.size))
        self._ck(self.lib.tabm_set_clumps(self.h, n, xyz, m_out, rt2, orb), 'set_clumps')
        self.n_clump, self._orb = n, orb

    def set_bank(self, kt, kp, kv, km, koff, M0, MT, hasT, m_init, m_final, spos, smass):
        """The orbits copies are read from. Knots (time, position, velocity, bound mass)
        concatenated over orbits, koff (n_orbit + 1) delimiting each; per orbit the two
        M(<d) rows on log_d_grid the bound-fraction blend runs between, whether the settled
        row is usable, the launch and final masses, and the rest position and mass of an
        orbit that has no knots."""
        kt, km = _c(kt, np.float64), _c(km, np.float64)
        kp, kv = _c(kp, np.float64, (-1, 3)), _c(kv, np.float64, (-1, 3))
        koff = _c(koff, np.int64)
        n_orbit = koff.size - 1
        M0, MT = _c(M0, np.float64, (n_orbit, self.n_d)), _c(MT, np.float64, (n_orbit, self.n_d))
        hasT = _c(hasT, np.int32)
        m_init, m_final = _c(m_init, np.float64), _c(m_final, np.float64)
        spos, smass = _c(spos, np.float64, (n_orbit, 3)), _c(smass, np.float64)
        if not (kt.size == kp.shape[0] == kv.shape[0] == km.size == koff[-1]):
            raise ValueError('knot arrays disagree: %d %d %d %d, koff ends at %d'
                             % (kt.size, kp.shape[0], kv.shape[0], km.size, koff[-1]))
        if not (hasT.size == m_init.size == m_final.size == smass.size == n_orbit):
            raise ValueError('per-orbit arrays disagree with %d orbits' % n_orbit)
        self._ck(self.lib.tabm_set_bank(self.h, n_orbit, kt.size, kt, kp, kv, km, koff, M0, MT,
                                        hasT, m_init, m_final, spos, smass, self.log_d_grid,
                                        np.exp(self.log_d_grid)), 'set_bank')

    def set_copies(self, orbit, phase, rot, t_alive):
        """The clump copies of one realization: the bank orbit each reads, its radial
        phase [Gyr], its orbital plane as a (3, 3) rotation, and the time past which it
        stops acting."""
        orbit = _c(orbit, np.int32)
        n = orbit.size
        phase, t_alive = _c(phase, np.float64), _c(t_alive, np.float64)
        rot = _c(rot, np.float64, (n, 9))
        if not (phase.size == t_alive.size == n):
            raise ValueError('copy arrays disagree on n: %d %d %d' % (n, phase.size, t_alive.size))
        self._ck(self.lib.tabm_set_copies(self.h, n, orbit, phase, rot, t_alive), 'set_copies')
        self.n_clump, self._orb = n, np.arange(n, dtype=np.int32)

    def advance(self, t_pos, tA, tB, wM, t_end):
        """Evaluate every copy for one outer step: position at t_pos, M(<d) row as the
        blend (weight wM) of its rows at the coarse times tA and tB, dead once t_alive <=
        t_end. Asynchronous, like step."""
        self._ck(self.lib.tabm_advance(self.h, float(t_pos), float(tA), float(tB), float(wM),
                                       float(t_end)), 'advance')

    def clumps(self):
        """(xyz, m_out, rt2, rows) of the clumps as the double-precision force sees them,
        rows gathered per clump."""
        n = self.n_clump
        n_rows = int(self._orb.max()) + 1 if n else 0
        xyz, m_out, rt2 = np.empty((n, 3)), np.empty(n), np.empty(n)
        rows = np.empty((n_rows, self.n_d))
        self._ck(self.lib.tabm_get_clumps(self.h, xyz, m_out, rt2, rows, n_rows), 'get_clumps')
        return xyz, m_out, rt2, rows[self._orb]

    def step(self, sub_dt, n_sub, m_star_sg=0., fp64=False):
        """n_sub Yoshida-4 sub-steps against the clumps currently set. Asynchronous: the
        launches are queued and the call returns; stars() or sync() waits for them."""
        self._ck(self.lib.tabm_step(self.h, float(sub_dt), int(n_sub), float(m_star_sg),
                                    int(bool(fp64))), 'step')

    def accel(self, m_star_sg=0., fp64=False):
        """The acceleration (n_star, 3) at the current positions, host term included."""
        out = np.empty((self.n_star, 3))
        self._ck(self.lib.tabm_accel(self.h, out, float(m_star_sg), int(bool(fp64))), 'accel')
        return out

    def time_force(self, n_repeat, fp64=False):
        """Kernel-only seconds per pair sum, averaged over n_repeat launches."""
        ms = ctypes.c_float()
        self._ck(self.lib.tabm_time_force(self.h, int(n_repeat), int(bool(fp64)), ms),
                 'time_force')
        return 1e-3 * ms.value

    def sync(self):
        self._ck(self.lib.tabm_sync(self.h), 'sync')
