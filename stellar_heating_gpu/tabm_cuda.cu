// Stellar N-body force and Yoshida-4 stepping on the GPU, clumps as tabulated M(<d).
// Same physics as _accel_cart_tabM_nb / _yoshida4_step_tabM_nb in
// stellar_heating_n_body/stellar_heating.py: host force from a log-linear M(<r) lookup
// plus the stars' own enclosed mass, clump force G M_j(<d) / d^2 read off a per-copy
// row on a uniform log d grid, no star-star pairs.
//
// Pair-parallel, not star-parallel. A thousand stars is eight warps, so each warp owns
// one star and its 32 lanes stride over a tile of clumps staged in shared memory, and
// grid.y splits the clump range so the grid has enough blocks to fill the device. Lane
// partials reduce by warp shuffle; block partials go to a buffer and are summed in a
// fixed order, so an evaluation is bit-reproducible -- the cache and the resume path
// treat it as a pure function of its inputs.
//
// Star state and clump data are double. Pair arithmetic is float or double by
// template; the float path casts at the staging point, so it sees exactly the float of
// each double it would have stored. Measured against the double path, the float pair
// force carries a median 1e-7 and a worst 5e-7 relative error on far clumps, 2e-7 and
// 2e-6 among the stars, and 7e-6 summed over 1e5 clumps; the tidal signal of a clump
// at 10 kpc on a star at 2 pc is 2e-4 of its monopole.
//
// Beyond a clump's flat-tail radius (d^2 >= rt2) the row is read at its last knot
// without the log and the two gathers; inside the first knot the row is extrapolated as
// the CPU extrapolates it, so a row's rt2 is never below the first knot.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

#define G_NB 4.4985e-06          // kpc^3 Gyr^-2 Msun^-1, stellar_heating._G_NB
#define STARS_PER_BLOCK 16
#define TILE 256
#define NTHREADS (STARS_PER_BLOCK * 32)

static char g_err[1024] = "";

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    snprintf(g_err, sizeof g_err, "%s (%s:%d)", cudaGetErrorString(e_), __FILE__, __LINE__); \
    return -1; } } while (0)

struct Clumps {
    double *x, *y, *z, *m, *rt2, *table;
    int *orb;
    int n, n_d;
    double log_d_min, inv_dld, inv_d_min;
};

// The evolved orbits a realization's clump copies are read from: per-step knots
// (time, Cartesian position and velocity, bound mass) concatenated over orbits, the two
// M(<d) rows the bound-fraction blend interpolates between, and the rest state of an
// orbit with no track at all.
struct Bank {
    int n_orbit;
    double *kt, *kp, *kv, *km;
    long long *koff;            // n_orbit + 1
    double *M0, *MT;            // (n_orbit, n_d)
    int *hasT;
    double *m_init, *m_final;
    double *spos, *smass;       // (n_orbit, 3), (n_orbit)
};

// One clump copy: which orbit, its radial phase, its orbital plane, and when it stops.
struct Copies {
    int n;
    int *orb;
    double *phase, *rot, *t_alive;   // rot is (n, 3, 3) row-major
};

struct TabM {
    int n_star, n_clump_cap, n_orbit_cap, n_d;
    double *pos, *vel, *acc, *part;
    int *rank, *orb;
    double *log_r, *M_enc;
    int n_rgrid;
    Clumps c;
    Bank bank;
    Copies cop;
    double *d_grid;             // exp(log_d), n_d
    double *hd;                 // host scratch, 3 * n_clump_cap
    int sms, max_split;
    cudaEvent_t e0, e1;
};

template <typename T> __device__ __forceinline__ T rsqrt_t(T x);
template <> __device__ __forceinline__ float rsqrt_t<float>(float x) { return rsqrtf(x); }
template <> __device__ __forceinline__ double rsqrt_t<double>(double x) { return rsqrt(x); }
template <typename T> __device__ __forceinline__ T log_t(T x);
template <> __device__ __forceinline__ float log_t<float>(float x) { return logf(x); }
template <> __device__ __forceinline__ double log_t<double>(double x) { return log(x); }

// Pair sum of one clump range for STARS_PER_BLOCK stars, written as this block's
// partial: part[(blockIdx.y * n_star + s) * 3 + i], times G.
template <typename T>
__global__ void __launch_bounds__(NTHREADS)
force_kernel(int n_star, const double* __restrict__ pos, double* __restrict__ part,
             Clumps c, int per_split)
{
    __shared__ T sx[TILE], sy[TILE], sz[TILE], sm[TILE], srt[TILE];
    __shared__ int sorb[TILE];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int s = blockIdx.x * STARS_PER_BLOCK + warp;
    const bool live = s < n_star;
    const int j0 = blockIdx.y * per_split;
    const int j1 = min(c.n, j0 + per_split);
    const T log_d_min = (T)c.log_d_min, inv_dld = (T)c.inv_dld, inv_d_min = (T)c.inv_d_min;
    T px = 0, py = 0, pz = 0;
    if (live) { px = (T)pos[3*s]; py = (T)pos[3*s+1]; pz = (T)pos[3*s+2]; }
    T ax = 0, ay = 0, az = 0;
    for (int t0 = j0; t0 < j1; t0 += TILE) {
        const int nt = min(TILE, j1 - t0);
        __syncthreads();
        for (int k = threadIdx.x; k < nt; k += NTHREADS) {
            sx[k] = (T)c.x[t0+k]; sy[k] = (T)c.y[t0+k]; sz[k] = (T)c.z[t0+k];
            sm[k] = (T)c.m[t0+k]; srt[k] = (T)c.rt2[t0+k]; sorb[k] = c.orb[t0+k];
        }
        __syncthreads();
        if (live) {
            for (int k = lane; k < nt; k += 32) {
                if (sm[k] == (T)0) continue;  // a dead or disrupted clump, as on the CPU
                const T dx = px - sx[k], dy = py - sy[k], dz = pz - sz[k];
                const T d2 = dx*dx + dy*dy + dz*dz;
                // negated, here and in every guard below, so a NaN skips or clamps too:
                // NaN fails every direct comparison, and int(NaN) indexes out of bounds
                if (!(d2 >= (T)1e-24)) continue;  // the CPU skips d < 1e-12 as well
                const T rinv = rsqrt_t<T>(d2);
                T M;
                if (d2 >= srt[k]) {
                    M = sm[k];
                } else {
                    const T d = d2 * rinv;
                    const T u = (log_t<T>(d) - log_d_min) * inv_dld;
                    const double* row = c.table + (size_t)sorb[k] * c.n_d;
                    if (!(u > (T)0)) {
                        const T q = d * inv_d_min;
                        M = (T)row[0] * q * q * q;
                    } else if (u >= (T)(c.n_d - 1) - (T)1e-12) {
                        M = (T)row[c.n_d - 1];
                    } else {
                        const int k0 = (int)u;
                        const T fd = u - (T)k0;
                        M = ((T)1 - fd) * (T)row[k0] + fd * (T)row[k0 + 1];
                    }
                }
                const T f = M * rinv * rinv * rinv;
                ax -= f * dx; ay -= f * dy; az -= f * dz;
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        ax += __shfl_down_sync(0xffffffffu, ax, o);
        ay += __shfl_down_sync(0xffffffffu, ay, o);
        az += __shfl_down_sync(0xffffffffu, az, o);
    }
    if (live && lane == 0) {
        double* p = part + ((size_t)blockIdx.y * n_star + s) * 3;
        p[0] = (double)ax * G_NB; p[1] = (double)ay * G_NB; p[2] = (double)az * G_NB;
    }
}

__global__ void reduce_kernel(int n_star, int ns, const double* __restrict__ part,
                              double* __restrict__ acc)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_star) return;
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int y = 0; y < ns; y++) {
        const double* p = part + ((size_t)y * n_star + s) * 3;
        ax += p[0]; ay += p[1]; az += p[2];
    }
    acc[3*s] = ax; acc[3*s+1] = ay; acc[3*s+2] = az;
}

// rank[s] = number of stars inside star s's radius, the stars' own enclosed mass in
// units of m_star_sg. Ties go to the lower index; the CPU's argsort makes no promise on
// a tie, and a tie in r^2 here can be a distinct pair of r there, so the two agree up to
// a measure-zero set. Every block stages all squared radii in shared memory once.
__global__ void rank_kernel(int n_star, const double* __restrict__ pos, int* __restrict__ rank)
{
    extern __shared__ double sr2[];
    for (int j = threadIdx.x; j < n_star; j += blockDim.x)
        sr2[j] = pos[3*j]*pos[3*j] + pos[3*j+1]*pos[3*j+1] + pos[3*j+2]*pos[3*j+2];
    __syncthreads();
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_star) return;
    const double rs = sr2[s];
    int cnt = 0;
    for (int j = 0; j < n_star; j++)
        cnt += (sr2[j] < rs) || (sr2[j] == rs && j < s);
    rank[s] = cnt;
}

__device__ __forceinline__ void host_term(const double* __restrict__ pos, int s,
        const double* __restrict__ log_r, const double* __restrict__ M_enc, int n_rgrid,
        const int* __restrict__ rank, double m_star_sg,
        double* hx, double* hy, double* hz)
{
    const double x = pos[3*s], y = pos[3*s+1], z = pos[3*s+2];
    const double r2 = x*x + y*y + z*z;
    double r = sqrt(r2);
    if (!(r > 1e-10)) r = 1e-10;
    const double log_r_min = log_r[0];
    const double inv_dlr = (n_rgrid - 1) / (log_r[n_rgrid - 1] - log_r_min);
    double t = (log(r) - log_r_min) * inv_dlr;
    if (!(t >= 0.0)) t = 0.0;
    else if (t > n_rgrid - 1 - 1e-12) t = n_rgrid - 1 - 1e-12;
    const int i0 = (int)t;
    const double frac = t - i0;
    const double M_r = (1.0 - frac) * M_enc[i0] + frac * M_enc[i0 + 1];
    const double extra = (rank != nullptr) ? m_star_sg * rank[s] : 0.0;
    const double fac = -G_NB * (M_r + extra) / (r2 * r);
    *hx = fac * x; *hy = fac * y; *hz = fac * z;
}

__global__ void host_add_kernel(int n_star, const double* __restrict__ pos, double* __restrict__ acc,
        const double* __restrict__ log_r, const double* __restrict__ M_enc, int n_rgrid,
        const int* __restrict__ rank, double m_star_sg)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_star) return;
    double hx, hy, hz;
    host_term(pos, s, log_r, M_enc, n_rgrid, rank, m_star_sg, &hx, &hy, &hz);
    acc[3*s] += hx; acc[3*s+1] += hy; acc[3*s+2] += hz;
}

// kick = 0 is a pure drift and reads no force.
__global__ void kick_drift_kernel(int n_star, double* __restrict__ pos, double* __restrict__ vel,
        const double* __restrict__ acc, const double* __restrict__ log_r,
        const double* __restrict__ M_enc, int n_rgrid, const int* __restrict__ rank,
        double m_star_sg, double kick, double drift)
{
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_star) return;
    double vx = vel[3*s], vy = vel[3*s+1], vz = vel[3*s+2];
    if (kick != 0.0) {
        double hx, hy, hz;
        host_term(pos, s, log_r, M_enc, n_rgrid, rank, m_star_sg, &hx, &hy, &hz);
        vx += kick * (acc[3*s] + hx);
        vy += kick * (acc[3*s+1] + hy);
        vz += kick * (acc[3*s+2] + hz);
        vel[3*s] = vx; vel[3*s+1] = vy; vel[3*s+2] = vz;
    }
    pos[3*s] += drift * vx; pos[3*s+1] += drift * vy; pos[3*s+2] += drift * vz;
}

// track_on_grid's read-out time: shifted by the copy's phase and wrapped into the track's
// own span when a phase is set (t0 = 0, the launch state is the first knot). Past the last
// knot -- reachable only at phase 0 -- the clump is held at its final state.
__device__ __forceinline__ double wrap_time(double t, double phase, double t_last, bool* past)
{
    double te = t + phase;
    if (phase != 0.0 && t_last > 0.0) te = fmod(te, t_last);
    *past = te > t_last;
    return te < 0.0 ? 0.0 : (te > t_last ? t_last : te);
}

// Largest k with t[k] <= te, clipped to the last interval: the interval scipy's PPoly
// evaluates a CubicHermiteSpline on. n >= 2, enforced when the bank is set.
__device__ __forceinline__ int knot_interval(const double* __restrict__ t, int n, double te)
{
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (t[mid] <= te) lo = mid; else hi = mid;
    }
    return lo > n - 2 ? n - 2 : lo;
}

__device__ __forceinline__ void hermite_pos(const double* __restrict__ t, const double* __restrict__ p,
        const double* __restrict__ v, int n, double te, double* x, double* y, double* z)
{
    const int k = knot_interval(t, n, te);
    const double h = t[k+1] - t[k];
    const double s = (te - t[k]) / h;
    const double s2 = s * s, s3 = s2 * s;
    const double h00 = 2.0*s3 - 3.0*s2 + 1.0, h10 = s3 - 2.0*s2 + s;
    const double h01 = -2.0*s3 + 3.0*s2, h11 = s3 - s2;
    const double* p0 = p + 3*k; const double* p1 = p0 + 3;
    const double* v0 = v + 3*k; const double* v1 = v0 + 3;
    *x = h00*p0[0] + h10*h*v0[0] + h01*p1[0] + h11*h*v1[0];
    *y = h00*p0[1] + h10*h*v0[1] + h01*p1[1] + h11*h*v1[1];
    *z = h00*p0[2] + h10*h*v0[2] + h01*p1[2] + h11*h*v1[2];
}

__device__ __forceinline__ double lin_mass(const double* __restrict__ t, const double* __restrict__ m,
        int n, double te)
{
    const int k = knot_interval(t, n, te);
    return m[k] + (m[k+1] - m[k]) / (t[k+1] - t[k]) * (te - t[k]);
}

// One copy's bound mass at time tq, read the way population() reads it.
__device__ __forceinline__ double copy_mass(const Bank& b, int o, double phase, double tq)
{
    const long long a = b.koff[o];
    const int n = (int)(b.koff[o+1] - a);
    if (n == 0) return b.smass[o];
    bool past;
    const double te = wrap_time(tq, phase, b.kt[a + n - 1], &past);
    return past ? b.km[a + n - 1] : lin_mass(b.kt + a, b.km + a, n, te);
}

// The step's clump copies from the bank: position at t_pos on the copy's rotated,
// phase-shifted track; M(<d) row as the time blend between the coarse-grid rows at tA
// and tB (weight wM), each the bound-fraction blend of the launch and settled profiles
// hard-truncated at the copy's mass then; zero once the copy is past t_alive. The one-step
// skew of the CPU driver -- position at the step's start, mass and death at its end -- is
// the caller's, through the times it passes.
__global__ void advance_kernel(Bank b, Copies cp, Clumps c, const double* __restrict__ d_grid,
        int n_d, double t_pos, double tA, double tB, double wM, double t_end)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cp.n) return;
    const int o = cp.orb[i];
    const double ph = cp.phase[i];
    const long long a = b.koff[o];
    const int n = (int)(b.koff[o+1] - a);
    double x, y, z;
    if (n == 0) {
        x = b.spos[3*o]; y = b.spos[3*o+1]; z = b.spos[3*o+2];
    } else {
        bool past;
        const double te = wrap_time(t_pos, ph, b.kt[a + n - 1], &past);
        if (past) { x = b.kp[3*(a+n-1)]; y = b.kp[3*(a+n-1)+1]; z = b.kp[3*(a+n-1)+2]; }
        else hermite_pos(b.kt + a, b.kp + 3*a, b.kv + 3*a, n, te, &x, &y, &z);
    }
    const double* R = cp.rot + 9*i;
    c.x[i] = R[0]*x + R[1]*y + R[2]*z;
    c.y[i] = R[3]*x + R[4]*y + R[5]*z;
    c.z[i] = R[6]*x + R[7]*y + R[8]*z;
    c.orb[i] = i;

    const bool dead = cp.t_alive[i] <= t_end;
    const double mA = copy_mass(b, o, ph, tA), mB = copy_mass(b, o, ph, tB);
    const double m0 = b.m_init[o], mT = b.m_final[o];
    double wA = 0.0, wB = 0.0;
    if (b.hasT[o]) {
        wA = (mA - m0) / (mT - m0); wA = wA < 0.0 ? 0.0 : (wA > 1.0 ? 1.0 : wA);
        wB = (mB - m0) / (mT - m0); wB = wB < 0.0 ? 0.0 : (wB > 1.0 ? 1.0 : wB);
    }
    const double* M0 = b.M0 + (size_t)o * n_d;
    const double* MT = b.MT + (size_t)o * n_d;
    double* row = c.table + (size_t)i * n_d;
    double last = 0.0;
    for (int d = 0; d < n_d; d++) {
        double MA = b.hasT[o] ? (1.0 - wA) * M0[d] + wA * MT[d] : M0[d];
        double MB = b.hasT[o] ? (1.0 - wB) * M0[d] + wB * MT[d] : M0[d];
        MA = MA < mA ? MA : mA;
        MB = MB < mB ? MB : mB;
        last = dead ? 0.0 : (1.0 - wM) * MA + wM * MB;
        row[d] = last;
    }
    // first knot of the flat tail, never below the first knot: inside it the CPU
    // extrapolates the row, which the lookup branch reproduces and the shortcut would not
    int k = n_d;
    while (k > 1 && row[k - 1] == last) k--;
    c.m[i] = last;
    c.rt2[i] = d_grid[k] * d_grid[k];
}

static int alloc_clumps(Clumps& c, int n_cap, int n_orbit_cap, int n_d, int* orb)
{
    CK(cudaMalloc(&c.x, n_cap * sizeof(double)));
    CK(cudaMalloc(&c.y, n_cap * sizeof(double)));
    CK(cudaMalloc(&c.z, n_cap * sizeof(double)));
    CK(cudaMalloc(&c.m, n_cap * sizeof(double)));
    CK(cudaMalloc(&c.rt2, n_cap * sizeof(double)));
    CK(cudaMalloc(&c.table, (size_t)n_orbit_cap * n_d * sizeof(double)));
    c.orb = orb;
    c.n = 0; c.n_d = n_d;
    c.log_d_min = 0; c.inv_dld = 0; c.inv_d_min = 0;
    return 0;
}

static void free_clumps(Clumps& c)
{
    cudaFree(c.x); cudaFree(c.y); cudaFree(c.z); cudaFree(c.m); cudaFree(c.rt2); cudaFree(c.table);
}

static void free_bank(Bank& b)
{
    cudaFree(b.kt); cudaFree(b.kp); cudaFree(b.kv); cudaFree(b.km); cudaFree(b.koff);
    cudaFree(b.M0); cudaFree(b.MT); cudaFree(b.hasT); cudaFree(b.m_init); cudaFree(b.m_final);
    cudaFree(b.spos); cudaFree(b.smass);
    b = Bank();
}

static void free_copies(Copies& c)
{
    cudaFree(c.orb); cudaFree(c.phase); cudaFree(c.rot); cudaFree(c.t_alive);
    c = Copies();
}

// ~8 blocks per SM, never more splits than tiles.
static int pick_split(const TabM* s, int n_clump)
{
    const int gx = (s->n_star + STARS_PER_BLOCK - 1) / STARS_PER_BLOCK;
    const int want = (8 * s->sms + gx - 1) / gx;
    const int most = (n_clump + TILE - 1) / TILE;
    return std::max(1, std::min(want, std::max(most, 1)));
}

// One pair sum into acc: the split partials, then their sum in a fixed order.
static int launch_force(TabM* s, int fp64)
{
    const Clumps& c = s->c;
    const int nb = (s->n_star + 255) / 256;
    if (c.n == 0) {
        CK(cudaMemsetAsync(s->acc, 0, 3 * s->n_star * sizeof(double)));
        return 0;
    }
    const int gx = (s->n_star + STARS_PER_BLOCK - 1) / STARS_PER_BLOCK;
    const int ns = pick_split(s, c.n);
    const int per = (c.n + ns - 1) / ns;
    dim3 grid(gx, ns);
    if (fp64) force_kernel<double><<<grid, NTHREADS>>>(s->n_star, s->pos, s->part, c, per);
    else      force_kernel<float><<<grid, NTHREADS>>>(s->n_star, s->pos, s->part, c, per);
    reduce_kernel<<<nb, 256>>>(s->n_star, ns, s->part, s->acc);
    CK(cudaGetLastError());
    return 0;
}

static void launch_rank(const TabM* s)
{
    const int nb = (s->n_star + 255) / 256;
    rank_kernel<<<nb, 256, (size_t)s->n_star * sizeof(double)>>>(s->n_star, s->pos, s->rank);
}

template <typename U>
static int upload(U** dst, const U* src, size_t n)
{
    CK(cudaMalloc(dst, (n ? n : 1) * sizeof(U)));
    if (n) CK(cudaMemcpy(*dst, src, n * sizeof(U), cudaMemcpyHostToDevice));
    return 0;
}

static int create_impl(TabM* s, const double* log_r, const double* M_enc)
{
    const int n3 = 3 * s->n_star;
    int dev = 0;
    cudaDeviceProp p;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&p, dev));
    s->sms = p.multiProcessorCount;
    s->max_split = pick_split(s, s->n_clump_cap);
    CK(cudaMalloc(&s->pos, n3 * sizeof(double)));
    CK(cudaMalloc(&s->vel, n3 * sizeof(double)));
    CK(cudaMalloc(&s->acc, n3 * sizeof(double)));
    CK(cudaMalloc(&s->part, (size_t)s->max_split * n3 * sizeof(double)));
    CK(cudaMemset(s->acc, 0, n3 * sizeof(double)));
    CK(cudaMalloc(&s->rank, s->n_star * sizeof(int)));
    CK(cudaMalloc(&s->orb, s->n_clump_cap * sizeof(int)));
    CK(cudaMalloc(&s->log_r, s->n_rgrid * sizeof(double)));
    CK(cudaMalloc(&s->M_enc, s->n_rgrid * sizeof(double)));
    CK(cudaMemcpy(s->log_r, log_r, s->n_rgrid * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->M_enc, M_enc, s->n_rgrid * sizeof(double), cudaMemcpyHostToDevice));
    if (alloc_clumps(s->c, s->n_clump_cap, s->n_orbit_cap, s->n_d, s->orb)) return -1;
    s->hd = (double*)malloc((size_t)3 * s->n_clump_cap * sizeof(double));
    if (!s->hd) { snprintf(g_err, sizeof g_err, "host scratch malloc failed"); return -1; }
    CK(cudaEventCreate(&s->e0));
    CK(cudaEventCreate(&s->e1));
    return 0;
}

static void set_dgrid(TabM* s, double log_d_min, double log_d_max)
{
    s->c.log_d_min = log_d_min;
    s->c.inv_dld = (s->n_d - 1) / (log_d_max - log_d_min);
    s->c.inv_d_min = exp(-log_d_min);
}

extern "C" {

const char* tabm_last_error() { return g_err; }

int tabm_device_info(char* name, int len, int* cc_major, int* cc_minor, int* sms, size_t* mem)
{
    int dev = 0;
    cudaDeviceProp p;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&p, dev));
    snprintf(name, len, "%s", p.name);
    *cc_major = p.major; *cc_minor = p.minor;
    *sms = p.multiProcessorCount; *mem = p.totalGlobalMem;
    return 0;
}

void tabm_destroy(TabM* s)
{
    if (!s) return;
    cudaFree(s->pos); cudaFree(s->vel); cudaFree(s->acc); cudaFree(s->part);
    cudaFree(s->rank); cudaFree(s->orb); cudaFree(s->log_r); cudaFree(s->M_enc);
    cudaFree(s->d_grid);
    free_clumps(s->c);
    free_bank(s->bank); free_copies(s->cop);
    free(s->hd);
    if (s->e0) cudaEventDestroy(s->e0);
    if (s->e1) cudaEventDestroy(s->e1);
    delete s;
}

TabM* tabm_create(int n_star, int n_clump_cap, int n_orbit_cap, int n_d,
                  const double* log_r, const double* M_enc, int n_rgrid)
{
    TabM* s = new TabM();
    s->n_star = n_star;
    s->n_clump_cap = std::max(n_clump_cap, 1);
    s->n_orbit_cap = std::max(n_orbit_cap, 1);
    s->n_d = n_d;
    s->n_rgrid = n_rgrid;
    if (create_impl(s, log_r, M_enc)) { tabm_destroy(s); return nullptr; }
    return s;
}

int tabm_set_stars(TabM* s, const double* pos, const double* vel)
{
    const size_t nb = 3 * s->n_star * sizeof(double);
    CK(cudaMemcpy(s->pos, pos, nb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->vel, vel, nb, cudaMemcpyHostToDevice));
    CK(cudaMemset(s->acc, 0, nb));
    return 0;
}

int tabm_get_stars(TabM* s, double* pos, double* vel)
{
    const size_t nb = 3 * s->n_star * sizeof(double);
    CK(cudaMemcpy(pos, s->pos, nb, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(vel, s->vel, nb, cudaMemcpyDeviceToHost));
    return 0;
}

// xyz is (n, 3) row-major; m_out the flat-tail mass; rt2 the squared radius past which
// the row is flat; orb the table row each clump reads.
int tabm_set_clumps(TabM* s, int n, const double* xyz, const double* m_out,
                    const double* rt2, const int* orb)
{
    if (n > s->n_clump_cap) {
        snprintf(g_err, sizeof g_err, "%d clumps exceed the capacity of %d", n, s->n_clump_cap);
        return -1;
    }
    s->c.n = n;
    if (n == 0) return 0;
    for (int j = 0; j < n; j++) {
        s->hd[j] = xyz[3*j]; s->hd[n + j] = xyz[3*j + 1]; s->hd[2*n + j] = xyz[3*j + 2];
    }
    CK(cudaMemcpy(s->c.x, s->hd, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->c.y, s->hd + n, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->c.z, s->hd + 2*n, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->c.m, m_out, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->c.rt2, rt2, n * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s->orb, orb, n * sizeof(int), cudaMemcpyHostToDevice));
    return 0;
}

// table is (n_orbit, n_d) row-major M(<d) on the uniform grid log_d_min .. log_d_max.
int tabm_set_table(TabM* s, int n_orbit, int n_d, const double* table,
                   double log_d_min, double log_d_max)
{
    if (n_d != s->n_d || n_orbit > s->n_orbit_cap) {
        snprintf(g_err, sizeof g_err, "table (%d, %d) against capacity (%d, %d)",
                 n_orbit, n_d, s->n_orbit_cap, s->n_d);
        return -1;
    }
    CK(cudaMemcpy(s->c.table, table, (size_t)n_orbit * n_d * sizeof(double),
                  cudaMemcpyHostToDevice));
    set_dgrid(s, log_d_min, log_d_max);
    return 0;
}

// The orbits copies are read from. Knots are concatenated over orbits with koff giving
// each orbit's range; an orbit with no knots is held at spos with smass, one with a
// single knot is refused, as a spline through it would be. log_d and d_grid are the
// n_d-point separation grid the M0 / MT rows sit on, d_grid exponentiated by the caller
// so rt2 here and clump_rows on the host square the same numbers. Built aside and
// swapped in on success, so a failed upload leaves the previous bank in place.
int tabm_set_bank(TabM* s, int n_orbit, long long n_knot, const double* kt, const double* kp,
                  const double* kv, const double* km, const long long* koff,
                  const double* M0, const double* MT, const int* hasT,
                  const double* m_init, const double* m_final,
                  const double* spos, const double* smass, const double* log_d,
                  const double* dg)
{
    for (int o = 0; o < n_orbit; o++)
        if (koff[o+1] - koff[o] == 1) {
            snprintf(g_err, sizeof g_err, "orbit %d has a single knot", o);
            return -1;
        }
    Bank b = Bank();
    b.n_orbit = n_orbit;
    if (upload(&b.kt, kt, n_knot) || upload(&b.kp, kp, 3 * n_knot) || upload(&b.kv, kv, 3 * n_knot)
        || upload(&b.km, km, n_knot) || upload(&b.koff, koff, n_orbit + 1)
        || upload(&b.M0, M0, (size_t)n_orbit * s->n_d) || upload(&b.MT, MT, (size_t)n_orbit * s->n_d)
        || upload(&b.hasT, hasT, n_orbit) || upload(&b.m_init, m_init, n_orbit)
        || upload(&b.m_final, m_final, n_orbit) || upload(&b.spos, spos, 3 * n_orbit)
        || upload(&b.smass, smass, n_orbit)) {
        free_bank(b);
        return -1;
    }
    double* d_grid = nullptr;
    if (upload(&d_grid, dg, s->n_d)) { free_bank(b); return -1; }
    free_bank(s->bank);
    cudaFree(s->d_grid);
    s->bank = b;
    s->d_grid = d_grid;
    set_dgrid(s, log_d[0], log_d[s->n_d - 1]);
    return 0;
}

int tabm_set_copies(TabM* s, int n, const int* orb, const double* phase, const double* rot,
                    const double* t_alive)
{
    if (n > s->n_clump_cap || n > s->n_orbit_cap) {
        snprintf(g_err, sizeof g_err, "%d copies against capacity %d clumps / %d rows",
                 n, s->n_clump_cap, s->n_orbit_cap);
        return -1;
    }
    for (int i = 0; i < n; i++)
        if (orb[i] < 0 || orb[i] >= s->bank.n_orbit) {
            snprintf(g_err, sizeof g_err, "copy %d reads orbit %d of a bank of %d",
                     i, orb[i], s->bank.n_orbit);
            return -1;
        }
    Copies c = Copies();
    c.n = n;
    if (upload(&c.orb, orb, n) || upload(&c.phase, phase, n) || upload(&c.rot, rot, 9 * (size_t)n)
        || upload(&c.t_alive, t_alive, n)) {
        free_copies(c);
        return -1;
    }
    free_copies(s->cop);
    s->cop = c;
    s->c.n = n;
    return 0;
}

int tabm_advance(TabM* s, double t_pos, double tA, double tB, double wM, double t_end)
{
    if (s->cop.n == 0) return 0;
    const int nb = (s->cop.n + 255) / 256;
    advance_kernel<<<nb, 256>>>(s->bank, s->cop, s->c, s->d_grid, s->n_d,
                                t_pos, tA, tB, wM, t_end);
    CK(cudaGetLastError());
    return 0;
}

// The current clumps as the force sees them: xyz, m_out and rt2 per clump, and the
// first n_rows table rows. Synchronises.
int tabm_get_clumps(TabM* s, double* xyz, double* m_out, double* rt2, double* rows, int n_rows)
{
    const int n = s->c.n;
    if (n_rows > s->n_orbit_cap) {
        snprintf(g_err, sizeof g_err, "%d rows asked of a table holding %d", n_rows, s->n_orbit_cap);
        return -1;
    }
    CK(cudaDeviceSynchronize());
    if (n) {
        CK(cudaMemcpy(s->hd, s->c.x, n * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s->hd + n, s->c.y, n * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s->hd + 2*n, s->c.z, n * sizeof(double), cudaMemcpyDeviceToHost));
        for (int j = 0; j < n; j++) {
            xyz[3*j] = s->hd[j]; xyz[3*j+1] = s->hd[n + j]; xyz[3*j+2] = s->hd[2*n + j];
        }
        CK(cudaMemcpy(m_out, s->c.m, n * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(rt2, s->c.rt2, n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    if (n_rows)
        CK(cudaMemcpy(rows, s->c.table, (size_t)n_rows * s->n_d * sizeof(double),
                      cudaMemcpyDeviceToHost));
    return 0;
}

// n_sub Yoshida-4 sub-steps of sub_dt with the clumps held fixed, the same composition
// _yoshida4_step_tabM_nb runs: drift c0, then three (force, kick d_i, drift c_{i+1}).
int tabm_step(TabM* s, double sub_dt, int n_sub, double m_star_sg, int fp64)
{
    const double c3 = pow(2.0, 1.0 / 3.0);
    const double W1 = 1.0 / (2.0 - c3), W0 = -c3 / (2.0 - c3);
    const double YC[4] = {W1 / 2.0, (W0 + W1) / 2.0, (W0 + W1) / 2.0, W1 / 2.0};
    const double YD[3] = {W1, W0, W1};
    const int* rank = (m_star_sg > 0.0) ? s->rank : nullptr;
    const int nb = (s->n_star + 255) / 256;
    for (int it = 0; it < n_sub; it++) {
        kick_drift_kernel<<<nb, 256>>>(s->n_star, s->pos, s->vel, s->acc, s->log_r, s->M_enc,
                                       s->n_rgrid, rank, m_star_sg, 0.0, YC[0] * sub_dt);
        for (int i = 0; i < 3; i++) {
            if (rank) launch_rank(s);
            if (launch_force(s, fp64)) return -1;
            kick_drift_kernel<<<nb, 256>>>(s->n_star, s->pos, s->vel, s->acc, s->log_r, s->M_enc,
                                           s->n_rgrid, rank, m_star_sg, YD[i] * sub_dt,
                                           YC[i + 1] * sub_dt);
        }
    }
    CK(cudaGetLastError());
    return 0;
}

// One force evaluation at the current positions, host term included, into out (n_star, 3).
int tabm_accel(TabM* s, double* out, double m_star_sg, int fp64)
{
    const size_t nb3 = 3 * s->n_star * sizeof(double);
    const int nb = (s->n_star + 255) / 256;
    const int* rank = (m_star_sg > 0.0) ? s->rank : nullptr;
    if (rank) launch_rank(s);
    if (launch_force(s, fp64)) return -1;
    host_add_kernel<<<nb, 256>>>(s->n_star, s->pos, s->acc, s->log_r, s->M_enc, s->n_rgrid,
                                 rank, m_star_sg);
    CK(cudaGetLastError());
    CK(cudaMemcpy(out, s->acc, nb3, cudaMemcpyDeviceToHost));
    return 0;
}

// Kernel time of n_repeat pair sums at the current positions, in ms per evaluation.
int tabm_time_force(TabM* s, int n_repeat, int fp64, float* ms)
{
    CK(cudaEventRecord(s->e0));
    for (int r = 0; r < n_repeat; r++)
        if (launch_force(s, fp64)) return -1;
    CK(cudaEventRecord(s->e1));
    CK(cudaEventSynchronize(s->e1));
    CK(cudaEventElapsedTime(ms, s->e0, s->e1));
    *ms /= n_repeat;
    return 0;
}

int tabm_sync(TabM* s)
{
    (void)s;
    CK(cudaDeviceSynchronize());
    return 0;
}

}  // extern "C"
