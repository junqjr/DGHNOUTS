/**
 * \file    bench_choices.cpp
 * \brief   Measures each optimisation of jacobi_optimized.cpp against the
 *          alternatives that were considered: the numbers behind the
 *          \ref optimizations "Optimisations" page.
 *
 * \details Not a test, and not part of the library: the alternatives below
 *          are small, self-contained copies of the code that was tried and
 *          rejected, written for the measurement only. Each table line
 *          answers one question ("is the normalised recurrence faster?",
 *          "which block size?", ...). Build and run it on the machine of
 *          interest (on a cluster: on a compute node):
 *
 *              cmake --preset gcc -DDGPOLY_NATIVE=ON
 *              cmake --build --preset gcc --target bench_choices
 *              ./build/gcc/bench_choices          # optional argument: degree n (default 16)
 *
 *          Build it once with and once without `-DDGPOLY_NATIVE=ON` to see
 *          the effect of SIMD width and fused multiply-add.
 *
 *          Units: nanoseconds per point and per degree (one step of the
 *          recurrence) for the array routines; nanoseconds or microseconds per
 *          call otherwise. Each time is the best of 7 repetitions; on a shared
 *          machine, differences below about 20 % are noise.
 */
#include "jacobi.hpp"
#include "jacobi_impl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using dg::internal::p1;
using dg::internal::rec_coef;

volatile double sink = 0.0;   ///< keeps the compiler from removing the timed work

/**
 * \brief  Best time of one call of `f`, in seconds.
 * \param[in] f  the work to time
 * \return    minimum over 7 measurements of the mean time per call, each
 *            measurement repeating `f` for at least 0.05 s
 */
template <class F>
double best_time(F&& f)
{
    using clock = std::chrono::steady_clock;
    int reps = 1;
    for (;;) {
        const auto t0 = clock::now();
        for (int r = 0; r < reps; ++r) f();
        if (std::chrono::duration<double>(clock::now() - t0).count() >= 0.05) break;
        reps *= 2;
    }
    double best = 1e300;
    for (int s = 0; s < 7; ++s) {
        const auto t0 = clock::now();
        for (int r = 0; r < reps; ++r) f();
        best = std::min(best, std::chrono::duration<double>(clock::now() - t0).count() / reps);
    }
    return best;
}

/**
 * \brief  Multiply-add, fused when the hardware has FMA: same rule as `mad()` in jacobi_optimized.cpp.
 * \param[in] x  first factor
 * \param[in] y  second factor
 * \param[in] z  addend
 * \return    \f$xy+z\f$
 */
inline double mad(double x, double y, double z)
{
#ifdef FP_FAST_FMA
    return std::fma(x, y, z);
#else
    return x * y + z;
#endif
}

/// Normalised coefficients \f$A_k, B_k, C_k\f$ of (A.1.3').
struct ABC {
    double A;   ///< \f$A_k = a_k^2/a_k^1\f$
    double B;   ///< \f$B_k = a_k^3/a_k^1\f$
    double C;   ///< \f$C_k = a_k^4/a_k^1\f$
};

/**
 * \brief  Table of \f$A_k, B_k, C_k\f$, \f$k = 1,\dots,n\f$.
 * \param[in] n  highest index
 * \param[in] a  \f$\alpha\f$
 * \param[in] b  \f$\beta\f$
 * \return    the table, indexed by \f$k\f$
 */
std::vector<ABC> table(int n, double a, double b)
{
    std::vector<ABC> t(static_cast<std::size_t>(n) + 1);
    for (int k = 1; k <= n; ++k) {
        const auto c = rec_coef(k, a, b);
        t[k] = { c.a2 / c.a1, c.a3 / c.a1, c.a4 / c.a1 };
    }
    return t;
}

// ---------------------------------------------------------------------------
// Alternatives for the array routine jacobi(): loop order and blocking
// ---------------------------------------------------------------------------

/**
 * \brief  Degree outer, points inner over the whole arrays, book's form (A.1.3) with the division.
 * \param[in]  n   degree, \f$n \ge 2\f$
 * \param[in]  a   \f$\alpha\f$
 * \param[in]  b   \f$\beta\f$
 * \param[in]  x   points \f$\xi_i\f$, size `np`
 * \param[in]  np  number of points
 * \param[out] p   values \f$P_n(\xi_i)\f$, size `np`
 * \param      pm  work array for \f$P_{k-1}\f$, size `np`
 */
void degree_outer_book(int n, double a, double b, const double* x, std::size_t np, double* p,
                       std::vector<double>& pm)
{
    for (std::size_t i = 0; i < np; ++i) { pm[i] = 1.0; p[i] = p1(a, b, x[i]); }
    for (int k = 1; k < n; ++k) {
        const auto c = rec_coef(k, a, b);
        for (std::size_t i = 0; i < np; ++i) {
            const double q = ((c.a2 + c.a3 * x[i]) * p[i] - c.a4 * pm[i]) / c.a1;
            pm[i] = p[i]; p[i] = q;
        }
    }
}

/**
 * \brief  Degree outer, points inner over the whole arrays, normalised form (A.1.3').
 * \param[in]  n   degree, \f$n \ge 2\f$
 * \param[in]  a   \f$\alpha\f$
 * \param[in]  b   \f$\beta\f$
 * \param[in]  x   points \f$\xi_i\f$, size `np`
 * \param[in]  np  number of points
 * \param[out] p   values \f$P_n(\xi_i)\f$, size `np`
 * \param      pm  work array for \f$P_{k-1}\f$, size `np`
 * \param[in]  t   coefficient table
 */
void degree_outer_norm(int n, double a, double b, const double* x, std::size_t np, double* p,
                       std::vector<double>& pm, const ABC* t)
{
    for (std::size_t i = 0; i < np; ++i) { pm[i] = 1.0; p[i] = p1(a, b, x[i]); }
    for (int k = 1; k < n; ++k)
        for (std::size_t i = 0; i < np; ++i) {
            const double q = mad(mad(t[k].B, x[i], t[k].A), p[i], -(t[k].C * pm[i]));
            pm[i] = p[i]; p[i] = q;
        }
}

/**
 * \brief  L1-cache blocking: degree outer on chunks of `L` points kept in small arrays.
 * \tparam     L   points per block
 * \param[in]  n   degree, \f$n \ge 2\f$
 * \param[in]  a   \f$\alpha\f$
 * \param[in]  b   \f$\beta\f$
 * \param[in]  x   points \f$\xi_i\f$, size `np`
 * \param[in]  np  number of points
 * \param[out] p   values \f$P_n(\xi_i)\f$, size `np`
 * \param[in]  t   coefficient table
 */
template <std::size_t L>
void cache_blocks(int n, double a, double b, const double* x, std::size_t np, double* p, const ABC* t)
{
    double pm[L], pk[L];
    for (std::size_t i0 = 0; i0 < np; i0 += L) {
        const std::size_t m = std::min(L, np - i0);
        for (std::size_t i = 0; i < m; ++i) { pm[i] = 1.0; pk[i] = p1(a, b, x[i0 + i]); }
        for (int k = 1; k < n; ++k)
            for (std::size_t i = 0; i < m; ++i) {
                const double q = mad(mad(t[k].B, x[i0 + i], t[k].A), pk[i], -(t[k].C * pm[i]));
                pm[i] = pk[i]; pk[i] = q;
            }
        std::copy(pk, pk + m, p + i0);
    }
}

/**
 * \brief  Register blocking with `L` points per block (full blocks only; np is a multiple of 64 here).
 * \tparam     L   points per block
 * \param[in]  n   degree, \f$n \ge 2\f$
 * \param[in]  a   \f$\alpha\f$
 * \param[in]  b   \f$\beta\f$
 * \param[in]  x   points \f$\xi_i\f$, size `np`
 * \param[in]  np  number of points
 * \param[out] p   values \f$P_n(\xi_i)\f$, size `np`
 * \param[in]  t   coefficient table
 */
template <int L>
void register_blocks(int n, double a, double b, const double* x, std::size_t np, double* p, const ABC* t)
{
    for (std::size_t i0 = 0; i0 + L <= np; i0 += L) {
        double xx[L], pm[L], pk[L];
        for (int i = 0; i < L; ++i) { xx[i] = x[i0 + i]; pm[i] = 1.0; pk[i] = p1(a, b, xx[i]); }
        for (int k = 1; k < n; ++k) {
            const double A = t[k].A, B = t[k].B, C = t[k].C;
            for (int i = 0; i < L; ++i) {
                const double q = mad(mad(B, xx[i], A), pk[i], -(C * pm[i]));
                pm[i] = pk[i]; pk[i] = q;
            }
        }
        for (int i = 0; i < L; ++i) p[i0 + i] = pk[i];
    }
}

// ---------------------------------------------------------------------------
// Alternatives for jacobi_and_deriv() and jacobi_all()
// ---------------------------------------------------------------------------

/**
 * \brief  Values and derivatives, degree outer over the whole arrays (four arrays in memory).
 * \param[in]  n   degree, \f$n \ge 2\f$
 * \param[in]  a   \f$\alpha\f$
 * \param[in]  b   \f$\beta\f$
 * \param[in]  x   points \f$\xi_i\f$, size `np`
 * \param[in]  np  number of points
 * \param[out] p   values, size `np`
 * \param[out] dp  derivatives, size `np`
 * \param      w   work array, size `2*np`
 * \param[in]  t   coefficient table
 */
void vd_degree_outer(int n, double a, double b, const double* x, std::size_t np,
                     double* p, double* dp, std::vector<double>& w, const ABC* t)
{
    double* pm = w.data();
    double* dm = w.data() + np;
    const double d1 = 0.5 * (a + b + 2.0);
    for (std::size_t i = 0; i < np; ++i) { pm[i] = 1.0; dm[i] = 0.0; p[i] = p1(a, b, x[i]); dp[i] = d1; }
    for (int k = 1; k < n; ++k)
        for (std::size_t i = 0; i < np; ++i) {
            const double axi = mad(t[k].B, x[i], t[k].A);
            const double q   = mad(axi, p[i], -(t[k].C * pm[i]));
            const double dq  = mad(axi, dp[i], mad(t[k].B, p[i], -(t[k].C * dm[i])));
            pm[i] = p[i]; p[i] = q; dm[i] = dp[i]; dp[i] = dq;
        }
}

/**
 * \brief  Values and derivatives, L1-cache blocks of 256 points (four small arrays).
 * \param[in]  n   degree, \f$n \ge 2\f$
 * \param[in]  a   \f$\alpha\f$
 * \param[in]  b   \f$\beta\f$
 * \param[in]  x   points \f$\xi_i\f$, size `np`
 * \param[in]  np  number of points
 * \param[out] p   values, size `np`
 * \param[out] dp  derivatives, size `np`
 * \param[in]  t   coefficient table
 */
void vd_cache_blocks(int n, double a, double b, const double* x, std::size_t np,
                     double* p, double* dp, const ABC* t)
{
    constexpr std::size_t L = 256;
    double pm[L], pk[L], dm[L], dk[L];
    const double d1 = 0.5 * (a + b + 2.0);
    for (std::size_t i0 = 0; i0 < np; i0 += L) {
        const std::size_t m = std::min(L, np - i0);
        for (std::size_t i = 0; i < m; ++i) { pm[i] = 1.0; dm[i] = 0.0; pk[i] = p1(a, b, x[i0 + i]); dk[i] = d1; }
        for (int k = 1; k < n; ++k)
            for (std::size_t i = 0; i < m; ++i) {
                const double axi = mad(t[k].B, x[i0 + i], t[k].A);
                const double q   = mad(axi, pk[i], -(t[k].C * pm[i]));
                const double dq  = mad(axi, dk[i], mad(t[k].B, pk[i], -(t[k].C * dm[i])));
                pm[i] = pk[i]; pk[i] = q; dm[i] = dk[i]; dk[i] = dq;
            }
        std::copy(pk, pk + m, p + i0);
        std::copy(dk, dk + m, dp + i0);
    }
}

/**
 * \brief  Vandermonde matrix column by column: degree outer, every row touched at every degree.
 * \param[in]  N   highest degree, \f$N \ge 2\f$
 * \param[in]  a   \f$\alpha\f$
 * \param[in]  b   \f$\beta\f$
 * \param[in]  x   points \f$\xi_i\f$, size `np`
 * \param[in]  np  number of points
 * \param[out] V   matrix, row-major, size `np*(N+1)`
 * \param[in]  t   coefficient table
 */
void vandermonde_columns(int N, double a, double b, const double* x, std::size_t np, double* V, const ABC* t)
{
    const std::size_t ld = static_cast<std::size_t>(N) + 1;
    for (std::size_t i = 0; i < np; ++i) { V[i * ld] = 1.0; V[i * ld + 1] = p1(a, b, x[i]); }
    for (int k = 1; k < N; ++k)
        for (std::size_t i = 0; i < np; ++i)
            V[i * ld + k + 1] = mad(mad(t[k].B, x[i], t[k].A), V[i * ld + k], -(t[k].C * V[i * ld + k - 1]));
}

// ---------------------------------------------------------------------------
// Alternatives for the single-point evaluation and for jacobi_zeros()
// ---------------------------------------------------------------------------

/**
 * \brief  One point, normalised coefficients read from a table (what jacobi_zeros() does).
 * \param[in] n  degree, \f$n \ge 1\f$
 * \param[in] a  \f$\alpha\f$
 * \param[in] b  \f$\beta\f$
 * \param[in] x  point \f$\xi\f$
 * \param[in] t  coefficient table
 * \return    \f$P_n(\xi)\f$
 */
double point_table(int n, double a, double b, double x, const ABC* t)
{
    double pm = 1.0, pk = p1(a, b, x);
    for (int k = 1; k < n; ++k) {
        const double q = mad(mad(t[k].B, x, t[k].A), pk, -(t[k].C * pm));
        pm = pk; pk = q;
    }
    return pk;
}

/**
 * \brief  Deflation sum \f$\sum_{j<k}1/(r-\xi_j)\f$ as one chain of additions.
 * \param[in] r  Newton iterate
 * \param[in] z  zeros already found
 * \param[in] k  their number
 * \return    the sum
 */
double deflation_chain(double r, const double* z, int k)
{
    double s = 0.0;
    for (int j = 0; j < k; ++j) s += 1.0 / (r - z[j]);
    return s;
}

/**
 * \brief  Deflation sum with eight partial sums (what jacobi_optimized.cpp does).
 * \param[in] r  Newton iterate
 * \param[in] z  zeros already found
 * \param[in] k  their number
 * \return    the sum
 */
double deflation_partial(double r, const double* z, int k)
{
    double acc[8] = {};
    int j = 0;
    for (; j + 8 <= k; j += 8)
        for (int l = 0; l < 8; ++l) acc[l] += 1.0 / (r - z[j + l]);
    double s = 0.0;
    for (; j < k; ++j) s += 1.0 / (r - z[j]);
    for (int l = 0; l < 8; ++l) s += acc[l];
    return s;
}

} // namespace

/**
 * \brief  Run every comparison and print the tables.
 * \param[in] argc  argument count
 * \param[in] argv  optional degree \f$n\f$ (default 16)
 * \return    0
 */
int main(int argc, char** argv)
{
    const int    n = argc > 1 ? std::atoi(argv[1]) : 16;
    const double a = 0.0, b = 0.0;
    const std::vector<ABC> t = table(n, a, b);

#ifdef FP_FAST_FMA
    const char* fma = "yes";
#else
    const char* fma = "no";
#endif
    std::printf("bench_choices: double, n = %d, alpha = beta = 0, fused multiply-add: %s\n", n, fma);

    for (std::size_t np : {std::size_t(4096), std::size_t(1) << 20}) {
        std::vector<double> x(np), p(np), dp(np), w(2 * np), V(np * (n + 1));
        for (std::size_t i = 0; i < np; ++i) x[i] = -1.0 + 2.0 * double(i) / double(np - 1);
        const double per = 1e9 / (double(np) * double(n - 1));   // seconds -> ns per point per degree
        auto line = [&](const char* what, double s) { std::printf("  %-52s %7.3f\n", what, s * per); };

        std::printf("\nnp = %zu points, ns per point per degree\n", np);
        std::printf(" jacobi (values)\n");
        line("reference: one point at a time, (A.1.3) with division",
             best_time([&] { dg::reference::jacobi(n, a, b, x.data(), np, p.data()); sink = p[1]; }));
        line("degree outer, whole arrays, (A.1.3) with division",
             best_time([&] { degree_outer_book(n, a, b, x.data(), np, p.data(), w); sink = p[1]; }));
        line("degree outer, whole arrays, normalised (A.1.3')",
             best_time([&] { degree_outer_norm(n, a, b, x.data(), np, p.data(), w, t.data()); sink = p[1]; }));
        line("L1-cache blocks of 256 points",
             best_time([&] { cache_blocks<256>(n, a, b, x.data(), np, p.data(), t.data()); sink = p[1]; }));
        line("register blocks of 8 points",
             best_time([&] { register_blocks<8>(n, a, b, x.data(), np, p.data(), t.data()); sink = p[1]; }));
        line("register blocks of 16 points",
             best_time([&] { register_blocks<16>(n, a, b, x.data(), np, p.data(), t.data()); sink = p[1]; }));
        line("register blocks of 32 points",
             best_time([&] { register_blocks<32>(n, a, b, x.data(), np, p.data(), t.data()); sink = p[1]; }));
        line("register blocks of 64 points",
             best_time([&] { register_blocks<64>(n, a, b, x.data(), np, p.data(), t.data()); sink = p[1]; }));
        line("optimized (register blocks of 32, chosen)",
             best_time([&] { dg::optimized::jacobi(n, a, b, x.data(), np, p.data()); sink = p[1]; }));

        std::printf(" jacobi_and_deriv (values and derivatives)\n");
        line("reference: one point at a time",
             best_time([&] { dg::reference::jacobi_and_deriv<double>(n, a, b, x.data(), np, p.data(), dp.data()); sink = p[1]; }));
        line("degree outer, whole arrays",
             best_time([&] { vd_degree_outer(n, a, b, x.data(), np, p.data(), dp.data(), w, t.data()); sink = p[1]; }));
        line("L1-cache blocks of 256 points",
             best_time([&] { vd_cache_blocks(n, a, b, x.data(), np, p.data(), dp.data(), t.data()); sink = p[1]; }));
        line("optimized (register blocks of 32, chosen)",
             best_time([&] { dg::optimized::jacobi_and_deriv<double>(n, a, b, x.data(), np, p.data(), dp.data()); sink = p[1]; }));

        std::printf(" jacobi_all (Vandermonde matrix, row-major)\n");
        line("reference: one row at a time",
             best_time([&] { dg::reference::jacobi_all(n, a, b, x.data(), np, V.data()); sink = V[1]; }));
        line("column by column (degree outer)",
             best_time([&] { vandermonde_columns(n, a, b, x.data(), np, V.data(), t.data()); sink = V[1]; }));
        line("optimized (rows of a block of 32 points, chosen)",
             best_time([&] { dg::optimized::jacobi_all(n, a, b, x.data(), np, V.data()); sink = V[1]; }));
    }

    {
        const int N = 1000;
        auto per_call = [&](auto f) {
            return best_time([&] { double s = 0.0; for (int i = 0; i < N; ++i) s += f(0.3 + 1e-6 * i); sink = s; }) / N * 1e9;
        };
        std::printf("\nsingle point, ns per call\n");
        std::printf("  %-52s %7.1f\n", "reference: (A.1.3) with division, coefficients on the fly",
                    per_call([&](double xi) { return dg::reference::jacobi(n, a, b, xi); }));
        std::printf("  %-52s %7.1f\n", "optimized: (A.1.3'), coefficients on the fly",
                    per_call([&](double xi) { return dg::optimized::jacobi(n, a, b, xi); }));
        std::printf("  %-52s %7.1f\n", "(A.1.3'), coefficients from a table",
                    per_call([&](double xi) { return point_table(n, a, b, xi, t.data()); }));
        std::printf("  %-52s %7.1f\n", "for comparison: heap allocation of a 3n table",
                    per_call([&](double xi) { std::vector<double> v(3 * std::size_t(n)); v[0] = xi; return v[0]; }));
    }

    std::printf("\njacobi_zeros\n");
    for (int k : {16, 64}) {
        std::vector<double> z(k);
        for (int j = 0; j < k; ++j) z[j] = -1.0 + 2.0 * (j + 0.5) / k;
        const int N = 100;
        std::printf("  deflation sum, k = %2d: one chain %6.1f ns, eight partial sums %6.1f ns\n", k,
                    best_time([&] { double s = 0.0; for (int i = 0; i < N; ++i) s += deflation_chain(0.123 + 1e-7 * i, z.data(), k); sink = s; }) / N * 1e9,
                    best_time([&] { double s = 0.0; for (int i = 0; i < N; ++i) s += deflation_partial(0.123 + 1e-7 * i, z.data(), k); sink = s; }) / N * 1e9);
    }
    for (int m : {32, 64}) {
        std::vector<double> c(m + 1), pc(m + 1);
        for (int i = 0; i <= m; ++i) c[i] = -1.0 + 2.0 * i / m;
        std::printf("  certification, m = %d: %d single-point calls %6.2f us, one array call %6.2f us\n", m, m + 1,
                    best_time([&] { for (int i = 0; i <= m; ++i) pc[i] = dg::optimized::jacobi(m, 1.0, 1.0, c[i]); sink = pc[1]; }) * 1e6,
                    best_time([&] { dg::optimized::jacobi(m, 1.0, 1.0, c.data(), c.size(), pc.data()); sink = pc[1]; }) * 1e6);
    }
    for (int q : {16, 32, 64}) {
        std::vector<double> z(q - 2);
        std::printf("  Gauss-Lobatto nodes, Q = %2d: reference %6.1f us, optimized %6.1f us\n", q,
                    best_time([&] { dg::reference::jacobi_zeros(q - 2, 1.0, 1.0, z.data()); sink = z[0]; }) * 1e6,
                    best_time([&] { dg::optimized::jacobi_zeros(q - 2, 1.0, 1.0, z.data()); sink = z[0]; }) * 1e6);
    }
    return 0;
}
