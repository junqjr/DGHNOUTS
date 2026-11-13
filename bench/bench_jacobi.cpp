/**
 * \file    bench_jacobi.cpp
 * \brief   Timing of every routine of the Jacobi module, reference and
 *          optimised implementations side by side.
 *
 * \details Not a test: it prints times, it does not check results (test_impl
 *          checks that the two implementations agree). Build it with the
 *          optimisation flags of the target machine, and run it on that
 *          machine (on a cluster, on a compute node, not on the login node):
 *
 *              cmake --preset gcc -DDGPOLY_NATIVE=ON
 *              cmake --build --preset gcc --target bench_jacobi
 *              ./build/gcc/bench_jacobi
 *
 *          Every time is the best of several repetitions, each lasting at
 *          least 0.1 s, to filter out noise from other processes.
 *
 *          Units:
 *          - recurrence routines: nanoseconds per point and per degree,
 *            i.e. per step of (A.1.3), so that different \f$n\f$ and
 *            \f$n_p\f$ can be compared;
 *          - jacobi_norm2(): nanoseconds per call (one implementation);
 *          - jacobi_zeros(): microseconds per call.
 */
#include "jacobi.hpp"
#include "jacobi_impl.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace {

volatile double sink = 0.0;   ///< keeps the compiler from removing the timed work

/**
 * \brief  Best time of one call of `f`, in seconds.
 * \param[in] f  the work to time
 * \return    minimum over 5 measurements of the mean time per call, each
 *            measurement repeating `f` for at least 0.1 s
 */
template <class F>
double best_time(F&& f)
{
    using clock = std::chrono::steady_clock;
    int reps = 1;
    for (;;) {                                    // calibrate: >= 0.1 s per measurement
        const auto t0 = clock::now();
        for (int r = 0; r < reps; ++r) f();
        if (std::chrono::duration<double>(clock::now() - t0).count() >= 0.1) break;
        reps *= 2;
    }
    double best = 1e300;
    for (int s = 0; s < 5; ++s) {
        const auto t0 = clock::now();
        for (int r = 0; r < reps; ++r) f();
        best = std::min(best, std::chrono::duration<double>(clock::now() - t0).count() / reps);
    }
    return best;
}

/**
 * \brief  Equispaced points in \f$[-1,1]\f$.
 * \param[in] np  number of points, \f$n_p \ge 2\f$
 * \return    the points
 */
template <typename T>
std::vector<T> points(std::size_t np)
{
    std::vector<T> xi(np);
    for (std::size_t i = 0; i < np; ++i) xi[i] = T(-1) + T(2) * T(i) / T(np - 1);
    return xi;
}

/**
 * \brief  Time all routines in precision `T`, both implementations, and print
 *         one table.
 * \param[in] tname  name of the type, for the header
 */
template <typename T>
void run(const char* tname)
{
    namespace R = dg::reference;
    namespace O = dg::optimized;
    const int    n = 16;
    const double a = 0.0, b = 0.0;
    std::printf("\n=== %s, n = %d: ns per point and per degree, reference -> optimized (speed-up) ===\n",
                tname, n);
    std::printf("%-26s %26s %26s %26s\n", "routine", "np = 32", "np = 4096", "np = 1M");

    const std::size_t nps[3] = {32, 4096, std::size_t(1) << 20};
    std::vector<T> xi[3], p[3], dp[3], V[3];
    for (int j = 0; j < 3; ++j) {
        xi[j] = points<T>(nps[j]);
        p[j].resize(nps[j]);
        dp[j].resize(nps[j]);
        V[j].resize(nps[j] * (n + 1));
    }
    auto row = [&](const char* name, auto&& ref, auto&& opt) {
        std::printf("%-26s", name);
        for (int j = 0; j < 3; ++j) {
            const double w  = double(nps[j]) * (n - 1);
            const double tr = best_time([&] { ref(j); sink = static_cast<double>(p[j][1] + dp[j][1] + V[j][1]); });
            const double to = best_time([&] { opt(j); sink = static_cast<double>(p[j][1] + dp[j][1] + V[j][1]); });
            std::printf("   %6.3f -> %6.3f (x%4.1f)", tr * 1e9 / w, to * 1e9 / w, tr / to);
        }
        std::printf("\n");
    };
    row("scalar jacobi, loop",
        [&](int j) { for (std::size_t i = 0; i < nps[j]; ++i) p[j][i] = R::jacobi(n, a, b, xi[j][i]); },
        [&](int j) { for (std::size_t i = 0; i < nps[j]; ++i) p[j][i] = O::jacobi(n, a, b, xi[j][i]); });
    row("scalar jacobi_and_deriv",
        [&](int j) { for (std::size_t i = 0; i < nps[j]; ++i) R::jacobi_and_deriv(n, a, b, xi[j][i], p[j][i], dp[j][i]); },
        [&](int j) { for (std::size_t i = 0; i < nps[j]; ++i) O::jacobi_and_deriv(n, a, b, xi[j][i], p[j][i], dp[j][i]); });
    row("jacobi (array)",
        [&](int j) { R::jacobi(n, a, b, xi[j].data(), nps[j], p[j].data()); },
        [&](int j) { O::jacobi(n, a, b, xi[j].data(), nps[j], p[j].data()); });
    row("jacobi_deriv (array)",
        [&](int j) { R::jacobi_deriv(n, a, b, xi[j].data(), nps[j], dp[j].data()); },
        [&](int j) { O::jacobi_deriv(n, a, b, xi[j].data(), nps[j], dp[j].data()); });
    row("jacobi_and_deriv (array)",
        [&](int j) { R::jacobi_and_deriv<T>(n, a, b, xi[j].data(), nps[j], p[j].data(), dp[j].data()); },
        [&](int j) { O::jacobi_and_deriv<T>(n, a, b, xi[j].data(), nps[j], p[j].data(), dp[j].data()); });
    row("jacobi_all",
        [&](int j) { R::jacobi_all(n, a, b, xi[j].data(), nps[j], V[j].data()); },
        [&](int j) { O::jacobi_all(n, a, b, xi[j].data(), nps[j], V[j].data()); });

    std::printf("\n%-26s", "jacobi_zeros, GLL (us)");
    for (int m : {8, 16, 32, 64}) {
        std::vector<T> z(m);
        const double tr = best_time([&] { R::jacobi_zeros(m, 1.0, 1.0, z.data()); sink = static_cast<double>(z[0]); });
        const double to = best_time([&] { O::jacobi_zeros(m, 1.0, 1.0, z.data()); sink = static_cast<double>(z[0]); });
        std::printf("   m=%-2d %6.2f -> %6.2f (x%3.1f)", m, tr * 1e6, to * 1e6, tr / to);
    }
    std::printf("\n%-26s", "jacobi_norm2 (ns)");
    for (int nn : {4, 16, 64}) {
        const double t = best_time([&] { sink = static_cast<double>(dg::jacobi_norm2<T>(nn, 0.5, 1.5)); });
        std::printf("   n=%-3d %6.1f", nn, t * 1e9);
    }
    std::printf("\n");
}

} // namespace

/**
 * \brief  Run the benchmark in double and float.
 * \return 0
 */
int main()
{
    std::printf("bench_jacobi: best of 5 repetitions, each >= 0.1 s; public functions use '%s'\n",
                dg::jacobi_kernels());
    run<double>("double");
    run<float>("float");
    return 0;
}
