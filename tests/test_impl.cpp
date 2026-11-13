/**
 * \file    test_impl.cpp
 * \brief   The two implementations of the Jacobi module, `dg::reference` and
 *          `dg::optimized`, must agree; the public functions must use the one
 *          chosen at build time.
 *
 * \details The implementations evaluate the same equations in different ways
 *          (division by \f$a_n^1\f$ vs. normalised coefficients, one point at a
 *          time vs. register blocks, with or without fused multiply-add), so
 *          they agree up to round-off, not bitwise. The tolerances below are a
 *          few times the accuracy of each against `long double` (theory
 *          document, Tables 1 and 3).
 *
 *          | Test | Check |
 *          |------|-------|
 *          | 1    | every evaluation routine of both implementations against `long double`; the optimised one must not be less accurate |
 *          | 2    | zeros: optimized vs reference, three precisions, \f$\alpha,\beta\f$ up to 21 |
 *          | 3    | the public functions are bitwise those of the chosen implementation |
 *          | 4    | both implementations reject invalid arguments |
 *
 * \par Theory
 * \theorypdf, Section 2.6.
 */
#include "jacobi.hpp"
#include "jacobi_impl.hpp"
#include "test_utils.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using dgtest::Params;
using dgtest::rel_err;
using dgtest::report;
using dgtest::report_bool;

/// Parameter pairs: DG values, Chebyshev, non-integer, close to the limit -1.
const std::vector<Params> params = {
    {0.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}, {2.0, 0.0},
    {-0.5, -0.5}, {0.5, 1.5}, {0.3, -0.7}, {-0.9, -0.9},
};

const int n_max = 24;   ///< highest degree compared

/**
 * \brief  Points in \f$[-1,1]\f$, including both end points: 100 points,
 *         so that the optimised routines run full blocks and a tail block.
 * \return the points
 */
template <typename T>
std::vector<T> points()
{
    std::vector<T> xi(100);
    for (std::size_t i = 0; i < xi.size(); ++i) xi[i] = T(-1) + T(2) * T(i) / T(xi.size() - 1);
    return xi;
}

/// Largest errors of one implementation against the `long double` reference.
struct Errors {
    long double val = 0;   ///< values (scalar, array, one-pass, Vandermonde)
    long double der = 0;   ///< derivatives ((A.1.8) and one-pass, scalar and array)
};

/**
 * \brief  Errors of the implementation `Impl` in precision `T` against
 *         `long double` (computed with the reference implementation), all
 *         routines, all parameter pairs, \f$n=0,\dots,24\f$.
 * \tparam T     floating-point type, `float` or `double`
 * \tparam Impl  `RefImpl` or `OptImpl`
 * \return the largest relative errors
 */
template <typename T, typename Impl>
Errors errors_vs_long_double()
{
    const std::vector<T> xi = points<T>();
    const std::size_t m = xi.size();
    std::vector<T> p(m), d(m), q(m), dq(m), V(m * (n_max + 1));
    Errors e;
    for (auto [a, b] : params) {
        Impl::all(n_max, a, b, xi.data(), m, V.data());
        for (int n = 0; n <= n_max; ++n) {
            Impl::values(n, a, b, xi.data(), m, p.data());
            Impl::derivs(n, a, b, xi.data(), m, d.data());
            Impl::both(n, a, b, xi.data(), m, q.data(), dq.data());
            for (std::size_t i = 0; i < m; ++i) {
                const long double x  = xi[i];
                const long double pr = dg::reference::jacobi(n, a, b, x);
                const long double dr = dg::reference::jacobi_deriv(n, a, b, x);
                T p1, d1;
                Impl::both1(n, a, b, xi[i], p1, d1);
                e.val = std::max({e.val, rel_err(p[i], pr), rel_err(q[i], pr), rel_err(p1, pr),
                                  rel_err(Impl::value1(n, a, b, xi[i]), pr),
                                  rel_err(V[i * (n_max + 1) + n], pr)});
                e.der = std::max({e.der, rel_err(d[i], dr), rel_err(dq[i], dr), rel_err(d1, dr),
                                  rel_err(Impl::deriv1(n, a, b, xi[i]), dr)});
            }
        }
    }
    return e;
}

/// The reference implementation, as a type (to pass it as a template argument).
struct RefImpl {
    /// \cond INTERNAL
    template <class... A> static auto value1(A... a) { return dg::reference::jacobi(a...); }
    template <class... A> static auto deriv1(A... a) { return dg::reference::jacobi_deriv(a...); }
    template <class T> static void both1(int n, double a, double b, T x, T& p, T& d) { dg::reference::jacobi_and_deriv(n, a, b, x, p, d); }
    template <class T> static void values(int n, double a, double b, const T* x, std::size_t m, T* p) { dg::reference::jacobi(n, a, b, x, m, p); }
    template <class T> static void derivs(int n, double a, double b, const T* x, std::size_t m, T* d) { dg::reference::jacobi_deriv(n, a, b, x, m, d); }
    template <class T> static void both(int n, double a, double b, const T* x, std::size_t m, T* p, T* d) { dg::reference::jacobi_and_deriv<T>(n, a, b, x, m, p, d); }
    template <class T> static void all(int n, double a, double b, const T* x, std::size_t m, T* V) { dg::reference::jacobi_all(n, a, b, x, m, V); }
    /// \endcond
};

/// The optimised implementation, as a type (to pass it as a template argument).
struct OptImpl {
    /// \cond INTERNAL
    template <class... A> static auto value1(A... a) { return dg::optimized::jacobi(a...); }
    template <class... A> static auto deriv1(A... a) { return dg::optimized::jacobi_deriv(a...); }
    template <class T> static void both1(int n, double a, double b, T x, T& p, T& d) { dg::optimized::jacobi_and_deriv(n, a, b, x, p, d); }
    template <class T> static void values(int n, double a, double b, const T* x, std::size_t m, T* p) { dg::optimized::jacobi(n, a, b, x, m, p); }
    template <class T> static void derivs(int n, double a, double b, const T* x, std::size_t m, T* d) { dg::optimized::jacobi_deriv(n, a, b, x, m, d); }
    template <class T> static void both(int n, double a, double b, const T* x, std::size_t m, T* p, T* d) { dg::optimized::jacobi_and_deriv<T>(n, a, b, x, m, p, d); }
    template <class T> static void all(int n, double a, double b, const T* x, std::size_t m, T* V) { dg::optimized::jacobi_all(n, a, b, x, m, V); }
    /// \endcond
};

/**
 * \brief  Report both implementations against `long double` in precision `T`,
 *         and check that the optimised one is not less accurate.
 * \tparam    T        `float` or `double`
 * \param[in] tname    name of the type, for the report
 * \param[in] tol_val  tolerance on the values
 * \param[in] tol_der  tolerance on the derivatives
 */
template <typename T>
void check_precision(const char* tname, double tol_val, double tol_der)
{
    const Errors r = errors_vs_long_double<T, RefImpl>();
    const Errors o = errors_vs_long_double<T, OptImpl>();
    char name[96];
    std::snprintf(name, sizeof name, "%s, reference: values", tname);   report(name, r.val, tol_val);
    std::snprintf(name, sizeof name, "%s, optimized: values", tname);   report(name, o.val, tol_val);
    std::snprintf(name, sizeof name, "%s, reference: derivatives", tname); report(name, r.der, tol_der);
    std::snprintf(name, sizeof name, "%s, optimized: derivatives", tname); report(name, o.der, tol_der);
    // regression guard: the optimised implementation must not lose accuracy
    const long double eps = std::numeric_limits<T>::epsilon();
    std::snprintf(name, sizeof name, "%s: optimized error <= 3 x reference error", tname);
    report_bool(name, o.val <= 3 * std::max(r.val, eps) && o.der <= 3 * std::max(r.der, eps));
}

/**
 * \brief  Largest absolute difference between the zeros of the two
 *         implementations, \f$m = 1,\dots,40\f$, in precision `T`.
 * \tparam T  floating-point type
 * \param[in] ps  parameter pairs
 * \return    \f$\max|\xi^{opt}-\xi^{ref}|\f$
 */
template <typename T>
long double compare_zeros(const std::vector<Params>& ps)
{
    long double e = 0;
    for (auto [a, b] : ps)
        for (int m = 1; m <= 40; ++m) {
            std::vector<T> zr(m), zo(m);
            dg::reference::jacobi_zeros(m, a, b, zr.data());
            dg::optimized::jacobi_zeros(m, a, b, zo.data());
            for (int i = 0; i < m; ++i) e = std::max(e, std::fabs((long double)zo[i] - zr[i]));
        }
    return e;
}

} // namespace

/**
 * \brief  Run all tests and print a PASS/FAIL report.
 * \return 0 if every check passes, 1 otherwise.
 */
int main()
{
    std::printf("Public functions use the '%s' implementation\n", dg::jacobi_kernels());

    // ---------------------------------------------------------------------
    std::printf("\n1. Both implementations against long double: all evaluation routines, n = 0..24\n");
    {
        // measured: double values ~7e-14, derivatives ~1e-12; float ~7e-5, ~5e-4
        check_precision<double>("double", 2e-13, 3e-12);
        check_precision<float>("float ", 2e-4, 1e-3);
        // long double: no higher precision available, compare the two directly
        long double ev = 0, ed = 0;
        const std::vector<long double> xi = points<long double>();
        for (auto [a, b] : params)
            for (int n = 0; n <= n_max; ++n)
                for (long double x : xi) {
                    ev = std::max(ev, rel_err(dg::optimized::jacobi(n, a, b, x), dg::reference::jacobi(n, a, b, x)));
                    ed = std::max(ed, rel_err(dg::optimized::jacobi_deriv(n, a, b, x), dg::reference::jacobi_deriv(n, a, b, x)));
                }
        report("long double: optimized vs reference, values", ev, 1e-16);
        report("long double: optimized vs reference, derivatives", ed, 1e-14);
    }

    // ---------------------------------------------------------------------
    std::printf("\n2. optimized vs reference: zeros (absolute), m = 1..40\n");
    {
        std::vector<Params> ps = params;
        ps.push_back({5.0, 5.0});
        ps.push_back({21.0, 0.0});
        ps.push_back({0.0, 21.0});
        report("double, alpha/beta up to 21", compare_zeros<double>(ps), 4e-16);
        report("long double, alpha/beta up to 21", compare_zeros<long double>(ps), 4e-19);
        report("float, DG parameters", compare_zeros<float>(params), 2.5e-7);
    }

    // ---------------------------------------------------------------------
    std::printf("\n3. The public functions are those of the chosen implementation\n");
    {
        const bool use_ref = std::strcmp(dg::jacobi_kernels(), "reference") == 0;
        const std::vector<double> xi = points<double>();
        const std::size_t m = xi.size();
        bool same = true;
        for (auto [a, b] : params)
            for (int n = 0; n <= n_max; n += 3) {
                std::vector<double> p(m), q(m), z(n), zz(n);
                dg::jacobi(n, a, b, xi.data(), m, p.data());
                if (use_ref) dg::reference::jacobi(n, a, b, xi.data(), m, q.data());
                else         dg::optimized::jacobi(n, a, b, xi.data(), m, q.data());
                same = same && p == q;
                dg::jacobi_zeros(n, a, b, z.data());
                if (use_ref) dg::reference::jacobi_zeros(n, a, b, zz.data());
                else         dg::optimized::jacobi_zeros(n, a, b, zz.data());
                same = same && z == zz;
            }
        report_bool("dg::jacobi and dg::jacobi_zeros match the chosen implementation bitwise", same);
    }

    // ---------------------------------------------------------------------
    std::printf("\n4. Both implementations reject invalid arguments\n");
    {
        auto throws = [](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
        double x = 0.0, y = 0.0;
        bool ok = true;
        ok = ok && throws([&] { dg::reference::jacobi(-1, 0.0, 0.0, 0.5); });
        ok = ok && throws([&] { dg::optimized::jacobi(-1, 0.0, 0.0, 0.5); });
        ok = ok && throws([&] { dg::reference::jacobi(3, -1.0, 0.0, &x, 1, &y); });
        ok = ok && throws([&] { dg::optimized::jacobi(3, -1.0, 0.0, &x, 1, &y); });
        ok = ok && throws([&] { dg::reference::jacobi_zeros(3, 0.0, -2.0, &y); });
        ok = ok && throws([&] { dg::optimized::jacobi_zeros(3, 0.0, -2.0, &y); });
        report_bool("n < 0, alpha <= -1, beta <= -1: std::invalid_argument in both", ok);
    }

    return dgtest::summary();
}
