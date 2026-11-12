/**
 * \file    test_jacobi.cpp
 * \brief   Unit tests of the Jacobi module against the identities of
 *          Karniadakis & Sherwin, Appendix A.1, and optionally against Polylib.
 *
 * \details Degrees \f$n = 0,\dots,20\f$; parameter pairs
 *          \f$(\alpha,\beta) \in \{(0,0),(1,0),(2,0),(1,1),(-\tfrac12,-\tfrac12),
 *          (\tfrac12,\tfrac32),(0.3,-0.7)\}\f$. Tests 1 to 8 run in `double`;
 *          test 9 compares the three precisions.
 *
 *          | Test | Identity                                     | Needs Polylib |
 *          |------|----------------------------------------------|---------------|
 *          | 1    | values vs Polylib `jacobfd`                  | yes |
 *          | 2    | interior derivatives vs Polylib `jacobfd`    | yes |
 *          | 3    | end-point value (A.1.5)                      | no  |
 *          | 4    | symmetry (A.1.6)                             | no  |
 *          | 5    | orthogonality and norm (A.1.7), Polylib `zwgj` quadrature | yes |
 *          | 6    | differential equation (A.1.1), \f$y''\f$ from (D.1), incl. \f$\xi=\pm1\f$ | no |
 *          | 7    | \f$L_n'(1) = n(n+1)/2\f$ (D.6)               | no  |
 *          | 8    | array routines vs single-point routines, `nullptr` outputs | no |
 *          | 9    | `float` and `double` vs `long double`; (A.1.5) in `long double` | no |
 *
 *          Tests 1, 2 and 5 are compiled only when CMake is given `POLYLIB_DIR`
 *          (which defines `DGPOLY_HAVE_POLYLIB`). The program returns 0 if all
 *          tests pass and 1 otherwise, so `ctest` reports failures.
 *
 * \par Theory
 * \theorypdf, Section 3.
 */
#include "jacobi.hpp"
#ifdef DGPOLY_HAVE_POLYLIB
#include "polylib.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

int n_fail = 0;   ///< number of failed checks
int n_pass = 0;   ///< number of passed checks

/**
 * \brief  Relative error with a floor of 1, so values near zero are compared absolutely.
 * \param[in] a    computed value
 * \param[in] ref  reference value
 * \return    \f$|a-\mathrm{ref}| / \max(1,|\mathrm{ref}|)\f$
 */
long double rel_err(long double a, long double ref)
{
    return std::fabs(a - ref) / std::max(1.0L, std::fabs(ref));
}

/**
 * \brief  Print one PASS/FAIL line and update the counters.
 * \param[in] name     description of the check
 * \param[in] max_err  largest error found
 * \param[in] tol      tolerance; the check passes if max_err <= tol
 */
void report(const char* name, long double max_err, double tol)
{
    const bool ok = max_err <= tol;
    std::printf("  [%s] %-58s max err = %.2Le (tol %.0e)\n", ok ? "PASS" : "FAIL", name, max_err, tol);
    ok ? ++n_pass : ++n_fail;
}

/// A pair of Jacobi weight parameters.
struct Params {
    double a;   ///< \f$\alpha\f$
    double b;   ///< \f$\beta\f$
};

#ifdef DGPOLY_HAVE_POLYLIB
/// Parameter pairs for the Polylib comparisons: integer or half-integer only,
/// since Polylib's `gammaF` handles nothing else.
const std::vector<Params> polylib_params = {
    {0.0, 0.0},    // Legendre
    {1.0, 0.0},    // collapsed coordinates (triangles)
    {2.0, 0.0},    // collapsed coordinates (tetrahedra)
    {1.0, 1.0},    // GLL interior points
    {-0.5, -0.5},  // Chebyshev
    {0.5, 1.5},
};
#endif

/// Parameter pairs for the tests that do not need Polylib (includes a non-integer pair).
const std::vector<Params> all_params = {
    {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {1.0, 1.0}, {-0.5, -0.5}, {0.5, 1.5}, {0.3, -0.7},
};

const int n_max = 20;   ///< highest degree tested

/**
 * \brief  Equispaced points strictly inside \f$(-1,1)\f$, from -0.995 to 0.995.
 *
 * The points are rounded to `float`, so that they are represented exactly in
 * every precision and test 9 measures only the error of the algorithm.
 *
 * \param[in] m  number of points, \f$m \ge 2\f$
 * \return    the points \f$\xi_i\f$
 */
std::vector<double> interior_points(int m)
{
    std::vector<double> xi(m);
    for (int i = 0; i < m; ++i)
        xi[i] = static_cast<float>(-0.995 + 1.99 * i / (m - 1));
    return xi;
}

/**
 * \brief  The points of interior_points() plus the end points \f$\xi=\pm1\f$.
 * \param[in] m  number of interior points
 * \return    the points \f$\xi_i\f$
 */
std::vector<double> points_with_ends(int m)
{
    std::vector<double> xi = interior_points(m);
    xi.push_back(-1.0);
    xi.push_back(1.0);
    return xi;
}

/**
 * \brief  Largest relative error of the precision `T` against `long double`,
 *         for the values, the derivatives and the norms, \f$n=0,\dots,20\f$.
 * \tparam     T      `float` or `double`
 * \param[in]  a      \f$\alpha\f$
 * \param[in]  b      \f$\beta\f$
 * \param[in]  xi     points (exactly representable in `float`)
 * \param[out] e_val  error of jacobi()
 * \param[out] e_der  error of jacobi_deriv()
 * \param[out] e_nrm  error of jacobi_norm2()
 */
template <typename T>
void precision_errors(double a, double b, const std::vector<double>& xi,
                      long double& e_val, long double& e_der, long double& e_nrm)
{
    e_val = e_der = e_nrm = 0.0L;
    for (int n = 0; n <= n_max; ++n) {
        for (double s : xi) {
            const long double xl = s;
            const T           xt = static_cast<T>(s);
            e_val = std::max(e_val, rel_err(dg::jacobi(n, a, b, xt), dg::jacobi(n, a, b, xl)));
            e_der = std::max(e_der, rel_err(dg::jacobi_deriv(n, a, b, xt), dg::jacobi_deriv(n, a, b, xl)));
        }
        const long double hl = dg::jacobi_norm2<long double>(n, a, b);
        e_nrm = std::max(e_nrm, std::fabs(dg::jacobi_norm2<T>(n, a, b) - hl) / hl);
    }
}

} // namespace

/**
 * \brief  Run all tests and print a PASS/FAIL report.
 * \return 0 if every check passes, 1 otherwise.
 */
int main()
{
    const std::vector<double> xi_in  = interior_points(41);    // strictly inside (-1,1)
    const std::vector<double> xi_all = points_with_ends(41);   // plus xi = -1, +1

#ifdef DGPOLY_HAVE_POLYLIB
    // ---------------------------------------------------------------------
    std::printf("1. Values vs Polylib jacobfd  (recurrence A.1.3)\n");
    for (auto [a, b] : polylib_params) {
        double err = 0.0;
        std::vector<double> xi = xi_all;                // values are fine at the end points
        std::vector<double> ref(xi.size()), p;
        for (int n = 0; n <= n_max; ++n) {
            polylib::jacobfd(static_cast<int>(xi.size()), xi.data(), ref.data(), nullptr, n, a, b);
            dg::jacobi(n, a, b, xi, p);
            for (std::size_t i = 0; i < xi.size(); ++i) err = std::max<double>(err, rel_err(p[i], ref[i]));
        }
        char name[80]; std::snprintf(name, sizeof name, "P_n^{%.1f,%.1f}, n = 0..%d", a, b, n_max);
        report(name, err, 1e-13);
    }

    // ---------------------------------------------------------------------
    std::printf("\n2. Derivatives vs Polylib jacobfd  (A.1.8 vs A.1.4, interior points only)\n");
    for (auto [a, b] : polylib_params) {
        double err = 0.0;
        std::vector<double> xi = xi_in;
        const int np = static_cast<int>(xi.size());
        std::vector<double> ref(np), dp;
        for (int n = 0; n <= n_max; ++n) {
            polylib::jacobfd(np, xi.data(), nullptr, ref.data(), n, a, b);
            dg::jacobi_deriv(n, a, b, xi, dp);
            for (int i = 0; i < np; ++i) err = std::max<double>(err, rel_err(dp[i], ref[i]));
        }
        char name[80]; std::snprintf(name, sizeof name, "dP_n^{%.1f,%.1f}/dxi, n = 0..%d", a, b, n_max);
        report(name, err, 1e-11);
    }

#else
    std::printf("1-2. Polylib comparisons skipped (configure with -DPOLYLIB_DIR=...)\n");
#endif

    // ---------------------------------------------------------------------
    std::printf("\n3. End-point value  P_n^{a,b}(1) = Gamma(n+a+1) / (Gamma(a+1) n!)   (A.1.5)\n");
    for (auto [a, b] : all_params) {
        double err = 0.0;
        for (int n = 0; n <= n_max; ++n) {
            const double ref = std::exp(std::lgamma(n + a + 1.0) - std::lgamma(a + 1.0) - std::lgamma(n + 1.0));
            err = std::max(err, std::fabs(dg::jacobi(n, a, b, 1.0) - ref) / std::fabs(ref));
        }
        char name[80]; std::snprintf(name, sizeof name, "a = %.1f, b = %.1f", a, b);
        report(name, err, 1e-13);
    }

    // ---------------------------------------------------------------------
    std::printf("\n4. Symmetry  P_n^{a,b}(-xi) = (-1)^n P_n^{b,a}(xi)   (A.1.6)\n");
    for (auto [a, b] : all_params) {
        double err = 0.0;
        for (int n = 0; n <= n_max; ++n)
            for (double s : xi_in) {
                const double sgn = (n % 2 == 0) ? 1.0 : -1.0;
                err = std::max<double>(err, rel_err(dg::jacobi(n, a, b, -s), sgn * dg::jacobi(n, b, a, s)));
            }
        char name[80]; std::snprintf(name, sizeof name, "a = %.1f, b = %.1f", a, b);
        report(name, err, 1e-13);
    }

#ifdef DGPOLY_HAVE_POLYLIB
    // ---------------------------------------------------------------------
    std::printf("\n5. Orthogonality and norm (A.1.7), with Polylib Gauss-Jacobi quadrature\n");
    for (auto [a, b] : polylib_params) {
        const int Q = n_max + 2;                       // exact for degree <= 2Q-1 >= 2 n_max
        std::vector<double> z(Q), w(Q);
        polylib::zwgj(z.data(), w.data(), Q, a, b);    // weights include (1-xi)^a (1+xi)^b

        double err = 0.0;
        for (int n = 0; n <= n_max; ++n)
            for (int m = 0; m <= n; ++m) {
                double s = 0.0;
                for (int i = 0; i < Q; ++i)
                    s += w[i] * dg::jacobi(n, a, b, z[i]) * dg::jacobi(m, a, b, z[i]);
                const double ref = (n == m) ? dg::jacobi_norm2(n, a, b) : 0.0;
                err = std::max(err, std::fabs(s - ref) / dg::jacobi_norm2(n, a, b));
            }
        char name[80]; std::snprintf(name, sizeof name, "(P_n, P_m) = h_n delta_nm, a = %.1f, b = %.1f", a, b);
        report(name, err, 1e-12);
    }

#else
    std::printf("\n5. Orthogonality skipped (needs a quadrature: Polylib for now)\n");
#endif

    // ---------------------------------------------------------------------
    std::printf("\n6. Sturm-Liouville equation (A.1.1), y'' from A.1.8 applied twice\n");
    for (auto [a, b] : all_params) {
        double err = 0.0;
        for (int n = 0; n <= n_max; ++n) {
            const double lambda = n * (n + a + b + 1.0);
            for (double s : xi_all) {
                const double y   = dg::jacobi(n, a, b, s);
                const double dy  = dg::jacobi_deriv(n, a, b, s);
                const double d2y = (n < 1) ? 0.0
                                 : 0.5 * (n + a + b + 1.0) * dg::jacobi_deriv(n - 1, a + 1.0, b + 1.0, s);
                const double res = (1 - s) * (1 + s) * d2y + (b - a - (a + b + 2) * s) * dy + lambda * y;
                err = std::max(err, std::fabs(res) / std::max(1.0, lambda * std::fabs(y)));
            }
        }
        char name[80]; std::snprintf(name, sizeof name, "residual, a = %.1f, b = %.1f (incl. xi = +/-1)", a, b);
        report(name, err, 1e-11);
    }

    // ---------------------------------------------------------------------
    std::printf("\n7. Legendre end-point derivative  L_n'(1) = n(n+1)/2\n");
    {
        double err = 0.0;
        for (int n = 0; n <= n_max; ++n)
            err = std::max<double>(err, rel_err(dg::jacobi_deriv(n, 0.0, 0.0, 1.0), 0.5 * n * (n + 1.0)));
        report("n = 0..20  (Polylib's A.1.4 gives inf/nan here)", err, 1e-13);
    }

    // ---------------------------------------------------------------------
    std::printf("\n8. Array routines vs single-point routines\n");
    for (auto [a, b] : all_params) {
        const std::vector<double>& xi = xi_all;
        const std::size_t m = xi.size();
        std::vector<double> p(m), dp(m), p2(m), dp2(m), p3(m), dp3(m);
        std::vector<double> V(m * (n_max + 1));
        dg::jacobi_all(n_max, a, b, xi.data(), m, V.data());

        long double e_val = 0, e_der = 0, e_both = 0, e_all = 0, e_null = 0;
        for (int n = 0; n <= n_max; ++n) {
            dg::jacobi(n, a, b, xi.data(), m, p.data());
            dg::jacobi_deriv(n, a, b, xi.data(), m, dp.data());
            dg::jacobi_and_deriv(n, a, b, xi.data(), m, p2.data(), dp2.data());
            dg::jacobi_and_deriv(n, a, b, xi.data(), m, p3.data(), nullptr);     // values only
            dg::jacobi_and_deriv(n, a, b, xi.data(), m, nullptr, dp3.data());    // derivatives only
            for (std::size_t i = 0; i < m; ++i) {
                const double ps = dg::jacobi(n, a, b, xi[i]);
                const double ds = dg::jacobi_deriv(n, a, b, xi[i]);
                e_val  = std::max(e_val,  rel_err(p[i], ps));
                e_der  = std::max(e_der,  rel_err(dp[i], ds));
                e_both = std::max({e_both, rel_err(p2[i], ps), rel_err(dp2[i], ds)});
                e_all  = std::max(e_all,  rel_err(V[i * (n_max + 1) + n], ps));
                e_null = std::max({e_null, rel_err(p3[i], p2[i]), rel_err(dp3[i], dp2[i])});
            }
        }
        char name[80];
        std::snprintf(name, sizeof name, "jacobi(xi*), a = %.1f, b = %.1f", a, b);
        report(name, e_val, 0.0);                      // same operations: bitwise identical
        std::snprintf(name, sizeof name, "jacobi_deriv(xi*), a = %.1f, b = %.1f", a, b);
        report(name, e_der, 0.0);
        std::snprintf(name, sizeof name, "jacobi_and_deriv (one pass), a = %.1f, b = %.1f", a, b);
        report(name, e_both, 1e-12);                   // derivative from a different recurrence
        std::snprintf(name, sizeof name, "jacobi_and_deriv, nullptr outputs, a = %.1f, b = %.1f", a, b);
        report(name, e_null, 0.0);
        std::snprintf(name, sizeof name, "jacobi_all (Vandermonde), a = %.1f, b = %.1f", a, b);
        report(name, e_all, 0.0);
    }

    // ---------------------------------------------------------------------
    std::printf("\n9. Precision: float and double vs long double (values, derivatives, norms)\n");
    std::printf("   long double: %d mantissa bits, double: %d, float: %d\n",
                std::numeric_limits<long double>::digits, std::numeric_limits<double>::digits,
                std::numeric_limits<float>::digits);
    {
        long double ef_val = 0, ef_der = 0, ef_nrm = 0, ed_val = 0, ed_der = 0, ed_nrm = 0;
        long double ld_end = 0;
        for (auto [a, b] : all_params) {
            long double v, d, h;
            precision_errors<float>(a, b, xi_all, v, d, h);
            ef_val = std::max(ef_val, v); ef_der = std::max(ef_der, d); ef_nrm = std::max(ef_nrm, h);
            precision_errors<double>(a, b, xi_all, v, d, h);
            ed_val = std::max(ed_val, v); ed_der = std::max(ed_der, d); ed_nrm = std::max(ed_nrm, h);

            // (A.1.5) in long double: checks that the long double instantiation
            // is genuinely more accurate than double, not just double in disguise
            for (int n = 0; n <= n_max; ++n) {
                const long double al = a;
                const long double ref = std::exp(std::lgamma(n + al + 1.0L) - std::lgamma(al + 1.0L)
                                                 - std::lgamma(n + 1.0L));
                ld_end = std::max(ld_end, std::fabs(dg::jacobi(n, a, b, 1.0L) - ref) / ref);
            }
        }
        // float: 2^-24 = 6e-8 per operation, amplified by the recurrence (values
        // ~5e-6, derivatives ~7e-5 at n = 20) and by the cancellation between the
        // large lgamma terms of the norm (~1e-5). Measured, not theoretical.
        report("float  vs long double: values,      all (a,b)", ef_val, 1e-5);
        report("float  vs long double: derivatives, all (a,b)", ef_der, 1e-4);
        report("float  vs long double: norms,       all (a,b)", ef_nrm, 1e-4);
        report("double vs long double: values,      all (a,b)", ed_val, 1e-13);
        report("double vs long double: derivatives, all (a,b)", ed_der, 1e-12);
        report("double vs long double: norms,       all (a,b)", ed_nrm, 1e-13);
        // x86 80-bit: eps 1e-19; Arm IEEE quad: eps 2e-34. Either way below double's 2e-16.
        report("long double end-point value (A.1.5),  all (a,b)", ld_end, 1e-16);
    }

    std::printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
