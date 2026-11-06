// test_jacobi.cpp - checks dg::jacobi against Polylib and against the
// identities of Appendix A.1 (Karniadakis & Sherwin, 2nd ed.)
//
// Tests 1, 2 and 5 need Polylib as a reference; they are compiled only when
// CMake is given POLYLIB_DIR (which defines DGPOLY_HAVE_POLYLIB).
#include "jacobi.hpp"
#ifdef DGPOLY_HAVE_POLYLIB
#include "polylib.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int n_fail = 0;
int n_pass = 0;

// relative error with a floor of 1, so values near zero are compared absolutely
double rel_err(double a, double ref) { return std::fabs(a - ref) / std::max(1.0, std::fabs(ref)); }

void report(const char* name, double max_err, double tol)
{
    const bool ok = max_err <= tol;
    std::printf("  [%s] %-58s max err = %.2e (tol %.0e)\n", ok ? "PASS" : "FAIL", name, max_err, tol);
    ok ? ++n_pass : ++n_fail;
}

struct Params { double a, b; };

#ifdef DGPOLY_HAVE_POLYLIB
// integer / half-integer pairs: Polylib's gammaF only handles those
const std::vector<Params> polylib_params = {
    {0.0, 0.0},    // Legendre
    {1.0, 0.0},    // collapsed coordinates (triangles)
    {2.0, 0.0},    // collapsed coordinates (tetrahedra)
    {1.0, 1.0},    // GLL interior points
    {-0.5, -0.5},  // Chebyshev
    {0.5, 1.5},
};
#endif

// adds a non-integer pair for the tests that do not need Polylib
const std::vector<Params> all_params = {
    {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {1.0, 1.0}, {-0.5, -0.5}, {0.5, 1.5}, {0.3, -0.7},
};

const int n_max = 20;

std::vector<double> interior_points(int m)          // m points strictly inside (-1,1)
{
    std::vector<double> x(m);
    for (int i = 0; i < m; ++i) x[i] = -0.995 + 1.99 * i / (m - 1);
    return x;
}

} // namespace

int main()
{
    const std::vector<double> x = interior_points(41);

#ifdef DGPOLY_HAVE_POLYLIB
    // ---------------------------------------------------------------------
    std::printf("1. Values vs Polylib jacobfd  (recurrence A.1.3)\n");
    for (auto [a, b] : polylib_params) {
        double err = 0.0;
        std::vector<double> xx = x;
        xx.push_back(-1.0); xx.push_back(1.0);       // values are fine at the end points
        std::vector<double> ref(xx.size()), p;
        for (int n = 0; n <= n_max; ++n) {
            polylib::jacobfd(static_cast<int>(xx.size()), xx.data(), ref.data(), nullptr, n, a, b);
            dg::jacobi(n, a, b, xx, p);
            for (std::size_t i = 0; i < xx.size(); ++i) err = std::max(err, rel_err(p[i], ref[i]));
        }
        char name[80]; std::snprintf(name, sizeof name, "P_n^{%.1f,%.1f}, n = 0..%d", a, b, n_max);
        report(name, err, 1e-13);
    }

    // ---------------------------------------------------------------------
    std::printf("\n2. Derivatives vs Polylib jacobfd  (A.1.8 vs A.1.4, interior points only)\n");
    for (auto [a, b] : polylib_params) {
        double err = 0.0;
        const int np = static_cast<int>(x.size());
        std::vector<double> ref(np), dp;
        for (int n = 0; n <= n_max; ++n) {
            polylib::jacobfd(np, const_cast<double*>(x.data()), nullptr, ref.data(), n, a, b);
            dg::jacobi_deriv(n, a, b, x, dp);
            for (int i = 0; i < np; ++i) err = std::max(err, rel_err(dp[i], ref[i]));
        }
        char name[80]; std::snprintf(name, sizeof name, "dP_n^{%.1f,%.1f}/dx, n = 0..%d", a, b, n_max);
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
    std::printf("\n4. Symmetry  P_n^{a,b}(-x) = (-1)^n P_n^{b,a}(x)   (A.1.6)\n");
    for (auto [a, b] : all_params) {
        double err = 0.0;
        for (int n = 0; n <= n_max; ++n)
            for (double xi : x) {
                const double sgn = (n % 2 == 0) ? 1.0 : -1.0;
                err = std::max(err, rel_err(dg::jacobi(n, a, b, -xi), sgn * dg::jacobi(n, b, a, xi)));
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
        polylib::zwgj(z.data(), w.data(), Q, a, b);    // weights include (1-x)^a (1+x)^b

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
        std::vector<double> xx = x; xx.push_back(-1.0); xx.push_back(1.0);
        for (int n = 0; n <= n_max; ++n) {
            const double lambda = n * (n + a + b + 1.0);
            for (double xi : xx) {
                const double y   = dg::jacobi(n, a, b, xi);
                const double dy  = dg::jacobi_deriv(n, a, b, xi);
                const double d2y = (n < 1) ? 0.0
                                 : 0.5 * (n + a + b + 1.0) * dg::jacobi_deriv(n - 1, a + 1.0, b + 1.0, xi);
                const double res = (1 - xi) * (1 + xi) * d2y + (b - a - (a + b + 2) * xi) * dy + lambda * y;
                err = std::max(err, std::fabs(res) / std::max(1.0, lambda * std::fabs(y)));
            }
        }
        char name[80]; std::snprintf(name, sizeof name, "residual, a = %.1f, b = %.1f (incl. x = +/-1)", a, b);
        report(name, err, 1e-11);
    }

    // ---------------------------------------------------------------------
    std::printf("\n7. Legendre end-point derivative  L_n'(1) = n(n+1)/2\n");
    {
        double err = 0.0;
        for (int n = 0; n <= n_max; ++n)
            err = std::max(err, rel_err(dg::jacobi_deriv(n, 0.0, 0.0, 1.0), 0.5 * n * (n + 1.0)));
        report("n = 0..20  (Polylib's A.1.4 gives inf/nan here)", err, 1e-13);
    }

    // ---------------------------------------------------------------------
    std::printf("\n8. Array routines vs single-point routines\n");
    for (auto [a, b] : all_params) {
        std::vector<double> xx = x; xx.push_back(-1.0); xx.push_back(1.0);
        const std::size_t m = xx.size();
        std::vector<double> p(m), dp(m), p2(m), dp2(m);
        std::vector<double> V(m * (n_max + 1));
        dg::jacobi_all(n_max, a, b, xx.data(), m, V.data());

        double e_val = 0.0, e_der = 0.0, e_both = 0.0, e_all = 0.0;
        for (int n = 0; n <= n_max; ++n) {
            dg::jacobi(n, a, b, xx.data(), m, p.data());
            dg::jacobi_deriv(n, a, b, xx.data(), m, dp.data());
            dg::jacobi_and_deriv(n, a, b, xx.data(), m, p2.data(), dp2.data());
            for (std::size_t i = 0; i < m; ++i) {
                const double ps = dg::jacobi(n, a, b, xx[i]);
                const double ds = dg::jacobi_deriv(n, a, b, xx[i]);
                e_val  = std::max(e_val,  rel_err(p[i], ps));
                e_der  = std::max(e_der,  rel_err(dp[i], ds));
                e_both = std::max({e_both, rel_err(p2[i], ps), rel_err(dp2[i], ds)});
                e_all  = std::max(e_all,  rel_err(V[i * (n_max + 1) + n], ps));
            }
        }
        char name[80];
        std::snprintf(name, sizeof name, "jacobi(x*), a = %.1f, b = %.1f", a, b);
        report(name, e_val, 0.0);                      // same operations: bitwise identical
        std::snprintf(name, sizeof name, "jacobi_deriv(x*), a = %.1f, b = %.1f", a, b);
        report(name, e_der, 0.0);
        std::snprintf(name, sizeof name, "jacobi_and_deriv (one pass), a = %.1f, b = %.1f", a, b);
        report(name, e_both, 1e-12);                   // derivative from a different recurrence
        std::snprintf(name, sizeof name, "jacobi_all (Vandermonde), a = %.1f, b = %.1f", a, b);
        report(name, e_all, 0.0);
    }

    std::printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}
