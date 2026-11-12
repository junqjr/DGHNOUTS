/**
 * \file    test_zeros.cpp
 * \brief   Unit tests of dg::jacobi_zeros(): closed forms, symmetry, interlacing,
 *          accuracy in every precision, robustness, and optionally Polylib.
 *
 * \details | Test | Check                                                         | Needs Polylib |
 *          |------|---------------------------------------------------------------|---------------|
 *          | 1    | basic properties: count, increasing order, inside \f$(-1,1)\f$, \f$m=0\f$ | no |
 *          | 2    | closed forms: Legendre \f$m\le4\f$, GLL interior points for \f$Q=5\f$, Chebyshev of both kinds | no |
 *          | 3    | symmetry \f$\alpha=\beta\f$ (exact) and reflection (D.10) (exact)       | no |
 *          | 4    | interlacing of the zeros of \f$P_m\f$ and \f$P_{m+1}\f$                  | no |
 *          | 5    | `double` and `float` against `long double`                     | no |
 *          | 6    | robustness: \f$\alpha,\beta\f$ up to 100, \f$m\le60\f$, in `double`; `float` never silently wrong | no |
 *          | 7    | invalid arguments are rejected                                | no |
 *          | 8    | Polylib's Gauss–Jacobi points (eigenvalue method)             | yes |
 *
 * \par Theory
 * \theorypdf, Section 4.
 */
#include "jacobi.hpp"
#include "test_utils.hpp"
#ifdef DGPOLY_HAVE_POLYLIB
#include "polylib.h"
#endif

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

using dgtest::Params;
using dgtest::report;
using dgtest::report_bool;

/// Parameter pairs used in DG and some general ones (includes a non-integer pair).
const std::vector<Params> params = {
    {0.0, 0.0},    // Legendre: Gauss-Legendre points
    {1.0, 1.0},    // Gauss-Lobatto-Legendre interior points
    {0.0, 1.0},    // Gauss-Radau-Legendre interior points (xi = -1 included)
    {1.0, 0.0},    // collapsed coordinates (triangles)
    {2.0, 0.0},    // collapsed coordinates (tetrahedra)
    {-0.5, -0.5},  // Chebyshev, first kind
    {0.5, 0.5},    // Chebyshev, second kind
    {0.5, 1.5},
    {0.3, -0.7},   // non-integer
    {-0.9, -0.9},  // weight close to the limit alpha, beta -> -1
};

const int m_max = 60;   ///< highest degree tested

/// \f$\pi\f$ in `long double`: the closed-form references are computed in
/// extended precision, so that their own rounding does not count as an error.
const long double pi = std::acos(-1.0L);

} // namespace

/**
 * \brief  Run all tests and print a PASS/FAIL report.
 * \return 0 if every check passes, 1 otherwise.
 */
int main()
{
    // ---------------------------------------------------------------------
    std::printf("1. Basic properties: m zeros, increasing, strictly inside (-1,1)\n");
    {
        bool ok = dg::jacobi_zeros(0, 0.0, 0.0).empty();
        for (auto [a, b] : params)
            for (int m = 1; m <= m_max; ++m) {
                const std::vector<double> z = dg::jacobi_zeros(m, a, b);
                ok = ok && static_cast<int>(z.size()) == m && z.front() > -1.0 && z.back() < 1.0;
                for (int i = 1; i < m; ++i) ok = ok && z[i] > z[i - 1];
                // pointer version gives the same zeros
                std::vector<double> zp(m);
                dg::jacobi_zeros(m, a, b, zp.data());
                ok = ok && zp == z;
            }
        report_bool("all parameter pairs, m = 0..60, pointer and vector versions agree", ok);
    }

    // ---------------------------------------------------------------------
    std::printf("\n2. Closed forms\n");
    {
        // Legendre (alpha = beta = 0)
        const std::vector<std::vector<double>> leg = {
            {0.0},
            {-1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0)},
            {-std::sqrt(0.6), 0.0, std::sqrt(0.6)},
            {-std::sqrt((3.0 + 2.0 * std::sqrt(1.2)) / 7.0), -std::sqrt((3.0 - 2.0 * std::sqrt(1.2)) / 7.0),
              std::sqrt((3.0 - 2.0 * std::sqrt(1.2)) / 7.0),  std::sqrt((3.0 + 2.0 * std::sqrt(1.2)) / 7.0)},
        };
        double err = 0.0;
        for (int m = 1; m <= 4; ++m) {
            const std::vector<double> z = dg::jacobi_zeros(m, 0.0, 0.0);
            for (int i = 0; i < m; ++i) err = std::max(err, std::fabs(z[i] - leg[m - 1][i]));
        }
        report("Legendre m = 1..4 (0, +-1/sqrt3, +-sqrt(3/5), ...)", err, 4e-16);

        // GLL, Q = 5: interior points = zeros of P_3^{1,1} = 0, +-sqrt(3/7)  (Exercise 2.1)
        const std::vector<double> z5 = dg::jacobi_zeros(3, 1.0, 1.0);
        err = std::max({std::fabs(z5[0] + std::sqrt(3.0 / 7.0)), std::fabs(z5[1]),
                        std::fabs(z5[2] - std::sqrt(3.0 / 7.0))});
        report("GLL Q = 5 interior points: zeros of P_3^{1,1} = 0, +-sqrt(3/7)", err, 4e-16);

        // Chebyshev first kind: -cos((2i+1) pi / 2m)
        long double errl = 0.0L;
        for (int m = 1; m <= m_max; ++m) {
            const std::vector<double> z = dg::jacobi_zeros(m, -0.5, -0.5);
            for (int i = 0; i < m; ++i)
                errl = std::max(errl, std::fabs(z[i] + std::cos((2 * i + 1) * pi / (2 * m))));
        }
        report("Chebyshev T_m, m = 1..60: -cos((2i+1) pi / 2m)", errl, 4e-16);

        // Chebyshev second kind: U_m proportional to P_m^{1/2,1/2}, zeros -cos((i+1) pi / (m+1))
        errl = 0.0L;
        for (int m = 1; m <= m_max; ++m) {
            const std::vector<double> z = dg::jacobi_zeros(m, 0.5, 0.5);
            for (int i = 0; i < m; ++i)
                errl = std::max(errl, std::fabs(z[i] + std::cos((i + 1) * pi / (m + 1))));
        }
        report("Chebyshev U_m, m = 1..60: -cos((i+1) pi / (m+1))", errl, 4e-16);
    }

    // ---------------------------------------------------------------------
    std::printf("\n3. Symmetry (A.1.6): exact for alpha = beta, and reflection (D.10)\n");
    {
        bool sym = true, refl = true;
        for (auto [a, b] : params)
            for (int m = 1; m <= m_max; ++m) {
                const std::vector<double> z  = dg::jacobi_zeros(m, a, b);
                const std::vector<double> zr = dg::jacobi_zeros(m, b, a);
                for (int i = 0; i < m; ++i) {
                    refl = refl && z[i] == -zr[m - 1 - i];               // bitwise
                    if (a == b) sym = sym && z[i] == -z[m - 1 - i];      // bitwise
                }
                if (a == b && m % 2 == 1) sym = sym && z[m / 2] == 0.0;
            }
        report_bool("alpha = beta: xi_i = -xi_{m-1-i} exactly, middle zero = 0 exactly", sym);
        report_bool("zeros of P^{a,b} = -(zeros of P^{b,a}), exactly", refl);
    }

    // ---------------------------------------------------------------------
    std::printf("\n4. Interlacing: zeros of P_m and P_{m+1} alternate\n");
    {
        bool ok = true;
        for (auto [a, b] : params)
            for (int m = 1; m < m_max; ++m) {
                const std::vector<double> z  = dg::jacobi_zeros(m, a, b);
                const std::vector<double> z1 = dg::jacobi_zeros(m + 1, a, b);
                for (int i = 0; i < m; ++i) ok = ok && z1[i] < z[i] && z[i] < z1[i + 1];
            }
        report_bool("xi_{i,m+1} < xi_{i,m} < xi_{i+1,m+1}, all pairs, m = 1..59", ok);
    }

    // ---------------------------------------------------------------------
    std::printf("\n5. Accuracy against long double (absolute errors, |xi| < 1)\n");
    {
        long double ed = 0, ef = 0;
        for (auto [a, b] : params)
            for (int m = 1; m <= m_max; ++m) {
                const std::vector<long double> zl = dg::jacobi_zeros<long double>(m, a, b);
                const std::vector<double>      zd = dg::jacobi_zeros<double>(m, a, b);
                const std::vector<float>       zf = dg::jacobi_zeros<float>(m, a, b);
                for (int i = 0; i < m; ++i) {
                    ed = std::max(ed, std::fabs(zd[i] - zl[i]));
                    ef = std::max(ef, std::fabs(zf[i] - zl[i]));
                }
            }
        // double: 2 ulp at |xi| ~ 1 is 2.2e-16; float: 2 ulp is 1.2e-7
        report("double, all pairs, m = 1..60", ed, 5e-16);
        report("float,  all pairs, m = 1..60", ef, 2e-7);
    }

    // ---------------------------------------------------------------------
    std::printf("\n6. Robustness: large alpha, beta (Newton far from the Chebyshev guess)\n");
    {
        const std::vector<Params> extreme = {{5, 5}, {20, 20}, {30, 30}, {21, 0}, {0, 21}, {100, 0}, {0, 100}};
        long double ed = 0;
        bool double_ok = true;
        int  float_thrown = 0, float_wrong = 0;
        for (auto [a, b] : extreme)
            for (int m = 1; m <= m_max; ++m) {
                try {
                    const std::vector<long double> zl = dg::jacobi_zeros<long double>(m, a, b);
                    const std::vector<double>      zd = dg::jacobi_zeros<double>(m, a, b);
                    for (int i = 0; i < m; ++i) ed = std::max(ed, std::fabs(zd[i] - zl[i]));
                    try {
                        const std::vector<float> zf = dg::jacobi_zeros<float>(m, a, b);
                        for (int i = 0; i < m; ++i)
                            if (std::fabs(zf[i] - zl[i]) > 2e-7) ++float_wrong;
                    } catch (const std::runtime_error&) {
                        ++float_thrown;                       // reported, never silently wrong
                    }
                } catch (const std::exception&) {
                    double_ok = false;
                }
            }
        report_bool("double and long double: no failure, alpha/beta up to 100, m = 1..60", double_ok);
        report("double vs long double, same range", ed, 5e-16);
        std::printf("         (float raised an error in %d of %zu cases: expected beyond the DG range)\n",
                    float_thrown, extreme.size() * m_max);
        report_bool("float: every result is either accurate or an exception, never wrong", float_wrong == 0);
    }

    // ---------------------------------------------------------------------
    std::printf("\n7. Invalid arguments are rejected\n");
    {
        auto throws = [](auto f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
        report_bool("m < 0, alpha <= -1, beta <= -1 throw std::invalid_argument",
                    throws([] { dg::jacobi_zeros(-1, 0.0, 0.0); }) &&
                    throws([] { dg::jacobi_zeros(3, -1.0, 0.0); }) &&
                    throws([] { dg::jacobi_zeros(3, 0.0, -1.5); }));
    }

#ifdef DGPOLY_HAVE_POLYLIB
    // ---------------------------------------------------------------------
    std::printf("\n8. Polylib Gauss-Jacobi points (independent eigenvalue method)\n");
    {
        const std::vector<Params> pl = {{0, 0}, {1, 1}, {0, 1}, {1, 0}, {2, 0}, {-0.5, -0.5}, {0.5, 1.5}};
        double err = 0.0;
        for (auto [a, b] : pl)
            for (int m = 1; m <= 40; ++m) {
                std::vector<double> zp(m), wp(m);
                polylib::zwgj(zp.data(), wp.data(), m, a, b);       // nodes = zeros of P_m^{a,b}
                const std::vector<double> z = dg::jacobi_zeros(m, a, b);
                for (int i = 0; i < m; ++i) err = std::max(err, std::fabs(z[i] - zp[i]));
            }
        report("zwgj nodes vs jacobi_zeros, m = 1..40", err, 1e-14);
    }
#else
    std::printf("\n8. Polylib comparison skipped (configure with -DPOLYLIB_DIR=...)\n");
#endif

    return dgtest::summary();
}
