/**
 * \file    jacobi.cpp
 * \brief   Implementation of the Jacobi module (see jacobi.hpp).
 *
 * \details The public functions are documented in jacobi.hpp. This file
 *          documents the internal helpers and, inside each function body,
 *          which equation each step implements. Equation numbers are those of
 *          Karniadakis & Sherwin, Appendix A.1, or (D.n) for equations of the
 *          theory document \theorypdf.
 */
#include "jacobi.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace dg {

namespace {

/**
 * \brief  Validate the arguments common to all Jacobi routines.
 * \param[in] n      degree; must be \f$\ge 0\f$
 * \param[in] alpha  must be \f$> -1\f$
 * \param[in] beta   must be \f$> -1\f$
 * \throws std::invalid_argument otherwise. For \f$\alpha\le-1\f$ or
 *         \f$\beta\le-1\f$ the weight \f$(1-x)^\alpha(1+x)^\beta\f$ is not
 *         integrable and the polynomials are not orthogonal.
 */
void check_args(int n, double alpha, double beta)
{
    if (n < 0)
        throw std::invalid_argument("jacobi: n must be >= 0, got " + std::to_string(n));
    if (alpha <= -1.0 || beta <= -1.0)
        throw std::invalid_argument("jacobi: alpha and beta must be > -1");
}

/**
 * \brief Coefficients \f$a_k^1,\dots,a_k^4\f$ of one step of the recurrence (A.1.3).
 *
 * One step computes
 * \f$a_k^1 P_{k+1} = (a_k^2 + a_k^3 x)\,P_k - a_k^4 P_{k-1}\f$.
 */
struct RecCoef {
    double a1;  ///< \f$a_k^1 = 2(k+1)(k+\alpha+\beta+1)(2k+\alpha+\beta)\f$
    double a2;  ///< \f$a_k^2 = (2k+\alpha+\beta+1)(\alpha^2-\beta^2)\f$
    double a3;  ///< \f$a_k^3 = (2k+\alpha+\beta)(2k+\alpha+\beta+1)(2k+\alpha+\beta+2)\f$
    double a4;  ///< \f$a_k^4 = 2(k+\alpha)(k+\beta)(2k+\alpha+\beta+2)\f$
};

/**
 * \brief  Coefficients of step \f$k\f$ of the recurrence (A.1.3).
 *
 * The index \f$k\f$ is exactly the book's \f$n\f$ in (A.1.3); the step
 * produces \f$P_{k+1}\f$ and is used for \f$k = 1,\dots,n-1\f$.
 * For \f$k\ge1\f$ and \f$\alpha,\beta>-1\f$, \f$a_k^1 > 0\f$.
 *
 * \param[in] k      recurrence step, \f$k \ge 1\f$
 * \param[in] alpha  \f$\alpha\f$
 * \param[in] beta   \f$\beta\f$
 * \return    the four coefficients
 *
 * \par Theory
 * \theorypdf, Section 2.2, Eq. (A.1.3).
 */
inline RecCoef rec_coef(int k, double alpha, double beta)
{
    const double apb = alpha + beta;
    const double c   = 2.0 * k + apb;                    // 2k + a + b
    return { 2.0 * (k + 1) * (k + apb + 1.0) * c,
             (c + 1.0) * (alpha * alpha - beta * beta),
             c * (c + 1.0) * (c + 2.0),
             2.0 * (k + alpha) * (k + beta) * (c + 2.0) };
}

/**
 * \brief  First-degree polynomial
 *         \f$P_1^{\alpha,\beta}(x) = \tfrac12\bigl[\alpha-\beta+(\alpha+\beta+2)x\bigr]\f$,
 *         the second starting value of (A.1.3).
 *
 * Its derivative, \f$\tfrac12(\alpha+\beta+2)\f$, is the starting value (D.3)
 * of the derivative recurrence.
 *
 * \param[in] alpha  \f$\alpha\f$
 * \param[in] beta   \f$\beta\f$
 * \param[in] x      evaluation point
 * \return    \f$P_1^{\alpha,\beta}(x)\f$
 */
inline double p1(double alpha, double beta, double x)
{
    return 0.5 * (alpha - beta + (alpha + beta + 2.0) * x);
}

} // namespace

// ===========================================================================
// Single point
// ===========================================================================
double jacobi(int n, double alpha, double beta, double x)
{
    check_args(n, alpha, beta);

    // Starting values of (A.1.3)
    double p_km1 = 1.0;                                  // P_0
    if (n == 0) return p_km1;
    double p_k = p1(alpha, beta, x);                     // P_1

    // (A.1.3): P_{k+1} from P_k and P_{k-1}, k = 1 .. n-1
    for (int k = 1; k < n; ++k) {
        const RecCoef c = rec_coef(k, alpha, beta);
        const double p_kp1 = ((c.a2 + c.a3 * x) * p_k - c.a4 * p_km1) / c.a1;
        p_km1 = p_k;
        p_k   = p_kp1;
    }
    return p_k;                                          // P_n
}

double jacobi_deriv(int n, double alpha, double beta, double x)
{
    check_args(n, alpha, beta);
    if (n == 0) return 0.0;                              // P_0 = 1 is constant

    // (A.1.8): d/dx P_n^{a,b} = 1/2 (n+a+b+1) P_{n-1}^{a+1,b+1}
    // Valid at x = +/-1, unlike (A.1.4), which divides by 1 - x^2.
    return 0.5 * (n + alpha + beta + 1.0) * jacobi(n - 1, alpha + 1.0, beta + 1.0, x);
}

// ===========================================================================
// Arrays of points
//
// Loop order: degree k outside (sequential: P_{k+1} needs P_k and P_{k-1}),
// points i inside (independent: the compiler vectorises this loop with SIMD).
// The recurrence coefficients are computed once per k, not once per point.
// ===========================================================================
void jacobi(int n, double alpha, double beta,
            const double* x, std::size_t np, double* p)
{
    check_args(n, alpha, beta);
    if (np == 0) return;

    if (n == 0) {
        for (std::size_t i = 0; i < np; ++i) p[i] = 1.0;
        return;
    }

    // Storage: the output p holds the current P_k, the work array pm1 holds
    // P_{k-1}. Both are updated in place at every step.
    std::vector<double> pm1(np, 1.0);                    // P_{k-1}, starts as P_0
    for (std::size_t i = 0; i < np; ++i) p[i] = p1(alpha, beta, x[i]);   // P_k = P_1

    // (A.1.3), k = 1 .. n-1
    for (int k = 1; k < n; ++k) {
        const RecCoef c = rec_coef(k, alpha, beta);      // once per k, not per point
        for (std::size_t i = 0; i < np; ++i) {
            const double pkp1 = ((c.a2 + c.a3 * x[i]) * p[i] - c.a4 * pm1[i]) / c.a1;
            pm1[i] = p[i];
            p[i]   = pkp1;
        }
    }
}

void jacobi_deriv(int n, double alpha, double beta,
                  const double* x, std::size_t np, double* dp)
{
    check_args(n, alpha, beta);
    if (n == 0) {
        for (std::size_t i = 0; i < np; ++i) dp[i] = 0.0;
        return;
    }
    // (A.1.8): one recurrence of degree n-1 with (a+1, b+1), then a scaling
    jacobi(n - 1, alpha + 1.0, beta + 1.0, x, np, dp);
    const double s = 0.5 * (n + alpha + beta + 1.0);
    for (std::size_t i = 0; i < np; ++i) dp[i] *= s;
}

void jacobi_and_deriv(int n, double alpha, double beta,
                      const double* x, std::size_t np, double* p, double* dp)
{
    check_args(n, alpha, beta);
    if (np == 0) return;

    // Values from (A.1.3) and derivatives from its x-derivative (D.2), in the
    // same loop. Four recurrence arrays are needed: P_k, P_{k-1}, P'_k, P'_{k-1}.
    //
    // Points are processed in blocks of B = 256 (4 x 256 doubles = 8 KiB, which
    // fits in L1 cache), so the four arrays stay in cache for all n steps.
    // Without blocking they are streamed through main memory at every k, and
    // the routine becomes slower than two separate calls (measured: x0.4).
    constexpr std::size_t B = 256;
    double pk[B], pkm1[B], dk[B], dkm1[B];             // P_k, P_{k-1}, P'_k, P'_{k-1}
    const double d1 = 0.5 * (alpha + beta + 2.0);

    for (std::size_t i0 = 0; i0 < np; i0 += B) {
        const std::size_t m  = (np - i0 < B) ? np - i0 : B;
        const double*     xb = x + i0;

        if (n == 0) {
            for (std::size_t i = 0; i < m; ++i) { pk[i] = 1.0; dk[i] = 0.0; }
        } else {
            // Starting values: (A.1.3) for P, (D.3) for P'
            for (std::size_t i = 0; i < m; ++i) {
                pkm1[i] = 1.0;                    dkm1[i] = 0.0;   // P_0, P_0'
                pk[i]   = p1(alpha, beta, xb[i]); dk[i]   = d1;    // P_1, P_1'
            }
            for (int k = 1; k < n; ++k) {
                const RecCoef c = rec_coef(k, alpha, beta);
                for (std::size_t i = 0; i < m; ++i) {
                    const double ax   = c.a2 + c.a3 * xb[i];
                    // (A.1.3):  a1 P_{k+1}  = (a2 + a3 x) P_k - a4 P_{k-1}
                    const double pkp1 = (ax * pk[i] - c.a4 * pkm1[i]) / c.a1;
                    // (D.2):    a1 P'_{k+1} = a3 P_k + (a2 + a3 x) P'_k - a4 P'_{k-1}
                    const double dkp1 = (c.a3 * pk[i] + ax * dk[i] - c.a4 * dkm1[i]) / c.a1;
                    pkm1[i] = pk[i];  pk[i] = pkp1;
                    dkm1[i] = dk[i];  dk[i] = dkp1;
                }
            }
        }
        // Copy the block results out; either output may be skipped (nullptr)
        if (p)  for (std::size_t i = 0; i < m; ++i) p[i0 + i]  = pk[i];
        if (dp) for (std::size_t i = 0; i < m; ++i) dp[i0 + i] = dk[i];
    }
}

void jacobi_all(int nmax, double alpha, double beta,
                const double* x, std::size_t np, double* V)
{
    check_args(nmax, alpha, beta);

    // (D.4): V[i*ld + k] = P_k(x_i), row-major, ld = nmax + 1 columns per row
    const std::size_t ld = static_cast<std::size_t>(nmax) + 1;

    for (std::size_t i = 0; i < np; ++i) V[i * ld] = 1.0;                    // P_0
    if (nmax == 0) return;
    for (std::size_t i = 0; i < np; ++i) V[i * ld + 1] = p1(alpha, beta, x[i]);  // P_1

    // (A.1.3): column k+1 from columns k and k-1. Each P_k is computed exactly
    // once, instead of nmax+1 separate recurrences of increasing length.
    for (int k = 1; k < nmax; ++k) {
        const RecCoef c = rec_coef(k, alpha, beta);
        for (std::size_t i = 0; i < np; ++i) {
            double* row = V + i * ld;
            row[k + 1] = ((c.a2 + c.a3 * x[i]) * row[k] - c.a4 * row[k - 1]) / c.a1;
        }
    }
}

// ===========================================================================
// std::vector wrappers
// ===========================================================================
void jacobi(int n, double alpha, double beta,
            const std::vector<double>& x, std::vector<double>& p)
{
    p.resize(x.size());
    jacobi(n, alpha, beta, x.data(), x.size(), p.data());
}

void jacobi_deriv(int n, double alpha, double beta,
                  const std::vector<double>& x, std::vector<double>& dp)
{
    dp.resize(x.size());
    jacobi_deriv(n, alpha, beta, x.data(), x.size(), dp.data());
}

// ===========================================================================
// Norm
// ===========================================================================
double jacobi_norm2(int n, double alpha, double beta)
{
    check_args(n, alpha, beta);
    const double apb = alpha + beta;

    // (D.5): h_0 = 2^{a+b+1} Gamma(a+1) Gamma(b+1) / Gamma(a+b+2)
    // (A.1.7) is 0 * inf at n = 0 when a + b = -1 (Chebyshev); (D.5) is the
    // same value, using (2n+a+b+1) Gamma(n+a+b+1) = Gamma(a+b+2) at n = 0.
    if (n == 0) {
        return std::exp((apb + 1.0) * std::log(2.0)
                        + std::lgamma(alpha + 1.0) + std::lgamma(beta + 1.0)
                        - std::lgamma(apb + 2.0));
    }

    // (A.1.7), from logarithms: Gamma(z) overflows double for z > 171.6
    //   h_n = 2^{a+b+1} / (2n+a+b+1) * Gamma(n+a+1) Gamma(n+b+1) / (n! Gamma(n+a+b+1))
    const double log_h = (apb + 1.0) * std::log(2.0) - std::log(2.0 * n + apb + 1.0)
                       + std::lgamma(n + alpha + 1.0) + std::lgamma(n + beta + 1.0)
                       - std::lgamma(n + 1.0)          // log n!
                       - std::lgamma(n + apb + 1.0);
    return std::exp(log_h);
}

} // namespace dg
