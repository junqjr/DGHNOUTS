// jacobi.cpp - see jacobi.hpp for the equations used
#include "jacobi.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace dg {

namespace {

void check_args(int n, double alpha, double beta)
{
    if (n < 0)
        throw std::invalid_argument("jacobi: n must be >= 0, got " + std::to_string(n));
    if (alpha <= -1.0 || beta <= -1.0)
        throw std::invalid_argument("jacobi: alpha and beta must be > -1");
}

// Coefficients of the recurrence (A.1.3), with k = the book's n:
//
//   a1_k P_{k+1} = (a2_k + a3_k x) P_k - a4_k P_{k-1},   k >= 1
//
//   a1_k = 2 (k+1) (k+a+b+1) (2k+a+b)
//   a2_k = (2k+a+b+1) (a^2 - b^2)
//   a3_k = (2k+a+b) (2k+a+b+1) (2k+a+b+2)
//   a4_k = 2 (k+a) (k+b) (2k+a+b+2)
struct RecCoef { double a1, a2, a3, a4; };

inline RecCoef rec_coef(int k, double alpha, double beta)
{
    const double apb = alpha + beta;
    const double c   = 2.0 * k + apb;                    // 2k + a + b
    return { 2.0 * (k + 1) * (k + apb + 1.0) * c,
             (c + 1.0) * (alpha * alpha - beta * beta),
             c * (c + 1.0) * (c + 2.0),
             2.0 * (k + alpha) * (k + beta) * (c + 2.0) };
}

// P_1 = 1/2 [ a - b + (a + b + 2) x ],   P_1' = 1/2 (a + b + 2)
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
    double p_km1 = 1.0;                                  // P_0
    if (n == 0) return p_km1;
    double p_k = p1(alpha, beta, x);                     // P_1

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
    if (n == 0) return 0.0;
    // (A.1.8)
    return 0.5 * (n + alpha + beta + 1.0) * jacobi(n - 1, alpha + 1.0, beta + 1.0, x);
}

// ===========================================================================
// Arrays of points
//
// Storage: the output array p holds the current P_k; one work array holds
// P_{k-1}. Each step of k updates all np points: the inner loop is
// independent across i and vectorises.
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

    std::vector<double> pm1(np, 1.0);                    // P_{k-1}, starts as P_0
    for (std::size_t i = 0; i < np; ++i) p[i] = p1(alpha, beta, x[i]);   // P_k = P_1

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

    // Points are processed in blocks that fit in L1 cache, so the four
    // recurrence arrays stay in cache for all n steps (otherwise this routine
    // is memory-bound for large np and slower than two separate calls).
    constexpr std::size_t B = 256;
    double pk[B], pkm1[B], dk[B], dkm1[B];             // P_k, P_{k-1}, P'_k, P'_{k-1}
    const double d1 = 0.5 * (alpha + beta + 2.0);

    for (std::size_t i0 = 0; i0 < np; i0 += B) {
        const std::size_t m  = (np - i0 < B) ? np - i0 : B;
        const double*     xb = x + i0;

        if (n == 0) {
            for (std::size_t i = 0; i < m; ++i) { pk[i] = 1.0; dk[i] = 0.0; }
        } else {
            for (std::size_t i = 0; i < m; ++i) {
                pkm1[i] = 1.0;                    dkm1[i] = 0.0;   // P_0, P_0'
                pk[i]   = p1(alpha, beta, xb[i]); dk[i]   = d1;    // P_1, P_1'
            }
            for (int k = 1; k < n; ++k) {
                const RecCoef c = rec_coef(k, alpha, beta);
                for (std::size_t i = 0; i < m; ++i) {
                    const double ax   = c.a2 + c.a3 * xb[i];
                    // (A.1.3) and its x-derivative
                    const double pkp1 = (ax * pk[i] - c.a4 * pkm1[i]) / c.a1;
                    const double dkp1 = (c.a3 * pk[i] + ax * dk[i] - c.a4 * dkm1[i]) / c.a1;
                    pkm1[i] = pk[i];  pk[i] = pkp1;
                    dkm1[i] = dk[i];  dk[i] = dkp1;
                }
            }
        }
        if (p)  for (std::size_t i = 0; i < m; ++i) p[i0 + i]  = pk[i];
        if (dp) for (std::size_t i = 0; i < m; ++i) dp[i0 + i] = dk[i];
    }
}

void jacobi_all(int nmax, double alpha, double beta,
                const double* x, std::size_t np, double* V)
{
    check_args(nmax, alpha, beta);
    const std::size_t ld = static_cast<std::size_t>(nmax) + 1;   // row length

    for (std::size_t i = 0; i < np; ++i) V[i * ld] = 1.0;                    // P_0
    if (nmax == 0) return;
    for (std::size_t i = 0; i < np; ++i) V[i * ld + 1] = p1(alpha, beta, x[i]);  // P_1

    // column k+1 from columns k and k-1: each P_k computed exactly once
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
// Norm: (A.1.7)
//
//   h_n = 2^{a+b+1} / (2n+a+b+1) * Gamma(n+a+1) Gamma(n+b+1) / ( n! Gamma(n+a+b+1) )
//
// Computed with std::lgamma to avoid overflow at large n.
// For n = 0 the formula is 0/0 when a+b = -1 (e.g. Chebyshev), so we use the
// equivalent form  h_0 = 2^{a+b+1} Gamma(a+1) Gamma(b+1) / Gamma(a+b+2),
// which follows from (2n+a+b+1) Gamma(n+a+b+1) = Gamma(a+b+2) at n = 0.
// ===========================================================================
double jacobi_norm2(int n, double alpha, double beta)
{
    check_args(n, alpha, beta);
    const double apb = alpha + beta;

    if (n == 0) {
        return std::exp((apb + 1.0) * std::log(2.0)
                        + std::lgamma(alpha + 1.0) + std::lgamma(beta + 1.0)
                        - std::lgamma(apb + 2.0));
    }

    const double log_h = (apb + 1.0) * std::log(2.0) - std::log(2.0 * n + apb + 1.0)
                       + std::lgamma(n + alpha + 1.0) + std::lgamma(n + beta + 1.0)
                       - std::lgamma(n + 1.0)          // log n!
                       - std::lgamma(n + apb + 1.0);
    return std::exp(log_h);
}

} // namespace dg
