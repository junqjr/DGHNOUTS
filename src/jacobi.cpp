/**
 * \file    jacobi.cpp
 * \brief   Implementation of the Jacobi module (see jacobi.hpp).
 *
 * \details The public functions are documented in jacobi.hpp. This file
 *          documents the internal helpers and, inside each function body,
 *          which equation each step implements. Equation numbers are those of
 *          Karniadakis & Sherwin, Appendix A.1, or (D.n) for equations of the
 *          theory document \theorypdf. The standard-element coordinate is
 *          \f$\xi\f$ (`xi`).
 *
 *          All functions are templates on the floating-point type `T`; they
 *          are explicitly instantiated at the end of this file for `float`,
 *          `double` and `long double`.
 */
#include "jacobi.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace dg {

namespace {

/**
 * \brief  Validate the arguments common to all Jacobi routines.
 * \param[in] n      degree; must be \f$\ge 0\f$
 * \param[in] alpha  must be \f$> -1\f$
 * \param[in] beta   must be \f$> -1\f$
 * \throws std::invalid_argument otherwise. For \f$\alpha\le-1\f$ or
 *         \f$\beta\le-1\f$ the weight \f$(1-\xi)^\alpha(1+\xi)^\beta\f$ is not
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
 * \brief Coefficients \f$a_n^1,\dots,a_n^4\f$ of the recurrence (A.1.3),
 *        in the book's notation.
 *
 * (A.1.3) reads
 * \f$a_n^1 P_{n+1} = (a_n^2 + a_n^3 \xi)\,P_n - a_n^4 P_{n-1}\f$.
 *
 * \tparam T floating-point type of the coefficients
 */
template <typename T>
struct RecCoef {
    T a1;  ///< \f$a_n^1 = 2(n+1)(n+\alpha+\beta+1)(2n+\alpha+\beta)\f$
    T a2;  ///< \f$a_n^2 = (2n+\alpha+\beta+1)(\alpha^2-\beta^2)\f$
    T a3;  ///< \f$a_n^3 = (2n+\alpha+\beta)(2n+\alpha+\beta+1)(2n+\alpha+\beta+2)\f$
    T a4;  ///< \f$a_n^4 = 2(n+\alpha)(n+\beta)(2n+\alpha+\beta+2)\f$
};

/**
 * \brief  Coefficients \f$a_n^1,\dots,a_n^4\f$ of the recurrence (A.1.3) for
 *         a given \f$n\f$, exactly as in the book.
 *
 * The recurrence with these coefficients produces \f$P_{n+1}\f$ from
 * \f$P_n\f$ and \f$P_{n-1}\f$. For \f$n\ge1\f$ and \f$\alpha,\beta>-1\f$,
 * \f$a_n^1 > 0\f$, so the recurrence never divides by zero.
 *
 * In the routines below, `n` is the target degree, so the recurrence step
 * is called with the loop index: `rec_coef(k, ...)`, \f$k = 1,\dots,n-1\f$.
 *
 * \tparam    T      floating-point type in which the coefficients are computed
 * \param[in] n      index of (A.1.3), \f$n \ge 1\f$
 * \param[in] alpha  \f$\alpha\f$, already converted to `T`
 * \param[in] beta   \f$\beta\f$, already converted to `T`
 * \return    the four coefficients
 *
 * \par Theory
 * \theorypdf, Section 2.2, Eq. (A.1.3).
 */
template <typename T>
inline RecCoef<T> rec_coef(int n, T alpha, T beta)
{
    const T apb = alpha + beta;
    const T c   = T(2) * n + apb;                                // 2n + a + b
    return { T(2) * (n + 1) * (n + apb + T(1)) * c,              // a_n^1
             (c + T(1)) * (alpha * alpha - beta * beta),          // a_n^2
             c * (c + T(1)) * (c + T(2)),                         // a_n^3
             T(2) * (n + alpha) * (n + beta) * (c + T(2)) };      // a_n^4
}

/**
 * \brief  First-degree polynomial
 *         \f$P_1^{\alpha,\beta}(\xi) = \tfrac12\bigl[\alpha-\beta+(\alpha+\beta+2)\xi\bigr]\f$,
 *         the second starting value of (A.1.3).
 *
 * Its derivative, \f$\tfrac12(\alpha+\beta+2)\f$, is the starting value (D.3)
 * of the derivative recurrence.
 *
 * \tparam    T      floating-point type
 * \param[in] alpha  \f$\alpha\f$
 * \param[in] beta   \f$\beta\f$
 * \param[in] xi     evaluation point \f$\xi\f$
 * \return    \f$P_1^{\alpha,\beta}(\xi)\f$
 */
template <typename T>
inline T p1(T alpha, T beta, T xi)
{
    return T(0.5) * (alpha - beta + (alpha + beta + T(2)) * xi);
}

} // namespace

// ===========================================================================
// Single point
// ===========================================================================
template <typename T>
T jacobi(int n, double alpha, double beta, T xi)
{
    check_args(n, alpha, beta);
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);

    // Starting values of (A.1.3)
    T p_km1 = T(1);                                      // P_0
    if (n == 0) return p_km1;
    T p_k = p1(a, b, xi);                                // P_1

    // (A.1.3) with the book's n = k: P_{k+1} from P_k and P_{k-1}, k = 1 .. n-1
    for (int k = 1; k < n; ++k) {
        const RecCoef<T> c = rec_coef(k, a, b);
        const T p_kp1 = ((c.a2 + c.a3 * xi) * p_k - c.a4 * p_km1) / c.a1;
        p_km1 = p_k;
        p_k   = p_kp1;
    }
    return p_k;                                          // P_n
}

template <typename T>
T jacobi_deriv(int n, double alpha, double beta, T xi)
{
    check_args(n, alpha, beta);
    if (n == 0) return T(0);                             // P_0 = 1 is constant

    // (A.1.8): d/dxi P_n^{a,b} = 1/2 (n+a+b+1) P_{n-1}^{a+1,b+1}
    // Valid at xi = +/-1, unlike (A.1.4), which divides by 1 - xi^2.
    const T s = T(0.5) * (T(n) + static_cast<T>(alpha) + static_cast<T>(beta) + T(1));
    return s * jacobi(n - 1, alpha + 1.0, beta + 1.0, xi);
}

template <typename T>
void jacobi_and_deriv(int n, double alpha, double beta, T xi, T& p, T& dp)
{
    check_args(n, alpha, beta);
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);

    if (n == 0) { p = T(1); dp = T(0); return; }

    // Starting values: (A.1.3) for P, (D.3) for P'
    T pkm1 = T(1), dkm1 = T(0);                          // P_0, P_0'
    T pk = p1(a, b, xi), dk = T(0.5) * (a + b + T(2));   // P_1, P_1'

    // Book's n = k, k = 1 .. n-1. Same operations, in the same order, as the
    // array version of jacobi_and_deriv, so the results are bitwise identical.
    for (int k = 1; k < n; ++k) {
        const RecCoef<T> c = rec_coef(k, a, b);
        const T axi  = c.a2 + c.a3 * xi;
        // (A.1.3):  a1 P_{k+1}  = (a2 + a3 xi) P_k - a4 P_{k-1}
        const T pkp1 = (axi * pk - c.a4 * pkm1) / c.a1;
        // (D.2):    a1 P'_{k+1} = a3 P_k + (a2 + a3 xi) P'_k - a4 P'_{k-1}
        const T dkp1 = (c.a3 * pk + axi * dk - c.a4 * dkm1) / c.a1;
        pkm1 = pk;  pk = pkp1;
        dkm1 = dk;  dk = dkp1;
    }
    p  = pk;
    dp = dk;
}

// ===========================================================================
// Arrays of points
//
// Loop order: degree k outside (sequential: P_{k+1} needs P_k and P_{k-1}),
// points i inside (independent: the compiler vectorises this loop with SIMD).
// The recurrence coefficients are computed once per k, not once per point.
// ===========================================================================
template <typename T>
void jacobi(int n, double alpha, double beta,
            const T* xi, std::size_t np, T* p)
{
    check_args(n, alpha, beta);
    if (np == 0) return;
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);

    if (n == 0) {
        for (std::size_t i = 0; i < np; ++i) p[i] = T(1);
        return;
    }

    // Storage: the output p holds the current P_k, the work array pm1 holds
    // P_{k-1}. Both are updated in place at every step.
    std::vector<T> pm1(np, T(1));                        // P_{k-1}, starts as P_0
    for (std::size_t i = 0; i < np; ++i) p[i] = p1(a, b, xi[i]);   // P_k = P_1

    // (A.1.3) with the book's n = k, k = 1 .. n-1
    for (int k = 1; k < n; ++k) {
        const RecCoef<T> c = rec_coef(k, a, b);          // once per k, not per point
        for (std::size_t i = 0; i < np; ++i) {
            const T pkp1 = ((c.a2 + c.a3 * xi[i]) * p[i] - c.a4 * pm1[i]) / c.a1;
            pm1[i] = p[i];
            p[i]   = pkp1;
        }
    }
}

template <typename T>
void jacobi_deriv(int n, double alpha, double beta,
                  const T* xi, std::size_t np, T* dp)
{
    check_args(n, alpha, beta);
    if (n == 0) {
        for (std::size_t i = 0; i < np; ++i) dp[i] = T(0);
        return;
    }
    // (A.1.8): one recurrence of degree n-1 with (a+1, b+1), then a scaling
    jacobi(n - 1, alpha + 1.0, beta + 1.0, xi, np, dp);
    const T s = T(0.5) * (T(n) + static_cast<T>(alpha) + static_cast<T>(beta) + T(1));
    for (std::size_t i = 0; i < np; ++i) dp[i] *= s;
}

template <typename T>
void jacobi_and_deriv(int n, double alpha, double beta,
                      const T* xi, std::size_t np,
                      detail::nondeduced_t<T>* p, detail::nondeduced_t<T>* dp)
{
    check_args(n, alpha, beta);
    if (np == 0) return;
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);

    // Values from (A.1.3) and derivatives from its xi-derivative (D.2), in the
    // same loop. Four recurrence arrays are needed: P_k, P_{k-1}, P'_k, P'_{k-1}.
    //
    // Points are processed in blocks of B = 256 (4 x 256 doubles = 8 KiB, which
    // fits in L1 cache), so the four arrays stay in cache for all n steps.
    // Without blocking they are streamed through main memory at every k, and
    // the routine becomes slower than two separate calls (measured: x0.4).
    constexpr std::size_t B = 256;
    T pk[B], pkm1[B], dk[B], dkm1[B];                    // P_k, P_{k-1}, P'_k, P'_{k-1}
    const T d1 = T(0.5) * (a + b + T(2));                // (D.3): P_1'

    for (std::size_t i0 = 0; i0 < np; i0 += B) {
        const std::size_t m   = (np - i0 < B) ? np - i0 : B;
        const T*          xib = xi + i0;

        if (n == 0) {
            for (std::size_t i = 0; i < m; ++i) { pk[i] = T(1); dk[i] = T(0); }
        } else {
            // Starting values: (A.1.3) for P, (D.3) for P'
            for (std::size_t i = 0; i < m; ++i) {
                pkm1[i] = T(1);              dkm1[i] = T(0);     // P_0, P_0'
                pk[i]   = p1(a, b, xib[i]);  dk[i]   = d1;       // P_1, P_1'
            }
            // Book's n = k, k = 1 .. n-1
            for (int k = 1; k < n; ++k) {
                const RecCoef<T> c = rec_coef(k, a, b);
                for (std::size_t i = 0; i < m; ++i) {
                    const T axi  = c.a2 + c.a3 * xib[i];
                    // (A.1.3):  a1 P_{k+1}  = (a2 + a3 xi) P_k - a4 P_{k-1}
                    const T pkp1 = (axi * pk[i] - c.a4 * pkm1[i]) / c.a1;
                    // (D.2):    a1 P'_{k+1} = a3 P_k + (a2 + a3 xi) P'_k - a4 P'_{k-1}
                    const T dkp1 = (c.a3 * pk[i] + axi * dk[i] - c.a4 * dkm1[i]) / c.a1;
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

template <typename T>
void jacobi_all(int nmax, double alpha, double beta,
                const T* xi, std::size_t np, T* V)
{
    check_args(nmax, alpha, beta);
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);

    // (D.4): V[i*ld + k] = P_k(xi_i), row-major, ld = nmax + 1 columns per row
    const std::size_t ld = static_cast<std::size_t>(nmax) + 1;

    for (std::size_t i = 0; i < np; ++i) V[i * ld] = T(1);                       // P_0
    if (nmax == 0) return;
    for (std::size_t i = 0; i < np; ++i) V[i * ld + 1] = p1(a, b, xi[i]);       // P_1

    // (A.1.3) with the book's n = k: column k+1 from columns k and k-1. Each
    // P_k is computed exactly once, instead of nmax+1 separate recurrences.
    for (int k = 1; k < nmax; ++k) {
        const RecCoef<T> c = rec_coef(k, a, b);
        for (std::size_t i = 0; i < np; ++i) {
            T* row = V + i * ld;
            row[k + 1] = ((c.a2 + c.a3 * xi[i]) * row[k] - c.a4 * row[k - 1]) / c.a1;
        }
    }
}

// ===========================================================================
// std::vector wrappers
// ===========================================================================
template <typename T>
void jacobi(int n, double alpha, double beta,
            const std::vector<T>& xi, std::vector<T>& p)
{
    p.resize(xi.size());
    jacobi(n, alpha, beta, xi.data(), xi.size(), p.data());
}

template <typename T>
void jacobi_deriv(int n, double alpha, double beta,
                  const std::vector<T>& xi, std::vector<T>& dp)
{
    dp.resize(xi.size());
    jacobi_deriv(n, alpha, beta, xi.data(), xi.size(), dp.data());
}

// ===========================================================================
// Norm
// ===========================================================================
template <typename T>
T jacobi_norm2(int n, double alpha, double beta)
{
    check_args(n, alpha, beta);
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);
    const T apb = a + b;
    const T ln2 = std::log(T(2));

    // (D.5): h_0 = 2^{a+b+1} Gamma(a+1) Gamma(b+1) / Gamma(a+b+2)
    // (A.1.7) is 0 * inf at n = 0 when a + b = -1 (Chebyshev); (D.5) is the
    // same value, using (2n+a+b+1) Gamma(n+a+b+1) = Gamma(a+b+2) at n = 0.
    if (n == 0) {
        return std::exp((apb + T(1)) * ln2
                        + std::lgamma(a + T(1)) + std::lgamma(b + T(1))
                        - std::lgamma(apb + T(2)));
    }

    // (A.1.7), from logarithms: Gamma(z) overflows double for z > 171.6
    //   h_n = 2^{a+b+1} / (2n+a+b+1) * Gamma(n+a+1) Gamma(n+b+1) / (n! Gamma(n+a+b+1))
    const T nn = static_cast<T>(n);
    const T log_h = (apb + T(1)) * ln2 - std::log(T(2) * nn + apb + T(1))
                  + std::lgamma(nn + a + T(1)) + std::lgamma(nn + b + T(1))
                  - std::lgamma(nn + T(1))                  // log n!
                  - std::lgamma(nn + apb + T(1));
    return std::exp(log_h);
}

// ===========================================================================
// Zeros: Newton-Raphson with polynomial deflation (Appendix B.2)
// ===========================================================================
template <typename T>
void jacobi_zeros(int m, double alpha, double beta, T* z)
{
    check_args(m, alpha, beta);
    if (m == 0) return;

    // alpha > beta: the zeros crowd towards xi = -1, where the search below
    // starts, and Newton can fail for large alpha - beta. By (A.1.6),
    // P_m^{a,b}(-xi) = (-1)^m P_m^{b,a}(xi), so the zeros of P_m^{a,b} are the
    // negated zeros of P_m^{b,a}, which are found in the robust direction.
    if (alpha > beta) {
        jacobi_zeros(m, beta, alpha, z);                 // (D.10)
        std::reverse(z, z + m);
        for (int i = 0; i < m; ++i) z[i] = -z[i];
        return;
    }

    const T   tol   = T(4) * std::numeric_limits<T>::epsilon();  // (D.9): |xi| < 1, so relative
    const int maxit = 1000;   // <= 11 iterations for DG parameters; up to ~170 for alpha = beta = 20
    const T   dth   = std::acos(T(-1)) / (T(2) * T(m));          // pi / (2m)

    for (int k = 0; k < m; ++k) {
        // (D.8): Chebyshev zero  -cos((2k+1) pi / 2m), averaged with the
        // previous root (book, B.2: "a better approximation for the initial guess")
        T r = -std::cos(T(2 * k + 1) * dth);
        if (k > 0) r = T(0.5) * (r + z[k - 1]);

        bool converged = false;
        for (int it = 0; it < maxit; ++it) {
            T p, dp;
            jacobi_and_deriv(m, alpha, beta, r, p, dp);  // (A.1.3) + (D.2), one pass

            // Deflation: s = sum_{j<k} 1/(r - xi_j) divides out the known roots
            T s = T(0);
            for (int j = 0; j < k; ++j) s += T(1) / (r - z[j]);

            // (D.7): Newton step for f_{m-k} = P_m / prod_{j<k} (xi - xi_j)
            const T delta = -p / (dp - p * s);
            r += delta;

            // (D.9): |delta| <= 4 eps (the book's test, with the absolute value)
            if (std::fabs(delta) <= tol) { converged = true; break; }
        }
        if (!converged || !std::isfinite(r) || std::fabs(r) >= T(1))
            throw std::runtime_error("jacobi_zeros: Newton did not converge for zero "
                                     + std::to_string(k) + " of P_" + std::to_string(m)
                                     + (std::is_same_v<T, float> ? " (float: compute in double"
                                        " and round instead)" : ""));
        z[k] = r;
    }

    // Increasing order, and the zeros must be distinct (they are simple)
    std::sort(z, z + m);
    for (int k = 1; k < m; ++k)
        if (!(z[k] > z[k - 1]))
            throw std::runtime_error("jacobi_zeros: two zeros of P_" + std::to_string(m)
                                     + " coincide");

    // alpha = beta: P_m has the parity of m (A.1.6), so make the zeros exactly
    // symmetric. Odd functions are then integrated to exactly zero.
    if (alpha == beta) {
        for (int i = 0; i < m / 2; ++i) {
            const T v = T(0.5) * (z[i] - z[m - 1 - i]);
            z[i]         = v;
            z[m - 1 - i] = -v;
        }
        if (m % 2 == 1) z[m / 2] = T(0);
    }

    // (D.11) Certification: P_m must change sign between consecutive points of
    //   -1 < (z_0+z_1)/2 < ... < (z_{m-2}+z_{m-1})/2 < 1.
    // These m+1 points delimit m intervals, each containing one computed zero;
    // m sign changes prove that each interval holds exactly one true zero, so
    // no zero was missed or found twice. Any failure (including overflow to
    // inf/nan in float) is reported instead of returning wrong zeros.
    T prev = jacobi(m, alpha, beta, T(-1));
    for (int i = 0; i <= m - 1; ++i) {
        const T xi_c = (i < m - 1) ? T(0.5) * (z[i] + z[i + 1]) : T(1);
        const T v    = jacobi(m, alpha, beta, xi_c);
        if (!(std::isfinite(v) && std::isfinite(prev) && v != T(0) && prev != T(0)
              && (v > T(0)) != (prev > T(0))))
            throw std::runtime_error("jacobi_zeros: certification failed for P_"
                                     + std::to_string(m) + " (zeros not separated by sign"
                                     " changes)" + (std::is_same_v<T, float> ?
                                     "; float: compute in double and round instead" : ""));
        prev = v;
    }
}

template <typename T>
std::vector<T> jacobi_zeros(int m, double alpha, double beta)
{
    check_args(m, alpha, beta);                          // before allocating
    std::vector<T> z(static_cast<std::size_t>(m));
    jacobi_zeros(m, alpha, beta, z.data());
    return z;
}

// ===========================================================================
// Explicit instantiations: float, double, long double
// ===========================================================================
/// \cond INTERNAL
#define DG_INSTANTIATE_JACOBI(T)                                                       \
    template T    jacobi<T>(int, double, double, T);                                   \
    template T    jacobi_deriv<T>(int, double, double, T);                             \
    template void jacobi<T>(int, double, double, const T*, std::size_t, T*);           \
    template void jacobi_deriv<T>(int, double, double, const T*, std::size_t, T*);     \
    template void jacobi_and_deriv<T>(int, double, double, const T*, std::size_t,      \
                                      detail::nondeduced_t<T>*,                        \
                                      detail::nondeduced_t<T>*);                       \
    template void jacobi_all<T>(int, double, double, const T*, std::size_t, T*);       \
    template void jacobi<T>(int, double, double, const std::vector<T>&,                \
                            std::vector<T>&);                                          \
    template void jacobi_deriv<T>(int, double, double, const std::vector<T>&,          \
                                  std::vector<T>&);                                    \
    template T    jacobi_norm2<T>(int, double, double);                                \
    template void jacobi_and_deriv<T>(int, double, double, T, T&, T&);                 \
    template void jacobi_zeros<T>(int, double, double, T*);                            \
    template std::vector<T> jacobi_zeros<T>(int, double, double);

DG_INSTANTIATE_JACOBI(float)
DG_INSTANTIATE_JACOBI(double)
DG_INSTANTIATE_JACOBI(long double)

#undef DG_INSTANTIATE_JACOBI
/// \endcond

} // namespace dg
