/**
 * \file    jacobi_reference.cpp
 * \brief   Reference implementation of the Jacobi module: the book's equations,
 *          written as directly as possible.
 *
 * \details **Start here.** Every routine follows Karniadakis & Sherwin,
 *          Appendices A.1 and B.2, with the same notation (the standard
 *          coordinate is \f$\xi\f$, `xi`), one point at a time, without any
 *          performance technique. Read it next to the theory document
 *          \theorypdf: each step is commented with its equation number.
 *
 *          The optimised implementation (jacobi_optimized.cpp) computes the
 *          same quantities, faster and in a less obvious way; the test
 *          test_impl.cpp checks that both agree. This one is selected with
 *          the CMake option `DGHNOUTS_KERNELS=reference`.
 */
#include "jacobi_impl.hpp"

namespace dg {
namespace reference {

using internal::check_args;
using internal::p1;
using internal::rec_coef;
using internal::RecCoef;

// ===========================================================================
// One point
// ===========================================================================

template <typename T>
T jacobi(int n, double alpha, double beta, T xi)
{
    check_args(n, alpha, beta);
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);

    // Starting values of (A.1.3): P_0 = 1, P_1 = (a - b + (a + b + 2) xi) / 2
    if (n == 0) return T(1);
    T p_km1 = T(1);                      // P_{k-1}
    T p_k   = p1(a, b, xi);              // P_k

    // (A.1.3), with the book's n written k (n is already the target degree):
    //   a_k^1 P_{k+1} = (a_k^2 + a_k^3 xi) P_k - a_k^4 P_{k-1},   k = 1, ..., n-1
    for (int k = 1; k < n; ++k) {
        const RecCoef<T> c = rec_coef(k, a, b);
        const T p_kp1 = ((c.a2 + c.a3 * xi) * p_k - c.a4 * p_km1) / c.a1;
        p_km1 = p_k;                     // shift: P_{k-1} <- P_k, P_k <- P_{k+1}
        p_k   = p_kp1;
    }
    return p_k;                          // P_n
}

template <typename T>
T jacobi_deriv(int n, double alpha, double beta, T xi)
{
    check_args(n, alpha, beta);
    if (n == 0) return T(0);             // P_0 = 1 is constant

    // (A.1.8): d/dxi P_n^{a,b} = 1/2 (n + a + b + 1) P_{n-1}^{a+1,b+1}
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
    T p_km1 = T(1),          dp_km1 = T(0);                       // P_0, P_0'
    T p_k   = p1(a, b, xi),  dp_k   = T(0.5) * (a + b + T(2));    // P_1, P_1'

    for (int k = 1; k < n; ++k) {
        const RecCoef<T> c = rec_coef(k, a, b);
        // (A.1.3): a1 P_{k+1}  = (a2 + a3 xi) P_k - a4 P_{k-1}
        const T p_kp1  = ((c.a2 + c.a3 * xi) * p_k - c.a4 * p_km1) / c.a1;
        // (D.2), the xi-derivative of (A.1.3):
        //          a1 P'_{k+1} = a3 P_k + (a2 + a3 xi) P'_k - a4 P'_{k-1}
        const T dp_kp1 = (c.a3 * p_k + (c.a2 + c.a3 * xi) * dp_k - c.a4 * dp_km1) / c.a1;
        p_km1  = p_k;   p_k  = p_kp1;
        dp_km1 = dp_k;  dp_k = dp_kp1;
    }
    p  = p_k;
    dp = dp_k;
}

// ===========================================================================
// Many points: simply the one-point routines in a loop over the points
// ===========================================================================

template <typename T>
void jacobi(int n, double alpha, double beta, const T* xi, std::size_t np, T* p)
{
    check_args(n, alpha, beta);
    for (std::size_t i = 0; i < np; ++i) p[i] = jacobi(n, alpha, beta, xi[i]);
}

template <typename T>
void jacobi_deriv(int n, double alpha, double beta, const T* xi, std::size_t np, T* dp)
{
    check_args(n, alpha, beta);
    for (std::size_t i = 0; i < np; ++i) dp[i] = jacobi_deriv(n, alpha, beta, xi[i]);
}

template <typename T>
void jacobi_and_deriv(int n, double alpha, double beta, const T* xi, std::size_t np,
                      detail::nondeduced_t<T>* p, detail::nondeduced_t<T>* dp)
{
    check_args(n, alpha, beta);
    for (std::size_t i = 0; i < np; ++i) {
        T pi, dpi;
        jacobi_and_deriv(n, alpha, beta, xi[i], pi, dpi);
        if (p)  p[i]  = pi;              // either output may be nullptr
        if (dp) dp[i] = dpi;
    }
}

template <typename T>
void jacobi_all(int nmax, double alpha, double beta, const T* xi, std::size_t np, T* V)
{
    check_args(nmax, alpha, beta);
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);
    const std::size_t ld = static_cast<std::size_t>(nmax) + 1;   // values per row

    // (D.4): row i of V holds P_0(xi_i), P_1(xi_i), ..., P_nmax(xi_i).
    // One recurrence (A.1.3) per point; every degree is kept along the way.
    for (std::size_t i = 0; i < np; ++i) {
        T* row = V + i * ld;
        row[0] = T(1);                                         // P_0
        if (nmax == 0) continue;
        row[1] = p1(a, b, xi[i]);                              // P_1
        for (int k = 1; k < nmax; ++k) {
            const RecCoef<T> c = rec_coef(k, a, b);
            row[k + 1] = ((c.a2 + c.a3 * xi[i]) * row[k] - c.a4 * row[k - 1]) / c.a1;
        }
    }
}

// ===========================================================================
// Zeros: Newton-Raphson with polynomial deflation (Appendix B.2)
// ===========================================================================

template <typename T>
void jacobi_zeros(int m, double alpha, double beta, T* z)
{
    check_args(m, alpha, beta);
    if (m == 0) return;

    // (D.10) Reflection: for alpha > beta the zeros crowd towards xi = -1,
    // where the search starts, and Newton can fail. By (A.1.6) the zeros of
    // P_m^{a,b} are the negated zeros of P_m^{b,a}: compute those instead.
    if (alpha > beta) {
        jacobi_zeros(m, beta, alpha, z);
        std::reverse(z, z + m);
        for (int i = 0; i < m; ++i) z[i] = -z[i];
        return;
    }

    const T   tol   = T(4) * std::numeric_limits<T>::epsilon();   // (D.9)
    const int maxit = 1000;
    const T   pi    = std::acos(T(-1));

    for (int k = 0; k < m; ++k) {
        // (D.8) Initial guess: the k-th Chebyshev zero, averaged with the
        // previous zero found (book, B.2)
        T r = -std::cos(T(2 * k + 1) * pi / (T(2) * T(m)));
        if (k > 0) r = T(0.5) * (r + z[k - 1]);

        bool converged = false;
        for (int it = 0; it < maxit; ++it) {
            T p, dp;
            jacobi_and_deriv(m, alpha, beta, r, p, dp);   // P_m(r), P_m'(r)

            // Deflation: the zeros z_0 .. z_{k-1} already found are divided out
            T s = T(0);
            for (int j = 0; j < k; ++j) s += T(1) / (r - z[j]);

            // (D.7) Newton step for f = P_m / prod_{j<k} (xi - z_j)
            const T delta = -p / (dp - p * s);
            r += delta;

            if (std::fabs(delta) <= tol) { converged = true; break; }   // (D.9)
        }
        if (!converged || !std::isfinite(r) || std::fabs(r) >= T(1))
            internal::throw_newton_failure<T>(k, m);
        z[k] = r;
    }

    internal::sort_zeros(m, z);                       // increasing, distinct
    if (alpha == beta) internal::symmetrize_zeros(m, z);

    // (D.11) Certification: P_m changes sign between consecutive cut points
    const std::vector<T> c = internal::cut_points(m, z);
    std::vector<T> pc(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) pc[i] = jacobi(m, alpha, beta, c[i]);
    internal::certify_zeros(m, pc.data());
}

// ===========================================================================
// Explicit instantiations: float, double, long double
// ===========================================================================
} // namespace reference

/// \cond INTERNAL
DG_INSTANTIATE_JACOBI_IMPL(reference, float)
DG_INSTANTIATE_JACOBI_IMPL(reference, double)
DG_INSTANTIATE_JACOBI_IMPL(reference, long double)
/// \endcond

} // namespace dg
