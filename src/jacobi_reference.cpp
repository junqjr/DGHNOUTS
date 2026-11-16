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
using internal::plus_one;
using internal::rec_coef;
using internal::RecCoef;

// ===========================================================================
// One point
// ===========================================================================

namespace {

/**
 * \brief  \f$P_n^{a,b}(\xi)\f$ by the recurrence (A.1.3), with \f$\alpha,\beta\f$ already
 *         converted to `T` and checked. Used by jacobi() and jacobi_deriv().
 *
 * \par Equations
 * (A.1.3), three-term recursion relation (Karniadakis & Sherwin, Appendix
 * A.1). Starting values, the second one from internal::p1():
 * \f[
 *   P_0^{\alpha,\beta}(\xi) = 1, \qquad
 *   P_1^{\alpha,\beta}(\xi) = \tfrac12\bigl[\alpha-\beta+(\alpha+\beta+2)\,\xi\bigr]
 * \f]
 * then, for each degree,
 * \f[
 *   a_n^1\,P_{n+1}^{\alpha,\beta}(\xi)
 *   = \bigl(a_n^2 + a_n^3\,\xi\bigr)\,P_n^{\alpha,\beta}(\xi) - a_n^4\,P_{n-1}^{\alpha,\beta}(\xi)
 * \f]
 * with the coefficients computed by internal::rec_coef():
 * \f[
 * \begin{aligned}
 *   a_n^1 &= 2(n+1)(n+\alpha+\beta+1)(2n+\alpha+\beta), &
 *   a_n^2 &= (2n+\alpha+\beta+1)(\alpha^2-\beta^2),\\
 *   a_n^3 &= (2n+\alpha+\beta)(2n+\alpha+\beta+1)(2n+\alpha+\beta+2), &
 *   a_n^4 &= 2(n+\alpha)(n+\beta)(2n+\alpha+\beta+2)
 * \end{aligned}
 * \f]
 *
 * \par From the equation to the code
 * `n` is the degree asked for, so the book's index \f$n\f$ of (A.1.3) is the
 * loop index `k` \f$=1,\dots,n-1\f$. Each step solves (A.1.3) for
 * \f$P_{k+1}\f$ and shifts the two values kept:
 * | Code     | Symbol |
 * |----------|--------|
 * | `p_km1`  | \f$P_{k-1}^{\alpha,\beta}(\xi)\f$ |
 * | `p_k`    | \f$P_{k}^{\alpha,\beta}(\xi)\f$ |
 * | `p_kp1`  | \f$P_{k+1}^{\alpha,\beta}(\xi)\f$ |
 * | `c.a1` … `c.a4` | \f$a_k^1,\dots,a_k^4\f$ |
 *
 * \tparam    T   floating-point type
 * \param[in] n   degree, \f$n \ge 0\f$
 * \param[in] a   \f$\alpha\f$
 * \param[in] b   \f$\beta\f$
 * \param[in] xi  evaluation point \f$\xi\f$
 * \return    \f$P_n^{a,b}(\xi)\f$
 */
template <typename T>
T recurrence(int n, T a, T b, T xi)
{
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

} // namespace

/**
 * \par Equations in the reference implementation
 *
 * The recursion relation (A.1.3), evaluated by recurrence():
 * \f[
 *   P_0^{\alpha,\beta}(\xi) = 1, \qquad
 *   P_1^{\alpha,\beta}(\xi) = \tfrac12\bigl[\alpha-\beta+(\alpha+\beta+2)\,\xi\bigr]
 * \f]
 * \f[
 *   a_n^1\,P_{n+1}^{\alpha,\beta}(\xi)
 *   = \bigl(a_n^2 + a_n^3\,\xi\bigr)\,P_n^{\alpha,\beta}(\xi) - a_n^4\,P_{n-1}^{\alpha,\beta}(\xi)
 * \f]
 * with \f$a_n^1,\dots,a_n^4\f$ from internal::rec_coef().
 */
template <typename T>
T jacobi(int n, double alpha, double beta, T xi)
{
    check_args(n, alpha, beta);
    return recurrence(n, static_cast<T>(alpha), static_cast<T>(beta), xi);
}

/**
 * \par Equations in the reference implementation
 *
 * (A.1.8), derivative by degree reduction:
 * \f[
 *   \frac{d}{d\xi}P_n^{\alpha,\beta}(\xi)
 *   = \frac12\,(n+\alpha+\beta+1)\,P_{n-1}^{\alpha+1,\beta+1}(\xi),
 *   \qquad \frac{d}{d\xi}P_0^{\alpha,\beta}(\xi) = 0
 * \f]
 * In the code, `s` \f$= \tfrac12(n+\alpha+\beta+1)\f$ multiplies
 * \f$P_{n-1}^{\alpha+1,\beta+1}(\xi)\f$, which recurrence() evaluates with
 * (A.1.3) for the parameters \f$(\alpha+1,\beta+1)\f$:
 * \f[
 *   a_n^1\,P_{n+1}^{\alpha,\beta}(\xi)
 *   = \bigl(a_n^2 + a_n^3\,\xi\bigr)\,P_n^{\alpha,\beta}(\xi) - a_n^4\,P_{n-1}^{\alpha,\beta}(\xi)
 * \f]
 */
template <typename T>
T jacobi_deriv(int n, double alpha, double beta, T xi)
{
    check_args(n, alpha, beta);
    if (n == 0) return T(0);             // P_0 = 1 is constant

    // (A.1.8): d/dxi P_n^{a,b} = 1/2 (n + a + b + 1) P_{n-1}^{a+1,b+1}
    // Valid at xi = +/-1, unlike (A.1.4), which divides by 1 - xi^2.
    const T s = T(0.5) * (T(n) + static_cast<T>(alpha) + static_cast<T>(beta) + T(1));
    return s * recurrence(n - 1, plus_one<T>(alpha), plus_one<T>(beta), xi);
}

/**
 * \par Equations in the reference implementation
 *
 * The value from (A.1.3) and the derivative from (D.2), the
 * \f$\xi\f$-derivative of (A.1.3), advanced together in the same loop.
 * Starting values, (A.1.3) and (D.3):
 * \f[
 *   P_0^{\alpha,\beta}(\xi) = 1, \qquad
 *   P_1^{\alpha,\beta}(\xi) = \tfrac12\bigl[\alpha-\beta+(\alpha+\beta+2)\,\xi\bigr]
 * \f]
 * \f[
 *   \bigl[P_0^{\alpha,\beta}\bigr]'(\xi) = 0, \qquad
 *   \bigl[P_1^{\alpha,\beta}\bigr]'(\xi) = \tfrac12\,(\alpha+\beta+2)
 * \f]
 * then, for `k` \f$=1,\dots,n-1\f$ (the book's \f$n\f$), (A.1.3) and (D.2):
 * \f[
 *   a_n^1\,P_{n+1}^{\alpha,\beta}(\xi)
 *   = \bigl(a_n^2 + a_n^3\,\xi\bigr)\,P_n^{\alpha,\beta}(\xi) - a_n^4\,P_{n-1}^{\alpha,\beta}(\xi)
 * \f]
 * \f[
 *   a_n^1\,\bigl[P_{n+1}^{\alpha,\beta}\bigr]'(\xi)
 *   = a_n^3\,P_n^{\alpha,\beta}(\xi)
 *   + \bigl(a_n^2 + a_n^3\,\xi\bigr)\bigl[P_n^{\alpha,\beta}\bigr]'(\xi)
 *   - a_n^4\,\bigl[P_{n-1}^{\alpha,\beta}\bigr]'(\xi)
 * \f]
 * | Code               | Symbol |
 * |--------------------|--------|
 * | `p_km1`, `p_k`, `p_kp1`    | \f$P_{k-1},\ P_k,\ P_{k+1}\f$ |
 * | `dp_km1`, `dp_k`, `dp_kp1` | \f$P_{k-1}',\ P_k',\ P_{k+1}'\f$ |
 * | `c.a1` … `c.a4`    | \f$a_k^1,\dots,a_k^4\f$ from internal::rec_coef() |
 */
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

/**
 * \par Equations in the reference implementation
 *
 * One call of the single-point jacobi() per point, i.e. (A.1.3) at each
 * \f$\xi_i\f$:
 * \f[
 *   p_i = P_n^{\alpha,\beta}(\xi_i), \qquad i = 0,\dots,n_p-1,
 * \f]
 * \f[
 *   P_0^{\alpha,\beta}(\xi) = 1, \qquad
 *   P_1^{\alpha,\beta}(\xi) = \tfrac12\bigl[\alpha-\beta+(\alpha+\beta+2)\,\xi\bigr]
 * \f]
 * \f[
 *   a_n^1\,P_{n+1}^{\alpha,\beta}(\xi)
 *   = \bigl(a_n^2 + a_n^3\,\xi\bigr)\,P_n^{\alpha,\beta}(\xi) - a_n^4\,P_{n-1}^{\alpha,\beta}(\xi)
 * \f]
 */
template <typename T>
void jacobi(int n, double alpha, double beta, const T* xi, std::size_t np, T* p)
{
    check_args(n, alpha, beta);
    for (std::size_t i = 0; i < np; ++i) p[i] = jacobi(n, alpha, beta, xi[i]);
}

/**
 * \par Equations in the reference implementation
 *
 * One call of the single-point jacobi_deriv() per point, i.e. (A.1.8) at each
 * \f$\xi_i\f$:
 * \f[
 *   dp_i = \frac{d}{d\xi}P_n^{\alpha,\beta}(\xi_i)
 *        = \frac12\,(n+\alpha+\beta+1)\,P_{n-1}^{\alpha+1,\beta+1}(\xi_i),
 *   \qquad i = 0,\dots,n_p-1 .
 * \f]
 */
template <typename T>
void jacobi_deriv(int n, double alpha, double beta, const T* xi, std::size_t np, T* dp)
{
    check_args(n, alpha, beta);
    for (std::size_t i = 0; i < np; ++i) dp[i] = jacobi_deriv(n, alpha, beta, xi[i]);
}

/**
 * \par Equations in the reference implementation
 *
 * One call of the single-point jacobi_and_deriv() per point: (A.1.3) for the
 * value and (D.2) for the derivative, started with (D.3), at each \f$\xi_i\f$:
 * \f[
 *   a_n^1\,P_{n+1}^{\alpha,\beta}(\xi)
 *   = \bigl(a_n^2 + a_n^3\,\xi\bigr)\,P_n^{\alpha,\beta}(\xi) - a_n^4\,P_{n-1}^{\alpha,\beta}(\xi)
 * \f]
 * \f[
 *   a_n^1\,\bigl[P_{n+1}^{\alpha,\beta}\bigr]'(\xi)
 *   = a_n^3\,P_n^{\alpha,\beta}(\xi)
 *   + \bigl(a_n^2 + a_n^3\,\xi\bigr)\bigl[P_n^{\alpha,\beta}\bigr]'(\xi)
 *   - a_n^4\,\bigl[P_{n-1}^{\alpha,\beta}\bigr]'(\xi)
 * \f]
 * \f[
 *   \bigl[P_0^{\alpha,\beta}\bigr]'(\xi) = 0, \qquad
 *   \bigl[P_1^{\alpha,\beta}\bigr]'(\xi) = \tfrac12\,(\alpha+\beta+2)
 * \f]
 */
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

/**
 * \par Equations in the reference implementation
 *
 * (D.4), the generalised Vandermonde matrix, stored row by row
 * (`row[k]` \f$= \mathbf{V}_{ik}\f$):
 * \f[
 *   \mathbf{V}_{ik} = P_k^{\alpha,\beta}(\xi_i),
 *   \qquad i = 0,\dots,n_p-1,\quad k = 0,\dots,N
 * \f]
 * For each point \f$\xi_i\f$ the row is filled from left to right by the
 * recursion relation (A.1.3), keeping every degree instead of only the last:
 * \f[
 *   P_0^{\alpha,\beta}(\xi) = 1, \qquad
 *   P_1^{\alpha,\beta}(\xi) = \tfrac12\bigl[\alpha-\beta+(\alpha+\beta+2)\,\xi\bigr]
 * \f]
 * \f[
 *   a_n^1\,P_{n+1}^{\alpha,\beta}(\xi)
 *   = \bigl(a_n^2 + a_n^3\,\xi\bigr)\,P_n^{\alpha,\beta}(\xi) - a_n^4\,P_{n-1}^{\alpha,\beta}(\xi)
 * \f]
 * so `row[k+1]` is computed from `row[k]` and `row[k-1]`.
 */
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

/**
 * \par Equations in the reference implementation
 *
 * The steps of the routine, in the order of the code.
 *
 * **1. Reflection (D.10)**, only if \f$\alpha>\beta\f$. From the symmetry
 * relation (A.1.6),
 * \f[
 *   P_n^{\alpha,\beta}(-\xi) = (-1)^n\,P_n^{\beta,\alpha}(\xi)
 * \f]
 * the zeros of \f$P_m^{\alpha,\beta}\f$ are the negated zeros of
 * \f$P_m^{\beta,\alpha}\f$, which are computed instead:
 * \f[
 *   \xi^{\alpha,\beta}_{i,m} = -\,\xi^{\beta,\alpha}_{m-1-i,m}, \qquad i=0,\dots,m-1
 * \f]
 * **2. Initial guess (D.8)** for zero \f$k\f$ (`r`): the Chebyshev zero,
 * averaged with the previous zero found:
 * \f[
 *   r^{(0)}_k = -\cos\Bigl(\frac{2k+1}{2m}\,\pi\Bigr),
 *   \qquad
 *   r^{(0)}_k \leftarrow \tfrac12\bigl(r^{(0)}_k + \xi_{k-1}\bigr) \quad (k>0)
 * \f]
 * **3. Newton step with deflation (D.7).** Newton's method is applied to
 * \f$P_m\f$ divided by the factors of the zeros already found,
 * \f[
 *   f_{m-k}(\xi) = \frac{P_m^{\alpha,\beta}(\xi)}{\prod_{j=0}^{k-1}(\xi-\xi_j)}
 * \f]
 * which gives the step (`delta`), with `p` \f$=P_m(r)\f$, `dp` \f$=P_m'(r)\f$
 * from jacobi_and_deriv() ((A.1.3) and (D.2)) and `s` the sum:
 * \f[
 *   \delta = -\frac{P_m^{\alpha,\beta}(r)}
 *                  {\bigl[P_m^{\alpha,\beta}\bigr]'(r) - P_m^{\alpha,\beta}(r)\,\displaystyle\sum_{j=0}^{k-1}\frac{1}{r-\xi_j}},
 *   \qquad r \leftarrow r + \delta
 * \f]
 * **4. Stopping test (D.9)** (`tol`):
 * \f[
 *   |\delta| \le 4\,\varepsilon,
 *   \qquad \varepsilon = \text{machine epsilon of } T
 * \f]
 * **5. Exact symmetry**, only if \f$\alpha=\beta\f$
 * (internal::symmetrize_zeros()):
 * \f[
 *   \xi_i \leftarrow \tfrac12\bigl(\xi_i-\xi_{m-1-i}\bigr),\qquad
 *   \xi_{m-1-i} = -\xi_i,\qquad
 *   \xi_{(m-1)/2}=0 \ \text{ for odd } m
 * \f]
 * **6. Certification (D.11).** With the cut points (internal::cut_points(),
 * `c`)
 * \f[
 *   c_0 = -1,\qquad c_i = \tfrac12\,(\xi_{i-1}+\xi_i)\quad (i=1,\dots,m-1),\qquad c_m = 1
 * \f]
 * the values `pc` \f$=P_m(c_i)\f$ must alternate in sign
 * (internal::certify_zeros()):
 * \f[
 *   \operatorname{sign} P_m^{\alpha,\beta}(c_i) \neq \operatorname{sign} P_m^{\alpha,\beta}(c_{i+1}),
 *   \qquad i=0,\dots,m-1
 * \f]
 */
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
