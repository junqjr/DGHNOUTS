/**
 * \file    jacobi.hpp
 * \brief   Jacobi polynomials \f$P_n^{\alpha,\beta}(\xi)\f$ on the standard interval
 *          \f$\xi\in[-1,1]\f$: values, derivatives, Vandermonde matrix and norms.
 *
 * \details Public interface of the Jacobi module of the DGHNOUTS spectral/\f$hp\f$
 *          library. Notation and equation numbers follow Karniadakis & Sherwin,
 *          *Spectral/hp Element Methods for CFD*, 2nd ed., Appendix A.1, with the
 *          standard-element coordinate written \f$\xi\f$ as in the book's Chapter 2
 *          (Appendix A writes \f$x\f$). Equations derived in DGHNOUTS (not in the
 *          book) are numbered (D.n) and are given in the theory document
 *          \theorypdf, Part I.
 *
 *          | Function               | Computes                                     | Equation       | Theory |
 *          |------------------------|----------------------------------------------|----------------|--------|
 *          | dg::jacobi()           | \f$P_n^{\alpha,\beta}(\xi)\f$               | (A.1.3)        | Sec. 2.2 |
 *          | dg::jacobi_deriv()     | \f$\frac{d}{d\xi}P_n^{\alpha,\beta}(\xi)\f$ | (A.1.8)        | Sec. 2.3 |
 *          | dg::jacobi_and_deriv() | both, in one pass                            | (A.1.3), (D.2) | Sec. 2.3 |
 *          | dg::jacobi_all()       | \f$P_0,\dots,P_N\f$ (Vandermonde)           | (D.4)          | Sec. 2.4 |
 *          | dg::jacobi_norm2()     | \f$h_n^{\alpha,\beta}\f$                    | (A.1.7), (D.5) | Sec. 2.5 |
 *
 *          **Conventions** (all functions):
 *          - namespace `dg`; the standard-element coordinate is `xi` (\f$\xi\f$);
 *          - templates on the floating-point type `T`, explicitly instantiated for
 *            `float`, `double` and `long double` (see \ref precision "Precision");
 *          - \f$n \ge 0\f$ and \f$\alpha,\beta > -1\f$, otherwise
 *            `std::invalid_argument` is thrown;
 *          - array routines take a raw pointer and a size `(xi, np)`, so they work
 *            on any contiguous storage (`std::vector::data()`, C arrays, Kokkos
 *            host views). Their loops run over the degree outside and over the
 *            points inside, so the inner loop is vectorised by the compiler.
 *
 * \author  Carlos Junqueira Junior (DynFluid, Arts et Métiers – ENSAM)
 */
#pragma once

#include <cstddef>
#include <vector>

/// DGHNOUTS spectral/\f$hp\f$ library.
namespace dg {

/// Implementation details that are not part of the public interface.
namespace detail {
/**
 * \brief Identity type: `nondeduced_t<T>` is `T`, but blocks template argument
 *        deduction.
 *
 * Used for output pointers that may be `nullptr`: the floating-point type `T`
 * is then deduced from the input `xi` only, so `jacobi_and_deriv(n, a, b, xi,
 * np, nullptr, dp)` compiles.
 */
template <typename T> struct nondeduced { using type = T; /**< the type `T` itself */ };
/// Alias for `nondeduced<T>::type`.
template <typename T> using nondeduced_t = typename nondeduced<T>::type;
} // namespace detail

/**
 * \defgroup jacobi Jacobi polynomials
 * \brief Jacobi polynomials \f$P_n^{\alpha,\beta}(\xi)\f$, Karniadakis & Sherwin Appendix A.1.
 *
 * Jacobi polynomials are the polynomial solutions of the singular
 * Sturm–Liouville problem (A.1.1)
 * \f[
 *   (1-\xi)(1+\xi)\,\frac{d^2y}{d\xi^2}
 *   + \bigl[\beta-\alpha-(\alpha+\beta+2)\xi\bigr]\frac{dy}{d\xi}
 *   = -\lambda_n\,y,
 *   \qquad \lambda_n = n(n+\alpha+\beta+1),
 *   \qquad y = P_n^{\alpha,\beta}(\xi),
 * \f]
 * orthogonal on \f$[-1,1]\f$ with respect to the weight
 * \f$(1-\xi)^\alpha(1+\xi)^\beta\f$. Special cases: Legendre
 * \f$L_n = P_n^{0,0}\f$, Chebyshev \f$T_n \propto P_n^{-1/2,-1/2}\f$.
 *
 * \anchor precision
 * \par Precision
 * Every function is a template on the floating-point type `T` of the points
 * and results, deduced from `xi` (or given explicitly for jacobi_norm2()).
 * It is instantiated for `float`, `double` and `long double`; any other type
 * gives a link error. The parameters \f$\alpha,\beta\f$ are always `double`;
 * the recurrence coefficients are computed in `T`.
 * - `double` is the default and the reference precision.
 * - `float` keeps only 4 to 5 correct digits for \f$n\le20\f$: relative
 *   errors measured against `long double` reach \f$5\times10^{-6}\f$ for the
 *   values, \f$7\times10^{-5}\f$ for the derivatives and \f$10^{-5}\f$ for
 *   the norms, since the recurrence and the `lgamma` cancellation amplify the
 *   \f$6\times10^{-8}\f$ rounding of each operation. Use it only to study the
 *   effect of single precision: nodes, weights and matrices for a
 *   single-precision solver should be computed in `double` and converted
 *   afterwards, which keeps them accurate to \f$6\times10^{-8}\f$.
 * - `long double` is 80-bit extended on x86 (about 3 extra digits) and IEEE
 *   quad on Arm (e.g. Rhea, Grace; slow, software-emulated). It is useful to
 *   compute high-order nodes and weights to full `double` accuracy.
 *
 * See \theorypdf, Sections 1 and 2.
 * @{
 */

// ===========================================================================
/// \name Single point
/// @{
// ===========================================================================

/**
 * \brief Value of the Jacobi polynomial \f$P_n^{\alpha,\beta}(\xi)\f$ at one point.
 *
 * Evaluated with the three-term recursion relation (A.1.3):
 * \f[
 * \begin{aligned}
 *   P_0^{\alpha,\beta}(\xi) &= 1, \qquad
 *   P_1^{\alpha,\beta}(\xi) = \tfrac12\bigl[\alpha-\beta+(\alpha+\beta+2)\xi\bigr],\\
 *   a_n^1\,P_{n+1}^{\alpha,\beta}(\xi) &= \bigl(a_n^2 + a_n^3 \xi\bigr)P_n^{\alpha,\beta}(\xi)
 *                                    - a_n^4\,P_{n-1}^{\alpha,\beta}(\xi),
 * \end{aligned}
 * \f]
 * with
 * \f[
 * \begin{aligned}
 *   a_n^1 &= 2(n+1)(n+\alpha+\beta+1)(2n+\alpha+\beta), &
 *   a_n^2 &= (2n+\alpha+\beta+1)(\alpha^2-\beta^2),\\
 *   a_n^3 &= (2n+\alpha+\beta)(2n+\alpha+\beta+1)(2n+\alpha+\beta+2), &
 *   a_n^4 &= 2(n+\alpha)(n+\beta)(2n+\alpha+\beta+2).
 * \end{aligned}
 * \f]
 * In the code the recurrence step is indexed by `k` = 1, ..., `n`−1, since `n`
 * is already the target degree: step `k` is the book's (A.1.3) with \f$n=k\f$.
 * Cost: \f$O(n)\f$.
 *
 * \tparam    T      floating-point type: `float`, `double` or `long double`
 * \param[in] n      degree, \f$n \ge 0\f$
 * \param[in] alpha  weight parameter \f$\alpha > -1\f$
 * \param[in] beta   weight parameter \f$\beta > -1\f$
 * \param[in] xi     evaluation point \f$\xi\f$; any real value, normally \f$\xi\in[-1,1]\f$
 * \return    \f$P_n^{\alpha,\beta}(\xi)\f$
 * \throws    std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.2, Eq. (A.1.3).
 *
 * \see jacobi(int, double, double, const T*, std::size_t, T*) for many points.
 */
template <typename T>
T jacobi(int n, double alpha, double beta, T xi);

/**
 * \brief Derivative \f$\frac{d}{d\xi}P_n^{\alpha,\beta}(\xi)\f$ at one point.
 *
 * Computed by degree reduction, relation (A.1.8):
 * \f[
 *   \frac{d}{d\xi}P_n^{\alpha,\beta}(\xi)
 *   = \frac12\,(\alpha+\beta+n+1)\,P_{n-1}^{\alpha+1,\beta+1}(\xi),
 *   \qquad \frac{d}{d\xi}P_0^{\alpha,\beta} = 0,
 * \f]
 * where the right-hand side is evaluated with jacobi().
 *
 * \note The book also gives the derivative through (A.1.4), whose left-hand
 *       side carries the factor \f$b_n^1(\xi) = (2n+\alpha+\beta)(1-\xi^2)\f$.
 *       That relation (used by Polylib's `jacobfd`) fails at \f$\xi=\pm1\f$,
 *       the end points of every Gauss–Lobatto rule. (A.1.8) is valid on the
 *       whole closed interval \f$[-1,1]\f$, which is why it is used here.
 *
 * \tparam    T      floating-point type: `float`, `double` or `long double`
 * \param[in] n      degree, \f$n \ge 0\f$
 * \param[in] alpha  weight parameter \f$\alpha > -1\f$
 * \param[in] beta   weight parameter \f$\beta > -1\f$
 * \param[in] xi     evaluation point \f$\xi\f$, including \f$\xi=\pm1\f$
 * \return    \f$\frac{d}{d\xi}P_n^{\alpha,\beta}(\xi)\f$
 * \throws    std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.3, Eqs. (A.1.4) and (A.1.8).
 */
template <typename T>
T jacobi_deriv(int n, double alpha, double beta, T xi);

/// @}

// ===========================================================================
/// \name Arrays of points
/// @{
// ===========================================================================

/**
 * \brief Values \f$p_i = P_n^{\alpha,\beta}(\xi_i)\f$ at \f$n_p\f$ points.
 *
 * Same recurrence (A.1.3) as the single-point jacobi(), with the loops
 * ordered for vectorisation:
 * \code
 * for k = 1 .. n-1                      // degree: sequential
 *     compute a1..a4 once
 *     for i = 0 .. np-1                 // points: independent -> SIMD
 *         p_{k+1}[i] = ((a2 + a3 xi[i]) p_k[i] - a4 p_{k-1}[i]) / a1
 * \endcode
 * The operation count is \f$O(n\,n_p)\f$, as for \f$n_p\f$ scalar calls; the
 * speed-up (about \f$\times5\f$ on \f$10^6\f$ points) comes from SIMD and from
 * computing the coefficients once per degree. The results are bitwise
 * identical to the single-point version.
 *
 * \tparam     T      floating-point type: `float`, `double` or `long double`
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  xi     points \f$\xi_i\f$, array of size `np`
 * \param[in]  np     number of points (may be 0)
 * \param[out] p      values, array of size `np`; must not alias `xi`
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.2, Eq. (A.1.3).
 */
template <typename T>
void jacobi(int n, double alpha, double beta,
            const T* xi, std::size_t np, T* p);

/**
 * \brief Derivatives \f$dp_i = \frac{d}{d\xi}P_n^{\alpha,\beta}(\xi_i)\f$ at \f$n_p\f$ points.
 *
 * Uses (A.1.8): one call to the array version of jacobi() with parameters
 * \f$(n-1,\alpha+1,\beta+1)\f$, followed by the scaling
 * \f$\tfrac12(n+\alpha+\beta+1)\f$. Valid at \f$\xi_i=\pm1\f$.
 * Results are bitwise identical to the single-point jacobi_deriv().
 *
 * \tparam     T      floating-point type: `float`, `double` or `long double`
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  xi     points \f$\xi_i\f$, array of size `np`
 * \param[in]  np     number of points (may be 0)
 * \param[out] dp     derivatives, array of size `np`; must not alias `xi`
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.3, Eq. (A.1.8).
 */
template <typename T>
void jacobi_deriv(int n, double alpha, double beta,
                  const T* xi, std::size_t np, T* dp);

/**
 * \brief Values and derivatives of \f$P_n^{\alpha,\beta}\f$ at \f$n_p\f$ points, in one pass.
 *
 * Runs the recurrence (A.1.3) for the values together with its
 * \f$\xi\f$-derivative, Eq. (D.2) of the theory document:
 * \f[
 *   a_n^1\,\bigl[P_{n+1}^{\alpha,\beta}\bigr]'
 *   = a_n^3\,P_n^{\alpha,\beta}
 *   + \bigl(a_n^2 + a_n^3 \xi\bigr)\bigl[P_n^{\alpha,\beta}\bigr]'
 *   - a_n^4\,\bigl[P_{n-1}^{\alpha,\beta}\bigr]',
 * \f]
 * started with \f$[P_0^{\alpha,\beta}]' = 0\f$ and
 * \f$[P_1^{\alpha,\beta}]' = \tfrac12(\alpha+\beta+2)\f$ (D.3).
 * Like (A.1.8) it has no division by \f$1-\xi^2\f$ and holds at \f$\xi=\pm1\f$.
 *
 * \par Implementation
 * The four recurrence arrays (\f$P_k\f$, \f$P_{k-1}\f$ and their derivatives)
 * are processed in blocks of 256 points, so that they stay in the L1 cache
 * for all \f$n\f$ steps. Without blocking, the routine is memory-bound for
 * large \f$n_p\f$ and slower than two separate calls.
 *
 * \tparam     T      floating-point type: `float`, `double` or `long double`
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  xi     points \f$\xi_i\f$, array of size `np`
 * \param[in]  np     number of points (may be 0)
 * \param[out] p      values \f$P_n^{\alpha,\beta}(\xi_i)\f$, size `np`, or `nullptr` if not needed
 * \param[out] dp     derivatives \f$[P_n^{\alpha,\beta}]'(\xi_i)\f$, size `np`, or `nullptr` if not needed
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \note The derivative agrees with jacobi_deriv() to about \f$10^{-14}\f$
 *       (relative, in double) for \f$n \le 20\f$, not bitwise, since it comes
 *       from a different recurrence.
 *
 * \par Theory
 * \theorypdf, Section 2.3, Eqs. (A.1.3), (D.2) and (D.3).
 */
template <typename T>
void jacobi_and_deriv(int n, double alpha, double beta,
                      const T* xi, std::size_t np,
                      detail::nondeduced_t<T>* p, detail::nondeduced_t<T>* dp);

/**
 * \brief All degrees \f$P_0^{\alpha,\beta},\dots,P_N^{\alpha,\beta}\f$ at \f$n_p\f$
 *        points: the generalised Vandermonde matrix.
 *
 * Fills, Eq. (D.4),
 * \f[
 *   \mathbf{V}_{ik} = P_k^{\alpha,\beta}(\xi_i),
 *   \qquad i = 0,\dots,n_p-1,\quad k = 0,\dots,N,
 * \f]
 * in a single pass of the recurrence (A.1.3): each degree is computed exactly
 * once, instead of \f$N+1\f$ recurrences of increasing length.
 *
 * With \f$\alpha=\beta=0\f$ and the nodes \f$\xi_i\f$ of a DG element,
 * \f$\mathbf{V}\f$ maps modal (Legendre) coefficients to nodal values,
 * \f$u(\xi_i) = \sum_k \mathbf{V}_{ik}\,\hat u_k\f$: the transform used by
 * modal filters and shock sensors.
 *
 * \par Storage
 * Row-major, one row per point and one column per degree:
 * `V[i*(nmax+1) + k]` \f$= P_k^{\alpha,\beta}(\xi_i)\f$.
 *
 * \tparam     T      floating-point type: `float`, `double` or `long double`
 * \param[in]  nmax   highest degree \f$N \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  xi     points \f$\xi_i\f$, array of size `np`
 * \param[in]  np     number of points (may be 0)
 * \param[out] V      matrix, array of size `np*(nmax+1)`
 * \throws     std::invalid_argument if \f$N<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.4, Eq. (D.4).
 */
template <typename T>
void jacobi_all(int nmax, double alpha, double beta,
                const T* xi, std::size_t np, T* V);

/// @}

// ===========================================================================
/// \name std::vector convenience overloads
/// @{
// ===========================================================================

/**
 * \brief Values \f$P_n^{\alpha,\beta}(\xi_i)\f$ for every element of a `std::vector`.
 *
 * Resizes `p` to `xi.size()` and calls
 * jacobi(int, double, double, const T*, std::size_t, T*).
 *
 * \tparam     T      floating-point type: `float`, `double` or `long double`
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  xi     points \f$\xi_i\f$
 * \param[out] p      values, resized to `xi.size()`
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 */
template <typename T>
void jacobi(int n, double alpha, double beta,
            const std::vector<T>& xi, std::vector<T>& p);

/**
 * \brief Derivatives \f$\frac{d}{d\xi}P_n^{\alpha,\beta}(\xi_i)\f$ for every element of a `std::vector`.
 *
 * Resizes `dp` to `xi.size()` and calls
 * jacobi_deriv(int, double, double, const T*, std::size_t, T*).
 *
 * \tparam     T      floating-point type: `float`, `double` or `long double`
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  xi     points \f$\xi_i\f$
 * \param[out] dp     derivatives, resized to `xi.size()`
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 */
template <typename T>
void jacobi_deriv(int n, double alpha, double beta,
                  const std::vector<T>& xi, std::vector<T>& dp);

/// @}

// ===========================================================================
/// \name Norm
/// @{
// ===========================================================================

/**
 * \brief Squared weighted \f$L^2\f$ norm \f$h_n^{\alpha,\beta}\f$ of \f$P_n^{\alpha,\beta}\f$.
 *
 * From the orthogonality relation (A.1.7):
 * \f[
 *   h_n^{\alpha,\beta}
 *   = \int_{-1}^{1}(1-\xi)^\alpha(1+\xi)^\beta\,\bigl[P_n^{\alpha,\beta}(\xi)\bigr]^2\,d\xi
 *   = \frac{2^{\alpha+\beta+1}}{2n+\alpha+\beta+1}\,
 *     \frac{\Gamma(n+\alpha+1)\,\Gamma(n+\beta+1)}{n!\,\Gamma(n+\alpha+\beta+1)}.
 * \f]
 * For \f$n=0\f$ the formula is \f$0\cdot\infty\f$ when \f$\alpha+\beta=-1\f$
 * (Chebyshev), so the equivalent finite form (D.5) is used:
 * \f[
 *   h_0^{\alpha,\beta}
 *   = \frac{2^{\alpha+\beta+1}\,\Gamma(\alpha+1)\,\Gamma(\beta+1)}{\Gamma(\alpha+\beta+2)}.
 * \f]
 * Both are evaluated from logarithms with `std::lgamma`, since
 * \f$\Gamma(z)\f$ overflows double precision for \f$z > 171.6\f$.
 *
 * Checks: \f$h_n^{0,0} = 2/(2n+1)\f$ (Legendre),
 * \f$h_0^{-1/2,-1/2} = \pi\f$ (Chebyshev).
 *
 * \tparam    T      floating-point type of the result, `double` by default;
 *                   other types are requested explicitly, e.g. `jacobi_norm2<float>(n, a, b)`
 * \param[in] n      degree, \f$n \ge 0\f$
 * \param[in] alpha  weight parameter \f$\alpha > -1\f$
 * \param[in] beta   weight parameter \f$\beta > -1\f$
 * \return    \f$h_n^{\alpha,\beta}\f$
 * \throws    std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.5, Eqs. (A.1.7) and (D.5).
 */
template <typename T = double>
T jacobi_norm2(int n, double alpha, double beta);

/// @}

/** @} */ // end of group jacobi

} // namespace dg
