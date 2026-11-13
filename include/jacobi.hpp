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
 *          \theorypdf, Parts I and II.
 *
 *          | Function               | Computes                                     | Equation       | Theory |
 *          |------------------------|----------------------------------------------|----------------|--------|
 *          | dg::jacobi()           | \f$P_n^{\alpha,\beta}(\xi)\f$               | (A.1.3)        | Sec. 2.2 |
 *          | dg::jacobi_deriv()     | \f$\frac{d}{d\xi}P_n^{\alpha,\beta}(\xi)\f$ | (A.1.8)        | Sec. 2.3 |
 *          | dg::jacobi_and_deriv() | both, in one pass                            | (A.1.3), (D.2) | Sec. 2.3 |
 *          | dg::jacobi_all()       | \f$P_0,\dots,P_N\f$ (Vandermonde)           | (D.4)          | Sec. 2.4 |
 *          | dg::jacobi_norm2()     | \f$h_n^{\alpha,\beta}\f$                    | (A.1.7), (D.5) | Sec. 2.5 |
 *          | dg::jacobi_zeros()     | zeros \f$\xi_{i,m}^{\alpha,\beta}\f$ of \f$P_m^{\alpha,\beta}\f$ | B.2, (D.7)–(D.11) | Sec. 4 |
 *
 *          **Conventions** (all functions):
 *          - namespace `dg`; the standard-element coordinate is `xi` (\f$\xi\f$);
 *          - templates on the floating-point type `T`, explicitly instantiated for
 *            `float`, `double` and `long double` (see \ref precision "Precision");
 *          - \f$n \ge 0\f$ and \f$\alpha,\beta > -1\f$, otherwise
 *            `std::invalid_argument` is thrown;
 *          - array routines take a raw pointer and a size `(xi, np)`, so they work
 *            on any contiguous storage (`std::vector::data()`, C arrays, Kokkos
 *            host views). They are register-blocked: the recurrence runs on
 *            blocks of 32 points held in SIMD registers (see \ref performance
 *            "Performance").
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
 * \anchor performance
 * \par Performance
 * - **Normalised recurrence (A.1.3').** The coefficients are divided by
 *   \f$a_n^1\f$ once, \f$P_{n+1} = (A_n + B_n\xi)P_n - C_nP_{n-1}\f$, so a step
 *   costs two multiply-adds and a multiplication, without division. The array
 *   routines and jacobi_zeros() compute the table \f$A_k,B_k,C_k\f$ once per call.
 * - **Register blocking.** The array routines run the whole recurrence on
 *   blocks of 32 points kept in SIMD registers, instead of sweeping the arrays
 *   in memory once per degree. Each point's recurrence is a dependency chain;
 *   32 doubles give 8 independent AVX2 registers, enough to hide its latency.
 * - **Fused multiply-add** is used when the hardware has it (`-march=native`,
 *   CMake option `DGPOLY_NATIVE=ON`); it halves the latency of a step.
 * - Measured with `bench_jacobi` (AVX-512 Xeon, `-march=native`, double,
 *   \f$n=16\f$): \f$\times3\f$ to \f$\times9\f$ for the array routines,
 *   \f$\times3\f$ for jacobi_zeros(), compared with the earlier
 *   degree-outside/points-inside loops. The single-point routines are latency-bound
 *   by nature: for many points, use the array versions.
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

/**
 * \brief Value and derivative of \f$P_n^{\alpha,\beta}\f$ at one point, in one pass.
 *
 * Runs the recurrence (A.1.3) for the value together with its
 * \f$\xi\f$-derivative (D.2), started with (D.3):
 * \f[
 *   a_n^1\,\bigl[P_{n+1}^{\alpha,\beta}\bigr]'
 *   = a_n^3\,P_n^{\alpha,\beta}
 *   + \bigl(a_n^2 + a_n^3 \xi\bigr)\bigl[P_n^{\alpha,\beta}\bigr]'
 *   - a_n^4\,\bigl[P_{n-1}^{\alpha,\beta}\bigr]'.
 * \f]
 * This is what the Newton iteration of jacobi_zeros() evaluates at every
 * step: one recurrence instead of two (jacobi() and jacobi_deriv()).
 * The results are bitwise identical to the array version of
 * jacobi_and_deriv() at the same point. Valid at \f$\xi=\pm1\f$.
 *
 * \tparam     T      floating-point type: `float`, `double` or `long double`
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  xi     evaluation point \f$\xi\f$
 * \param[out] p      \f$P_n^{\alpha,\beta}(\xi)\f$
 * \param[out] dp     \f$\frac{d}{d\xi}P_n^{\alpha,\beta}(\xi)\f$
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.3, Eqs. (A.1.3), (D.2) and (D.3).
 */
template <typename T>
void jacobi_and_deriv(int n, double alpha, double beta, T xi, T& p, T& dp);

/// @}

// ===========================================================================
/// \name Arrays of points
/// @{
// ===========================================================================

/**
 * \brief Values \f$p_i = P_n^{\alpha,\beta}(\xi_i)\f$ at \f$n_p\f$ points.
 *
 * Same recurrence as the single-point jacobi(), in its normalised form
 * (A.1.3'), register-blocked:
 * \code
 * compute A_k, B_k, C_k once                  // k = 1 .. n-1
 * for each block of 32 points                 // xi, P_k, P_{k-1} in registers
 *     for k = 1 .. n-1                         // degree: sequential
 *         for i in block                       // points: independent -> SIMD
 *             q = (A_k + B_k xi[i]) P_k[i] - C_k P_{k-1}[i]
 *             P_{k-1}[i] = P_k[i];  P_k[i] = q
 *     write P_n for the block
 * \endcode
 * The operation count is \f$O(n\,n_p)\f$, as for \f$n_p\f$ scalar calls. The
 * speed-up comes from SIMD, from the 8 independent register chains per block
 * that hide the latency of each step, and from touching memory only once per
 * point: \f$\times3.4\f$ on 4096 points and \f$\times5.8\f$ on \f$10^6\f$ points
 * against the earlier loop over degrees outside and points inside (double,
 * \f$n=16\f$, AVX-512). The results are bitwise identical to the single-point
 * version.
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
 * Register-blocked like jacobi(): for each block of 32 points, the four
 * recurrence variables (\f$P_k\f$, \f$P_{k-1}\f$ and their derivatives) stay
 * in registers for all \f$n\f$ steps, using the normalised forms (A.1.3') and
 * (D.2'). This replaces an earlier L1-cache blocking of 256 points:
 * \f$\times3.2\f$ to \f$\times3.9\f$ faster (double, \f$n=16\f$, AVX-512).
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
 * \par Implementation
 * Register-blocked like jacobi(): each block of 32 points fills its own rows
 * of \f$\mathbf{V}\f$ while they are in the L1 cache (the earlier version wrote
 * \f$\mathbf{V}\f$ column by column, touching every row at every degree).
 * For large \f$n_p\f$ the routine is limited by writing its own output,
 * \f$N+1\f$ values per point.
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

// ===========================================================================
/// \name Zeros
/// @{
// ===========================================================================

/**
 * \brief Zeros \f$\xi_{0,m}^{\alpha,\beta} < \dots < \xi_{m-1,m}^{\alpha,\beta}\f$
 *        of \f$P_m^{\alpha,\beta}\f$: the nodes of the Gauss-type quadrature rules.
 *
 * The \f$m\f$ zeros are real, simple and lie in \f$(-1,1)\f$. They give the
 * nodes of every Gauss-type rule with \f$Q\f$ points (Appendix B.1):
 *
 * | Rule           | Nodes \f$\xi_i\f$                                                     |
 * |----------------|-------------------------------------------------------------------------|
 * | Gauss–Jacobi   | zeros of \f$P_Q^{\alpha,\beta}\f$                                      |
 * | Gauss–Radau    | \f$-1\f$ and the zeros of \f$P_{Q-1}^{\alpha,\beta+1}\f$               |
 * | Gauss–Lobatto  | \f$\pm1\f$ and the zeros of \f$P_{Q-2}^{\alpha+1,\beta+1}\f$           |
 *
 * \par Algorithm (Appendix B.2)
 * Newton–Raphson with polynomial deflation. Root \f$k\f$ is found by applying
 * Newton to \f$f_{m-k}(\xi) = P_m(\xi)/\prod_{j<k}(\xi-\xi_j)\f$, so that the
 * roots already found are divided out; the Newton step is (D.7)
 * \f[
 *   \delta = -\frac{P_m(r)}{P_m'(r) - P_m(r)\sum_{j=0}^{k-1}\frac{1}{r-\xi_j}},
 *   \qquad r \leftarrow r + \delta .
 * \f]
 * The initial guess (D.8) is the Chebyshev zero
 * \f$-\cos\bigl(\tfrac{2k+1}{2m}\pi\bigr)\f$, averaged with the previous root
 * \f$\xi_{k-1}\f$ for \f$k>0\f$, as recommended in the book.
 * \f$P_m\f$ and \f$P_m'\f$ come from one pass of (A.1.3) and (D.2).
 * Root \f$k\f$ is accepted when \f$|\delta| \le 4\varepsilon\f$ (D.9), with
 * \f$\varepsilon\f$ the machine epsilon of `T`; since \f$|\xi|<1\f$ this is a
 * relative criterion.
 *
 * \par Additions to the book's algorithm
 * - **Reflection (D.10).** For \f$\alpha>\beta\f$ the zeros crowd towards
 *   \f$\xi=-1\f$, where the search starts, and Newton can fail. By (A.1.6) the
 *   zeros of \f$P_m^{\alpha,\beta}\f$ are the negated zeros of
 *   \f$P_m^{\beta,\alpha}\f$, which are computed instead.
 * - **Exact symmetry.** For \f$\alpha=\beta\f$ the zeros are made exactly
 *   symmetric, \f$\xi_{m-1-i}=-\xi_i\f$, and \f$\xi_{(m-1)/2}=0\f$ for odd
 *   \f$m\f$, so that odd functions are integrated to exactly zero.
 * - **Certification (D.11).** \f$P_m\f$ must change sign between consecutive
 *   points of \f$-1\f$, the midpoints of consecutive zeros, and \f$+1\f$. These
 *   \f$m\f$ sign changes prove that each computed zero sits alone in its own
 *   interval: none was missed or found twice. A failure raises an exception
 *   instead of returning wrong zeros.
 * - The iteration limit is 1000. DG parameters need at most 11 iterations;
 *   far from the Chebyshev guess (e.g. \f$\alpha=\beta=20\f$) Newton first
 *   creeps towards the root and needs up to about 170.
 *
 * \par Cost
 * Each Newton step evaluates \f$P_m\f$ and \f$P_m'\f$ at one point: a single
 * dependency chain of \f$m\f$ steps, so the cost is about
 * \f$m^2\times\f$(iterations) chain steps. The coefficient table is built
 * once per call, the deflation sum uses eight partial sums (SIMD divisions),
 * and the certification evaluates its \f$m+1\f$ points in one array call.
 * Measured (GLL, double, AVX-512): 4.5 µs for \f$m=16\f$, 22 µs for
 * \f$m=32\f$, 80 µs for \f$m=64\f$, about 3 times faster than without these
 * optimisations; negligible in a solver, where it runs once.
 *
 * \par Accuracy and range (measured, \f$m \le 60\f$)
 * - `double`: within \f$2\times10^{-16}\f$ of the `long double` zeros (1–2 ulp)
 *   for \f$\alpha,\beta\f$ up to 100; Polylib's eigenvalue method agrees only
 *   to \f$10^{-15}\f$.
 * - `long double`: no failure in the same range.
 * - `float`: within \f$10^{-7}\f$ (1–2 ulp) for the DG parameters
 *   \f$|\alpha|,|\beta|\le2\f$. For large parameters (e.g. \f$\alpha=\beta\ge5\f$
 *   with \f$m>40\f$, or \f$\beta=100\f$) it raises `std::runtime_error`; compute
 *   in `double` and round instead.
 *
 * \tparam     T      floating-point type: `float`, `double` or `long double`
 * \param[in]  m      degree of \f$P_m^{\alpha,\beta}\f$ = number of zeros, \f$m \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[out] z      the \f$m\f$ zeros in increasing order, array of size `m`
 * \throws     std::invalid_argument if \f$m<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 * \throws     std::runtime_error if Newton does not converge for a zero, or if
 *             the certification (D.11) fails; this happens only in `float`
 *             beyond the DG parameter range
 *
 * \par Theory
 * \theorypdf, Section 4, Eqs. (D.7)–(D.11), and Karniadakis & Sherwin,
 * Appendix B.2.
 */
template <typename T>
void jacobi_zeros(int m, double alpha, double beta, T* z);

/**
 * \brief Zeros of \f$P_m^{\alpha,\beta}\f$, returned in a `std::vector`.
 *
 * Convenience overload of jacobi_zeros(int, double, double, T*). The type is
 * `double` unless requested explicitly, e.g. `jacobi_zeros<long double>(m, a, b)`.
 *
 * \tparam    T      floating-point type, `double` by default
 * \param[in] m      number of zeros, \f$m \ge 0\f$
 * \param[in] alpha  weight parameter \f$\alpha > -1\f$
 * \param[in] beta   weight parameter \f$\beta > -1\f$
 * \return    the \f$m\f$ zeros in increasing order
 * \throws    see jacobi_zeros(int, double, double, T*)
 */
template <typename T = double>
std::vector<T> jacobi_zeros(int m, double alpha, double beta);

/// @}

/** @} */ // end of group jacobi

} // namespace dg
