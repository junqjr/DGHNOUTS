/**
 * \file    jacobi.hpp
 * \brief   Jacobi polynomials \f$P_n^{\alpha,\beta}(x)\f$ on \f$[-1,1]\f$:
 *          values, derivatives, Vandermonde matrix and norms.
 *
 * \details Public interface of the Jacobi module of the DGHNOUTS spectral/\f$hp\f$
 *          library. Notation and equation numbers follow Karniadakis & Sherwin,
 *          *Spectral/hp Element Methods for CFD*, 2nd ed., Appendix A.1.
 *          Equations derived in DGHNOUTS (not in the book) are numbered (D.n)
 *          and are given in the theory document \theorypdf, Part I.
 *
 *          | Function             | Computes                                   | Equation       | Theory |
 *          |----------------------|--------------------------------------------|----------------|--------|
 *          | dg::jacobi()         | \f$P_n^{\alpha,\beta}(x)\f$               | (A.1.3)        | Sec. 2.2 |
 *          | dg::jacobi_deriv()   | \f$\frac{d}{dx}P_n^{\alpha,\beta}(x)\f$   | (A.1.8)        | Sec. 2.3 |
 *          | dg::jacobi_and_deriv() | both, in one pass                        | (A.1.3), (D.2) | Sec. 2.3 |
 *          | dg::jacobi_all()     | \f$P_0,\dots,P_N\f$ (Vandermonde)         | (D.4)          | Sec. 2.4 |
 *          | dg::jacobi_norm2()   | \f$h_n^{\alpha,\beta}\f$                  | (A.1.7), (D.5) | Sec. 2.5 |
 *
 *          **Conventions** (all functions):
 *          - double precision, namespace `dg`;
 *          - \f$n \ge 0\f$ and \f$\alpha,\beta > -1\f$, otherwise
 *            `std::invalid_argument` is thrown;
 *          - array routines take a raw pointer and a size `(x, np)`, so they work
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

/**
 * \defgroup jacobi Jacobi polynomials
 * \brief Jacobi polynomials \f$P_n^{\alpha,\beta}(x)\f$, Karniadakis & Sherwin Appendix A.1.
 *
 * Jacobi polynomials are the polynomial solutions of the singular
 * Sturm–Liouville problem (A.1.1)
 * \f[
 *   (1-x)(1+x)\,\frac{d^2y}{dx^2}
 *   + \bigl[\beta-\alpha-(\alpha+\beta+2)x\bigr]\frac{dy}{dx}
 *   = -\lambda_n\,y,
 *   \qquad \lambda_n = n(n+\alpha+\beta+1),
 *   \qquad y = P_n^{\alpha,\beta}(x),
 * \f]
 * orthogonal on \f$[-1,1]\f$ with respect to the weight
 * \f$(1-x)^\alpha(1+x)^\beta\f$. Special cases: Legendre
 * \f$L_n = P_n^{0,0}\f$, Chebyshev \f$T_n \propto P_n^{-1/2,-1/2}\f$.
 *
 * See \theorypdf, Section 2.
 * @{
 */

// ===========================================================================
/// \name Single point
/// @{
// ===========================================================================

/**
 * \brief Value of the Jacobi polynomial \f$P_n^{\alpha,\beta}(x)\f$ at one point.
 *
 * Evaluated with the three-term recursion relation (A.1.3):
 * \f[
 * \begin{aligned}
 *   P_0^{\alpha,\beta}(x) &= 1, \qquad
 *   P_1^{\alpha,\beta}(x) = \tfrac12\bigl[\alpha-\beta+(\alpha+\beta+2)x\bigr],\\
 *   a_k^1\,P_{k+1}^{\alpha,\beta}(x) &= \bigl(a_k^2 + a_k^3 x\bigr)P_k^{\alpha,\beta}(x)
 *                                   - a_k^4\,P_{k-1}^{\alpha,\beta}(x),
 *   \qquad k = 1,\dots,n-1,
 * \end{aligned}
 * \f]
 * with
 * \f[
 * \begin{aligned}
 *   a_k^1 &= 2(k+1)(k+\alpha+\beta+1)(2k+\alpha+\beta), &
 *   a_k^2 &= (2k+\alpha+\beta+1)(\alpha^2-\beta^2),\\
 *   a_k^3 &= (2k+\alpha+\beta)(2k+\alpha+\beta+1)(2k+\alpha+\beta+2), &
 *   a_k^4 &= 2(k+\alpha)(k+\beta)(2k+\alpha+\beta+2).
 * \end{aligned}
 * \f]
 * The loop index \f$k\f$ is the book's \f$n\f$ in (A.1.3).
 * Cost: \f$O(n)\f$.
 *
 * \param[in] n      degree, \f$n \ge 0\f$
 * \param[in] alpha  weight parameter \f$\alpha > -1\f$
 * \param[in] beta   weight parameter \f$\beta > -1\f$
 * \param[in] x      evaluation point; any real value, normally \f$x\in[-1,1]\f$
 * \return    \f$P_n^{\alpha,\beta}(x)\f$
 * \throws    std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.2, Eq. (A.1.3).
 *
 * \see jacobi(int, double, double, const double*, std::size_t, double*) for many points.
 */
double jacobi(int n, double alpha, double beta, double x);

/**
 * \brief Derivative \f$\frac{d}{dx}P_n^{\alpha,\beta}(x)\f$ at one point.
 *
 * Computed by degree reduction, relation (A.1.8):
 * \f[
 *   \frac{d}{dx}P_n^{\alpha,\beta}(x)
 *   = \frac12\,(\alpha+\beta+n+1)\,P_{n-1}^{\alpha+1,\beta+1}(x),
 *   \qquad \frac{d}{dx}P_0^{\alpha,\beta} = 0,
 * \f]
 * where the right-hand side is evaluated with jacobi().
 *
 * \note The book also gives the derivative through (A.1.4), whose left-hand
 *       side carries the factor \f$b_n^1(x) = (2n+\alpha+\beta)(1-x^2)\f$.
 *       That relation (used by Polylib's `jacobfd`) fails at \f$x=\pm1\f$,
 *       the end points of every Gauss–Lobatto rule. (A.1.8) is valid on the
 *       whole closed interval \f$[-1,1]\f$, which is why it is used here.
 *
 * \param[in] n      degree, \f$n \ge 0\f$
 * \param[in] alpha  weight parameter \f$\alpha > -1\f$
 * \param[in] beta   weight parameter \f$\beta > -1\f$
 * \param[in] x      evaluation point, including \f$x=\pm1\f$
 * \return    \f$\frac{d}{dx}P_n^{\alpha,\beta}(x)\f$
 * \throws    std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.3, Eqs. (A.1.4) and (A.1.8).
 */
double jacobi_deriv(int n, double alpha, double beta, double x);

/// @}

// ===========================================================================
/// \name Arrays of points
/// @{
// ===========================================================================

/**
 * \brief Values \f$p_i = P_n^{\alpha,\beta}(x_i)\f$ at \f$n_p\f$ points.
 *
 * Same recurrence (A.1.3) as the single-point jacobi(), with the loops
 * ordered for vectorisation:
 * \code
 * for k = 1 .. n-1                      // degree: sequential
 *     compute a1..a4 once
 *     for i = 0 .. np-1                 // points: independent -> SIMD
 *         p_{k+1}[i] = ((a2 + a3 x[i]) p_k[i] - a4 p_{k-1}[i]) / a1
 * \endcode
 * The operation count is \f$O(n\,n_p)\f$, as for \f$n_p\f$ scalar calls; the
 * speed-up (about \f$\times5\f$ on \f$10^6\f$ points) comes from SIMD and from
 * computing the coefficients once per degree. The results are bitwise
 * identical to the single-point version.
 *
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  x      points, array of size `np`
 * \param[in]  np     number of points (may be 0)
 * \param[out] p      values, array of size `np`; must not alias `x`
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.2, Eq. (A.1.3).
 */
void jacobi(int n, double alpha, double beta,
            const double* x, std::size_t np, double* p);

/**
 * \brief Derivatives \f$dp_i = \frac{d}{dx}P_n^{\alpha,\beta}(x_i)\f$ at \f$n_p\f$ points.
 *
 * Uses (A.1.8): one call to the array version of jacobi() with parameters
 * \f$(n-1,\alpha+1,\beta+1)\f$, followed by the scaling
 * \f$\tfrac12(n+\alpha+\beta+1)\f$. Valid at \f$x_i=\pm1\f$.
 * Results are bitwise identical to the single-point jacobi_deriv().
 *
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  x      points, array of size `np`
 * \param[in]  np     number of points (may be 0)
 * \param[out] dp     derivatives, array of size `np`; must not alias `x`
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.3, Eq. (A.1.8).
 */
void jacobi_deriv(int n, double alpha, double beta,
                  const double* x, std::size_t np, double* dp);

/**
 * \brief Values and derivatives of \f$P_n^{\alpha,\beta}\f$ at \f$n_p\f$ points, in one pass.
 *
 * Runs the recurrence (A.1.3) for the values together with its
 * \f$x\f$-derivative, Eq. (D.2) of the theory document:
 * \f[
 *   a_k^1\,\bigl[P_{k+1}^{\alpha,\beta}\bigr]'
 *   = a_k^3\,P_k^{\alpha,\beta}
 *   + \bigl(a_k^2 + a_k^3 x\bigr)\bigl[P_k^{\alpha,\beta}\bigr]'
 *   - a_k^4\,\bigl[P_{k-1}^{\alpha,\beta}\bigr]',
 * \f]
 * started with \f$[P_0^{\alpha,\beta}]' = 0\f$ and
 * \f$[P_1^{\alpha,\beta}]' = \tfrac12(\alpha+\beta+2)\f$ (D.3).
 * Like (A.1.8) it has no division by \f$1-x^2\f$ and holds at \f$x=\pm1\f$.
 *
 * \par Implementation
 * The four recurrence arrays (\f$P_k\f$, \f$P_{k-1}\f$ and their derivatives)
 * are processed in blocks of 256 points, so that they stay in the L1 cache
 * for all \f$n\f$ steps. Without blocking, the routine is memory-bound for
 * large \f$n_p\f$ and slower than two separate calls.
 *
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  x      points, array of size `np`
 * \param[in]  np     number of points (may be 0)
 * \param[out] p      values \f$P_n^{\alpha,\beta}(x_i)\f$, size `np`, or `nullptr` if not needed
 * \param[out] dp     derivatives \f$[P_n^{\alpha,\beta}]'(x_i)\f$, size `np`, or `nullptr` if not needed
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \note The derivative agrees with jacobi_deriv() to about \f$10^{-14}\f$
 *       (relative) for \f$n \le 20\f$, not bitwise, since it comes from a
 *       different recurrence.
 *
 * \par Theory
 * \theorypdf, Section 2.3, Eqs. (A.1.3), (D.2) and (D.3).
 */
void jacobi_and_deriv(int n, double alpha, double beta,
                      const double* x, std::size_t np, double* p, double* dp);

/**
 * \brief All degrees \f$P_0^{\alpha,\beta},\dots,P_N^{\alpha,\beta}\f$ at \f$n_p\f$
 *        points: the generalised Vandermonde matrix.
 *
 * Fills, Eq. (D.4),
 * \f[
 *   \mathbf{V}_{ik} = P_k^{\alpha,\beta}(x_i),
 *   \qquad i = 0,\dots,n_p-1,\quad k = 0,\dots,N,
 * \f]
 * in a single pass of the recurrence (A.1.3): each degree is computed exactly
 * once, instead of \f$N+1\f$ recurrences of increasing length.
 *
 * With \f$\alpha=\beta=0\f$ and the nodes \f$x_i\f$ of a DG element,
 * \f$\mathbf{V}\f$ maps modal (Legendre) coefficients to nodal values,
 * \f$u(x_i) = \sum_k \mathbf{V}_{ik}\,\hat u_k\f$: the transform used by
 * modal filters and shock sensors.
 *
 * \par Storage
 * Row-major, one row per point and one column per degree:
 * `V[i*(nmax+1) + k]` \f$= P_k^{\alpha,\beta}(x_i)\f$.
 *
 * \param[in]  nmax   highest degree \f$N \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  x      points, array of size `np`
 * \param[in]  np     number of points (may be 0)
 * \param[out] V      matrix, array of size `np*(nmax+1)`
 * \throws     std::invalid_argument if \f$N<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.4, Eq. (D.4).
 */
void jacobi_all(int nmax, double alpha, double beta,
                const double* x, std::size_t np, double* V);

/// @}

// ===========================================================================
/// \name std::vector convenience overloads
/// @{
// ===========================================================================

/**
 * \brief Values \f$P_n^{\alpha,\beta}(x_i)\f$ for every element of a `std::vector`.
 *
 * Resizes `p` to `x.size()` and calls
 * jacobi(int, double, double, const double*, std::size_t, double*).
 *
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  x      points
 * \param[out] p      values, resized to `x.size()`
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 */
void jacobi(int n, double alpha, double beta,
            const std::vector<double>& x, std::vector<double>& p);

/**
 * \brief Derivatives \f$\frac{d}{dx}P_n^{\alpha,\beta}(x_i)\f$ for every element of a `std::vector`.
 *
 * Resizes `dp` to `x.size()` and calls
 * jacobi_deriv(int, double, double, const double*, std::size_t, double*).
 *
 * \param[in]  n      degree, \f$n \ge 0\f$
 * \param[in]  alpha  weight parameter \f$\alpha > -1\f$
 * \param[in]  beta   weight parameter \f$\beta > -1\f$
 * \param[in]  x      points
 * \param[out] dp     derivatives, resized to `x.size()`
 * \throws     std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 */
void jacobi_deriv(int n, double alpha, double beta,
                  const std::vector<double>& x, std::vector<double>& dp);

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
 *   = \int_{-1}^{1}(1-x)^\alpha(1+x)^\beta\,\bigl[P_n^{\alpha,\beta}(x)\bigr]^2\,dx
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
 * \param[in] n      degree, \f$n \ge 0\f$
 * \param[in] alpha  weight parameter \f$\alpha > -1\f$
 * \param[in] beta   weight parameter \f$\beta > -1\f$
 * \return    \f$h_n^{\alpha,\beta}\f$
 * \throws    std::invalid_argument if \f$n<0\f$, \f$\alpha\le-1\f$ or \f$\beta\le-1\f$
 *
 * \par Theory
 * \theorypdf, Section 2.5, Eqs. (A.1.7) and (D.5).
 */
double jacobi_norm2(int n, double alpha, double beta);

/// @}

/** @} */ // end of group jacobi

} // namespace dg
