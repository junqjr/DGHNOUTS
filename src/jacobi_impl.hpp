/**
 * \file    jacobi_impl.hpp
 * \brief   Internal header of the Jacobi module: the two implementations and
 *          the helpers they share. Not part of the public interface.
 *
 * \details The Jacobi module has two implementations of the same routines,
 *          with the same signatures and the same results up to round-off:
 *
 *          | Namespace         | File                   | Written for |
 *          |-------------------|------------------------|-------------|
 *          | `dg::reference`   | jacobi_reference.cpp   | reading: the book's equations, one point at a time |
 *          | `dg::optimized`   | jacobi_optimized.cpp   | speed: normalised recurrence, register blocking, FMA |
 *
 *          Both are always compiled. The public functions `dg::jacobi(...)`
 *          etc. (jacobi.cpp) forward to one of them, chosen at compile time by
 *          the macro `DGHNOUTS_REFERENCE_KERNELS`, which the CMake option
 *          `DGHNOUTS_KERNELS=reference` defines (default: `optimized`).
 *          The tests compare the two (test_impl.cpp) and the benchmark times
 *          them side by side (bench_jacobi.cpp).
 *
 *          New to the code? Read jacobi_reference.cpp first, next to the
 *          theory document \theorypdf.
 */
#pragma once

#include "jacobi.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dg {

/// Helpers shared by the two implementations (not part of the public interface).
namespace internal {

/**
 * \brief  Validate the arguments common to all Jacobi routines.
 * \param[in] n      degree; must be \f$\ge 0\f$
 * \param[in] alpha  must be \f$> -1\f$
 * \param[in] beta   must be \f$> -1\f$
 * \throws std::invalid_argument otherwise. For \f$\alpha\le-1\f$ or
 *         \f$\beta\le-1\f$ the weight \f$(1-\xi)^\alpha(1+\xi)^\beta\f$ is not
 *         integrable and the polynomials are not orthogonal.
 */
inline void check_args(int n, double alpha, double beta)
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
 * In the routines, `n` is the target degree, so the recurrence step is called
 * with the loop index: `rec_coef(k, ...)`, \f$k = 1,\dots,n-1\f$.
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

/**
 * \brief  Report that Newton did not converge for a zero (jacobi_zeros()).
 * \tparam    T  floating-point type (the message suggests `double` for `float`)
 * \param[in] k  index of the zero
 * \param[in] m  degree of the polynomial
 * \throws    std::runtime_error always
 */
template <typename T>
[[noreturn]] void throw_newton_failure(int k, int m)
{
    throw std::runtime_error("jacobi_zeros: Newton did not converge for zero "
                             + std::to_string(k) + " of P_" + std::to_string(m)
                             + (std::is_same_v<T, float> ? " (float: compute in double"
                                " and round instead)" : ""));
}

/**
 * \brief  Sort the zeros in increasing order and check that they are distinct.
 * \tparam        T  floating-point type
 * \param[in]     m  number of zeros
 * \param[in,out] z  the zeros
 * \throws std::runtime_error if two zeros coincide (the zeros of \f$P_m\f$ are simple)
 */
template <typename T>
void sort_zeros(int m, T* z)
{
    std::sort(z, z + m);
    for (int k = 1; k < m; ++k)
        if (!(z[k] > z[k - 1]))
            throw std::runtime_error("jacobi_zeros: two zeros of P_" + std::to_string(m)
                                     + " coincide");
}

/**
 * \brief  Make the zeros exactly symmetric, for \f$\alpha=\beta\f$.
 *
 * Then \f$P_m\f$ has the parity of \f$m\f$ (A.1.6):
 * \f$\xi_i \leftarrow \tfrac12(\xi_i-\xi_{m-1-i})\f$, \f$\xi_{m-1-i}=-\xi_i\f$,
 * and the middle zero is exactly 0 for odd \f$m\f$. Odd functions are then
 * integrated to exactly zero.
 *
 * \tparam        T  floating-point type
 * \param[in]     m  number of zeros
 * \param[in,out] z  the zeros, sorted
 */
template <typename T>
void symmetrize_zeros(int m, T* z)
{
    for (int i = 0; i < m / 2; ++i) {
        const T v = T(0.5) * (z[i] - z[m - 1 - i]);
        z[i]         = v;
        z[m - 1 - i] = -v;
    }
    if (m % 2 == 1) z[m / 2] = T(0);
}

/**
 * \brief  Cut points of the certification (D.11):
 *         \f$c_0=-1\f$, \f$c_i=\tfrac12(\xi_{i-1}+\xi_i)\f$, \f$c_m=1\f$.
 * \tparam    T  floating-point type
 * \param[in] m  number of zeros, \f$m\ge1\f$
 * \param[in] z  the zeros, sorted
 * \return    the \f$m+1\f$ points; zero \f$\xi_i\f$ lies in \f$(c_i, c_{i+1})\f$
 */
template <typename T>
std::vector<T> cut_points(int m, const T* z)
{
    std::vector<T> c(static_cast<std::size_t>(m) + 1);
    c.front() = T(-1);
    for (int i = 1; i < m; ++i) c[i] = T(0.5) * (z[i - 1] + z[i]);
    c.back() = T(1);
    return c;
}

/**
 * \brief  Certification (D.11): \f$P_m\f$ must change sign between consecutive cut points.
 *
 * The \f$m+1\f$ cut points delimit \f$m\f$ intervals, each containing one
 * computed zero. \f$m\f$ sign changes prove that each interval holds exactly
 * one true zero, so no zero was missed or found twice. Any failure, including
 * overflow to inf or nan in `float`, is reported instead of returning wrong
 * zeros.
 *
 * \tparam    T   floating-point type
 * \param[in] m   degree, \f$m\ge1\f$
 * \param[in] pc  \f$P_m(c_i)\f$, \f$i=0,\dots,m\f$, at the points of cut_points()
 * \throws    std::runtime_error if a sign change is missing
 */
template <typename T>
void certify_zeros(int m, const T* pc)
{
    for (int i = 0; i < m; ++i) {
        const T u = pc[i], v = pc[i + 1];
        if (!(std::isfinite(u) && std::isfinite(v) && u != T(0) && v != T(0)
              && (u > T(0)) != (v > T(0))))
            throw std::runtime_error("jacobi_zeros: certification failed for P_"
                                     + std::to_string(m) + " (zeros not separated by sign"
                                     " changes)" + (std::is_same_v<T, float> ?
                                     "; float: compute in double and round instead" : ""));
    }
}

} // namespace internal

// ===========================================================================
// The two implementations. Same signatures and meaning as the public
// functions of jacobi.hpp, which document them.
// ===========================================================================

/**
 * \brief Reference implementation: the book's equations written as directly as
 *        possible, one point at a time (jacobi_reference.cpp).
 *
 * Selected by `DGHNOUTS_KERNELS=reference`. Always compiled: test_impl.cpp
 * uses it as the oracle for the optimised implementation.
 */
namespace reference {
/// \copydoc dg::jacobi(int, double, double, T)
template <typename T> T jacobi(int n, double alpha, double beta, T xi);
/// \copydoc dg::jacobi_deriv(int, double, double, T)
template <typename T> T jacobi_deriv(int n, double alpha, double beta, T xi);
/// \copydoc dg::jacobi_and_deriv(int, double, double, T, T&, T&)
template <typename T> void jacobi_and_deriv(int n, double alpha, double beta, T xi, T& p, T& dp);
/// \copydoc dg::jacobi(int, double, double, const T*, std::size_t, T*)
template <typename T> void jacobi(int n, double alpha, double beta, const T* xi, std::size_t np, T* p);
/// \copydoc dg::jacobi_deriv(int, double, double, const T*, std::size_t, T*)
template <typename T> void jacobi_deriv(int n, double alpha, double beta, const T* xi, std::size_t np, T* dp);
/// \copydoc dg::jacobi_and_deriv(int, double, double, const T*, std::size_t, detail::nondeduced_t<T>*, detail::nondeduced_t<T>*)
template <typename T> void jacobi_and_deriv(int n, double alpha, double beta, const T* xi, std::size_t np,
                                            detail::nondeduced_t<T>* p, detail::nondeduced_t<T>* dp);
/// \copydoc dg::jacobi_all(int, double, double, const T*, std::size_t, T*)
template <typename T> void jacobi_all(int nmax, double alpha, double beta, const T* xi, std::size_t np, T* V);
/// \copydoc dg::jacobi_zeros(int, double, double, T*)
template <typename T> void jacobi_zeros(int m, double alpha, double beta, T* z);
} // namespace reference

/**
 * \brief Optimised implementation: normalised recurrence (A.1.3'), register
 *        blocking and fused multiply-add (jacobi_optimized.cpp).
 *
 * Selected by default (`DGHNOUTS_KERNELS=optimized`). See \theorypdf,
 * Section 2.6, for the techniques and the measured speed-ups.
 */
namespace optimized {
/// \copydoc dg::jacobi(int, double, double, T)
template <typename T> T jacobi(int n, double alpha, double beta, T xi);
/// \copydoc dg::jacobi_deriv(int, double, double, T)
template <typename T> T jacobi_deriv(int n, double alpha, double beta, T xi);
/// \copydoc dg::jacobi_and_deriv(int, double, double, T, T&, T&)
template <typename T> void jacobi_and_deriv(int n, double alpha, double beta, T xi, T& p, T& dp);
/// \copydoc dg::jacobi(int, double, double, const T*, std::size_t, T*)
template <typename T> void jacobi(int n, double alpha, double beta, const T* xi, std::size_t np, T* p);
/// \copydoc dg::jacobi_deriv(int, double, double, const T*, std::size_t, T*)
template <typename T> void jacobi_deriv(int n, double alpha, double beta, const T* xi, std::size_t np, T* dp);
/// \copydoc dg::jacobi_and_deriv(int, double, double, const T*, std::size_t, detail::nondeduced_t<T>*, detail::nondeduced_t<T>*)
template <typename T> void jacobi_and_deriv(int n, double alpha, double beta, const T* xi, std::size_t np,
                                            detail::nondeduced_t<T>* p, detail::nondeduced_t<T>* dp);
/// \copydoc dg::jacobi_all(int, double, double, const T*, std::size_t, T*)
template <typename T> void jacobi_all(int nmax, double alpha, double beta, const T* xi, std::size_t np, T* V);
/// \copydoc dg::jacobi_zeros(int, double, double, T*)
template <typename T> void jacobi_zeros(int m, double alpha, double beta, T* z);
} // namespace optimized

} // namespace dg

/// \cond INTERNAL
/// Explicit instantiations of one implementation (namespace NS) for type T.
#define DG_INSTANTIATE_JACOBI_IMPL(NS, T)                                                  \
    template T    NS::jacobi<T>(int, double, double, T);                                   \
    template T    NS::jacobi_deriv<T>(int, double, double, T);                             \
    template void NS::jacobi_and_deriv<T>(int, double, double, T, T&, T&);                 \
    template void NS::jacobi<T>(int, double, double, const T*, std::size_t, T*);           \
    template void NS::jacobi_deriv<T>(int, double, double, const T*, std::size_t, T*);     \
    template void NS::jacobi_and_deriv<T>(int, double, double, const T*, std::size_t,      \
                                          detail::nondeduced_t<T>*,                        \
                                          detail::nondeduced_t<T>*);                       \
    template void NS::jacobi_all<T>(int, double, double, const T*, std::size_t, T*);       \
    template void NS::jacobi_zeros<T>(int, double, double, T*);
/// \endcond
