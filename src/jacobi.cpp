/**
 * \file    jacobi.cpp
 * \brief   Public functions of the Jacobi module (jacobi.hpp): choice of the
 *          implementation, and the parts both implementations share.
 *
 * \details The routines exist in two implementations with the same results up
 *          to round-off (see jacobi_impl.hpp):
 *          - `dg::reference` (jacobi_reference.cpp): the book's equations,
 *            written to be read;
 *          - `dg::optimized` (jacobi_optimized.cpp): the same, 15 to 40 times
 *            faster for many points.
 *
 *          The public functions below forward to one of them. The choice is
 *          made here, at compile time, and only here:
 *
 *              cmake --preset gcc -DDGHNOUTS_KERNELS=reference   # or optimized (default)
 *
 *          which defines (or not) the macro `DGHNOUTS_REFERENCE_KERNELS`.
 *          The `std::vector` overloads and jacobi_norm2() have a single
 *          implementation, in this file.
 */
#include "jacobi_impl.hpp"

namespace dg {

// ===========================================================================
// Choice of the implementation: the only #ifdef of the module
// ===========================================================================
#ifdef DGHNOUTS_REFERENCE_KERNELS
namespace impl = reference;            ///< implementation used by the public functions
static constexpr const char* kKernels = "reference";
#else
namespace impl = optimized;            ///< implementation used by the public functions
static constexpr const char* kKernels = "optimized";
#endif

const char* jacobi_kernels() { return kKernels; }

// ===========================================================================
// Public functions: forward to the chosen implementation
// ===========================================================================
template <typename T>
T jacobi(int n, double alpha, double beta, T xi)
{ return impl::jacobi(n, alpha, beta, xi); }

template <typename T>
T jacobi_deriv(int n, double alpha, double beta, T xi)
{ return impl::jacobi_deriv(n, alpha, beta, xi); }

template <typename T>
void jacobi_and_deriv(int n, double alpha, double beta, T xi, T& p, T& dp)
{ impl::jacobi_and_deriv(n, alpha, beta, xi, p, dp); }

template <typename T>
void jacobi(int n, double alpha, double beta, const T* xi, std::size_t np, T* p)
{ impl::jacobi(n, alpha, beta, xi, np, p); }

template <typename T>
void jacobi_deriv(int n, double alpha, double beta, const T* xi, std::size_t np, T* dp)
{ impl::jacobi_deriv(n, alpha, beta, xi, np, dp); }

template <typename T>
void jacobi_and_deriv(int n, double alpha, double beta, const T* xi, std::size_t np,
                      detail::nondeduced_t<T>* p, detail::nondeduced_t<T>* dp)
{ impl::jacobi_and_deriv<T>(n, alpha, beta, xi, np, p, dp); }

template <typename T>
void jacobi_all(int nmax, double alpha, double beta, const T* xi, std::size_t np, T* V)
{ impl::jacobi_all(nmax, alpha, beta, xi, np, V); }

template <typename T>
void jacobi_zeros(int m, double alpha, double beta, T* z)
{ impl::jacobi_zeros(m, alpha, beta, z); }

// ===========================================================================
// std::vector overloads (one implementation, built on the pointer versions)
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

template <typename T>
std::vector<T> jacobi_zeros(int m, double alpha, double beta)
{
    internal::check_args(m, alpha, beta);                // before allocating
    std::vector<T> z(static_cast<std::size_t>(m));
    jacobi_zeros(m, alpha, beta, z.data());
    return z;
}

// ===========================================================================
// Norm (one implementation: n multiplications, nothing to optimise)
// ===========================================================================
template <typename T>
T jacobi_norm2(int n, double alpha, double beta)
{
    internal::check_args(n, alpha, beta);
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);
    const T apb = a + b;

    // (D.5): h_0 = 2^{a+b+1} Gamma(a+1) Gamma(b+1) / Gamma(a+b+2)
    // (A.1.7) is 0 * inf at n = 0 when a + b = -1 (Chebyshev); (D.5) is the
    // same value, using (2n+a+b+1) Gamma(n+a+b+1) = Gamma(a+b+2) at n = 0.
    T h;
    if (apb + T(2) < T(30)) {
        // Usual parameters: the Gamma functions themselves (all < 1e34, no
        // overflow even in float). The quotient is <= 1.
        h = std::exp2(apb + T(1))
          * (std::tgamma(a + T(1)) * std::tgamma(b + T(1)) / std::tgamma(apb + T(2)));
    } else {
        // Large parameters: Gamma would overflow, use logarithms
        h = std::exp((apb + T(1)) * std::log(T(2))
                     + std::lgamma(a + T(1)) + std::lgamma(b + T(1))
                     - std::lgamma(apb + T(2)));
    }
    if (n == 0) return h;

    // (D.5'): h_k from h_{k-1}, the quotient of (A.1.7) at k and k-1.
    //   h_1 / h_0     = (a+1)(b+1) / (a+b+3)
    //   h_k / h_{k-1} = (2k+a+b-1)(k+a)(k+b) / [(2k+a+b+1) k (k+a+b)],   k >= 2
    // Each factor is close to 1, so nothing cancels: the error grows like a
    // few rounding errors per step. (Evaluating (A.1.7) from logarithms
    // subtracts lgamma terms of size n ln n and loses that many digits:
    // 8e-15 at n = 20, 2e-13 at n = 300 in double.)
    h *= (a + T(1)) * (b + T(1)) / (apb + T(3));
    for (int k = 2; k <= n; ++k) {
        const T c = T(2) * k + apb;                             // 2k + a + b
        h *= ((c - T(1)) * (k + a) * (k + b)) / ((c + T(1)) * k * (k + apb));
    }
    return h;
}

// ===========================================================================
// Explicit instantiations: float, double, long double
// ===========================================================================
/// \cond INTERNAL
#define DG_INSTANTIATE_JACOBI(T)                                                       \
    template T    jacobi<T>(int, double, double, T);                                   \
    template T    jacobi_deriv<T>(int, double, double, T);                             \
    template void jacobi_and_deriv<T>(int, double, double, T, T&, T&);                 \
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
    template void jacobi_zeros<T>(int, double, double, T*);                            \
    template std::vector<T> jacobi_zeros<T>(int, double, double);

DG_INSTANTIATE_JACOBI(float)
DG_INSTANTIATE_JACOBI(double)
DG_INSTANTIATE_JACOBI(long double)

#undef DG_INSTANTIATE_JACOBI
/// \endcond

} // namespace dg
