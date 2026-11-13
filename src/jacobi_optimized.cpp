/**
 * \file    jacobi_optimized.cpp
 * \brief   Optimised implementation of the Jacobi module: normalised
 *          recurrence, register blocking, fused multiply-add.
 *
 * \details Computes the same quantities as the reference implementation
 *          (jacobi_reference.cpp, read that one first), 3 to 9 times faster
 *          for many points (theory document \theorypdf, Section 2.6):
 *          - normalised recurrence (A.1.3'), coefficients tabulated once per call;
 *          - register blocking: the recurrence runs on blocks of 32 points held
 *            in SIMD registers;
 *          - fused multiply-add when the hardware has it;
 *          - in jacobi_zeros(): coefficient table once per call, SIMD deflation
 *            sum, certification in one array call.
 *
 *          The equations are those of Karniadakis & Sherwin, Appendix A.1, or
 *          (D.n) of the theory document. This implementation is the default
 *          (`DGHNOUTS_KERNELS=optimized`); test_impl.cpp checks it against the
 *          reference implementation.
 */
#include "jacobi_impl.hpp"

namespace dg {
namespace optimized {

using internal::check_args;
using internal::p1;
using internal::rec_coef;
using internal::RecCoef;

namespace {

/**
 * \brief Normalised coefficients of the recurrence: (A.1.3) divided by \f$a_n^1\f$,
 *        Eq. (A.1.3').
 *
 * \f[
 *   P_{n+1} = (A_n + B_n\xi)\,P_n - C_n\,P_{n-1},\qquad
 *   A_n = \frac{a_n^2}{a_n^1},\quad B_n = \frac{a_n^3}{a_n^1},\quad C_n = \frac{a_n^4}{a_n^1}.
 * \f]
 * One step then costs two multiply-adds and a multiplication per point, with
 * no division (see mad() for when the multiply-adds are fused).
 *
 * \tparam T floating-point type of the coefficients
 */
template <typename T>
struct RecABC {
    T A;  ///< \f$A_n = a_n^2/a_n^1\f$
    T B;  ///< \f$B_n = a_n^3/a_n^1\f$
    T C;  ///< \f$C_n = a_n^4/a_n^1\f$
};

/**
 * \brief  Normalised coefficients \f$A_n, B_n, C_n\f$ of (A.1.3') for a given \f$n\f$.
 *
 * Computed with three true divisions, so that each coefficient is rounded
 * once. (Multiplying by a rounded \f$1/a_n^1\f$ rounds twice; for integer
 * \f$\alpha,\beta\f$, where the book's \f$a_n^j\f$ are exact integers, that
 * made the worst case about twice less accurate than the reference
 * implementation where the recurrence cancels, e.g. \f$\alpha=2\f$, \f$n=24\f$,
 * \f$\xi\to1\f$; test_impl.cpp guards against it.) The divisions cost
 * nothing in the array routines and jacobi_zeros(), which tabulate the
 * coefficients once per call. Every routine obtains its coefficients from
 * this function, so the single-point and array versions give bitwise
 * identical results.
 *
 * \tparam    T      floating-point type
 * \param[in] n      index of (A.1.3), \f$n \ge 1\f$
 * \param[in] alpha  \f$\alpha\f$, already converted to `T`
 * \param[in] beta   \f$\beta\f$, already converted to `T`
 * \return    \f$A_n, B_n, C_n\f$
 *
 * \par Theory
 * \theorypdf, Section 2.2, Eq. (A.1.3').
 */
template <typename T>
inline RecABC<T> rec_abc(int n, T alpha, T beta)
{
    const RecCoef<T> c = rec_coef(n, alpha, beta);
    return { c.a2 / c.a1, c.a3 / c.a1, c.a4 / c.a1 };
}

/**
 * \brief Table of \f$A_k, B_k, C_k\f$, \f$k = 1,\dots,n-1\f$, computed once per call.
 *
 * The array routines and jacobi_zeros() reuse the same coefficients for many
 * points or many Newton iterations, so they are computed once instead of at
 * every step. For \f$n \le 128\f$ the table lives on the stack (no heap
 * allocation, which would cost as much as the work itself for small arrays);
 * larger \f$n\f$ use a heap buffer.
 *
 * \tparam T floating-point type
 */
template <typename T>
class RecTable {
public:
    /**
     * \brief Fill the table for the recurrence of degree \f$n\f$.
     * \param[in] n      target degree (entries \f$k = 1,\dots,n-1\f$ are filled)
     * \param[in] alpha  \f$\alpha\f$, converted to `T`
     * \param[in] beta   \f$\beta\f$, converted to `T`
     */
    RecTable(int n, T alpha, T beta)
    {
        const std::size_t len = static_cast<std::size_t>(n > 0 ? n : 1);
        T* base = stack_;
        if (len > kStack) { heap_.resize(3 * len); base = heap_.data(); }
        A_ = base; B_ = base + len; C_ = base + 2 * len;
        for (int k = 1; k < n; ++k) {
            const RecABC<T> c = rec_abc(k, alpha, beta);
            A_[k] = c.A; B_[k] = c.B; C_[k] = c.C;
        }
    }
    RecTable(const RecTable&)            = delete;   ///< not copyable: holds pointers into itself
    RecTable& operator=(const RecTable&) = delete;   ///< not copyable

    /// \return pointer to \f$A_k\f$, indexed by \f$k = 1,\dots,n-1\f$
    const T* A() const { return A_; }
    /// \return pointer to \f$B_k\f$, indexed by \f$k = 1,\dots,n-1\f$
    const T* B() const { return B_; }
    /// \return pointer to \f$C_k\f$, indexed by \f$k = 1,\dots,n-1\f$
    const T* C() const { return C_; }

private:
    static constexpr std::size_t kStack = 128;   ///< largest n kept on the stack
    T              stack_[3 * kStack];           ///< stack storage for n <= kStack
    std::vector<T> heap_;                        ///< heap storage for n > kStack
    T*             A_;                           ///< start of the A_k
    T*             B_;                           ///< start of the B_k
    T*             C_;                           ///< start of the C_k
};

/**
 * \brief  Multiply-add \f$xy+z\f$, fused (one rounding) when the hardware has FMA.
 *
 * A fused multiply-add halves the latency of a recurrence step (one 4-cycle
 * operation instead of a multiplication followed by an addition), which
 * matters for the single-point evaluations inside the Newton iteration of
 * jacobi_zeros(). `std::fma` is used only when `FP_FAST_FMA` (double) or
 * `FP_FAST_FMAF` (float) says the hardware does it in one instruction, e.g.
 * with `-march=native` (`DGPOLY_NATIVE=ON`) on any recent x86 or Arm CPU;
 * otherwise it would call a slow software routine, and `x*y + z` is used.
 * All routines use this helper in the same way, so single-point and array
 * results stay bitwise identical on a given machine and build.
 *
 * \tparam    T  floating-point type (`long double` never uses FMA)
 * \param[in] x  first factor
 * \param[in] y  second factor
 * \param[in] z  addend
 * \return    \f$xy+z\f$
 */
template <typename T>
inline T mad(T x, T y, T z)
{
    if constexpr (std::is_same_v<T, double>) {
#ifdef FP_FAST_FMA
        return std::fma(x, y, z);
#endif
    } else if constexpr (std::is_same_v<T, float>) {
#ifdef FP_FAST_FMAF
        return std::fma(x, y, z);
#endif
    }
    return x * y + z;
}

// ---------------------------------------------------------------------------
// Register blocking
//
// A recurrence step for one point depends on the previous step: a latency
// chain of 4 cycles with FMA, 8 without (multiplication, then subtraction).
// A core can start two such operations per cycle, so it needs about 8
// independent SIMD registers in flight to stay busy. The kernels below take
// kBlock points at once and run the whole recurrence on them: with 32
// doubles (8 AVX2 registers) the chains of different points overlap, and
// P_k, P_{k-1} never leave the registers. Memory traffic falls to one read
// of xi and one write of the result per point. Measured with bench_jacobi:
// 8 and 16 points per block are latency-bound and about 2.5x slower.
// ---------------------------------------------------------------------------

/// Number of points per register block (measured optimum, see bench_jacobi).
constexpr int kBlock = 32;

/**
 * \brief  Values \f$s\,P_n(\xi_i)\f$ for one block of points, recurrence in registers.
 *
 * \tparam    T      floating-point type
 * \tparam    Len    `int` for a tail block, or `std::integral_constant<int, kBlock>`
 *                   for a full block, whose constant trip count lets the compiler
 *                   unroll and vectorise the point loop
 * \param[in]  n      degree, \f$n \ge 1\f$
 * \param[in]  t      coefficient table for degree \f$n\f$
 * \param[in]  a      \f$\alpha\f$
 * \param[in]  b      \f$\beta\f$
 * \param[in]  xi     the points of the block
 * \param[in]  m      number of points in the block, \f$\le\f$ kBlock
 * \param[in]  scale  factor applied to the result (1 for values; the factor of
 *                    (A.1.8) for derivatives)
 * \param[out] out    \f$s\,P_n(\xi_i)\f$
 */
template <typename T, typename Len>
inline void values_block(int n, const RecTable<T>& t, T a, T b,
                         const T* xi, Len m, T scale, T* out)
{
    T x[kBlock], pk[kBlock], pm[kBlock];
    for (int i = 0; i < m; ++i) { x[i] = xi[i]; pm[i] = T(1); pk[i] = p1(a, b, x[i]); }
    const T* A = t.A(); const T* B = t.B(); const T* C = t.C();
    for (int k = 1; k < n; ++k) {                                  // (A.1.3') with the book's n = k
        const T Ak = A[k], Bk = B[k], Ck = C[k];
        for (int i = 0; i < m; ++i) {
            const T q = mad(mad(Bk, x[i], Ak), pk[i], -(Ck * pm[i]));
            pm[i] = pk[i];
            pk[i] = q;
        }
    }
    for (int i = 0; i < m; ++i) out[i] = scale * pk[i];
}

/**
 * \brief  Values and derivatives for one block of points, (A.1.3') and (D.2').
 * \tparam     T    floating-point type
 * \tparam     Len  see values_block()
 * \param[in]  n    degree, \f$n \ge 1\f$
 * \param[in]  t    coefficient table for degree \f$n\f$
 * \param[in]  a    \f$\alpha\f$
 * \param[in]  b    \f$\beta\f$
 * \param[in]  xi   the points of the block
 * \param[in]  m    number of points in the block, \f$\le\f$ kBlock
 * \param[out] p    values, or `nullptr`
 * \param[out] dp   derivatives, or `nullptr`
 */
template <typename T, typename Len>
inline void values_derivs_block(int n, const RecTable<T>& t, T a, T b,
                                const T* xi, Len m, T* p, T* dp)
{
    T x[kBlock], pk[kBlock], pm[kBlock], dk[kBlock], dm[kBlock];
    const T d1 = T(0.5) * (a + b + T(2));                          // (D.3): P_1'
    for (int i = 0; i < m; ++i) {
        x[i]  = xi[i];
        pm[i] = T(1);           dm[i] = T(0);                      // P_0, P_0'
        pk[i] = p1(a, b, x[i]); dk[i] = d1;                        // P_1, P_1'
    }
    const T* A = t.A(); const T* B = t.B(); const T* C = t.C();
    for (int k = 1; k < n; ++k) {
        const T Ak = A[k], Bk = B[k], Ck = C[k];
        for (int i = 0; i < m; ++i) {
            const T axi = mad(Bk, x[i], Ak);
            const T q   = mad(axi, pk[i], -(Ck * pm[i]));
            const T dq  = mad(axi, dk[i], mad(Bk, pk[i], -(Ck * dm[i])));
            pm[i] = pk[i];  pk[i] = q;
            dm[i] = dk[i];  dk[i] = dq;
        }
    }
    if (p)  for (int i = 0; i < m; ++i) p[i]  = pk[i];
    if (dp) for (int i = 0; i < m; ++i) dp[i] = dk[i];
}

/**
 * \brief  All degrees \f$P_0,\dots,P_N\f$ for one block of points: rows of \f$\mathbf{V}\f$, (D.4).
 * \tparam     T     floating-point type
 * \tparam     Len   see values_block()
 * \param[in]  nmax  highest degree \f$N \ge 1\f$
 * \param[in]  t     coefficient table for degree \f$N\f$
 * \param[in]  a     \f$\alpha\f$
 * \param[in]  b     \f$\beta\f$
 * \param[in]  xi    the points of the block
 * \param[in]  m     number of points in the block, \f$\le\f$ kBlock
 * \param[out] V     first row of the block in \f$\mathbf{V}\f$ (row-major, `nmax+1` columns)
 */
template <typename T, typename Len>
inline void all_block(int nmax, const RecTable<T>& t, T a, T b,
                      const T* xi, Len m, T* V)
{
    const std::size_t ld = static_cast<std::size_t>(nmax) + 1;
    T x[kBlock], pk[kBlock], pm[kBlock];
    for (int i = 0; i < m; ++i) {
        x[i] = xi[i]; pm[i] = T(1); pk[i] = p1(a, b, x[i]);
        V[i * ld] = pm[i];  V[i * ld + 1] = pk[i];
    }
    const T* A = t.A(); const T* B = t.B(); const T* C = t.C();
    for (int k = 1; k < nmax; ++k) {
        const T Ak = A[k], Bk = B[k], Ck = C[k];
        for (int i = 0; i < m; ++i) {
            const T q = mad(mad(Bk, x[i], Ak), pk[i], -(Ck * pm[i]));
            pm[i] = pk[i];
            pk[i] = q;
            V[i * ld + k + 1] = q;     // the block's rows (m x ld values) stay in L1
        }
    }
}

/**
 * \brief  Apply a block kernel to all points: full blocks of kBlock points
 *         (constant trip count), then one tail block.
 * \param[in] np      number of points
 * \param[in] kernel  callable `(i0, len)` where `len` is
 *                    `std::integral_constant<int, kBlock>` or an `int`
 */
template <typename Kernel>
inline void for_each_block(std::size_t np, Kernel&& kernel)
{
    std::size_t i0 = 0;
    for (; i0 + kBlock <= np; i0 += kBlock) kernel(i0, std::integral_constant<int, kBlock>{});
    if (i0 < np) kernel(i0, static_cast<int>(np - i0));
}

/**
 * \brief  Value and derivative at one point with a precomputed table: the
 *         evaluation used by every Newton step of jacobi_zeros().
 *
 * Same operations as values_derivs_block() and the single-point
 * jacobi_and_deriv(), so all three agree bitwise.
 *
 * \tparam     T   floating-point type
 * \param[in]  n   degree, \f$n \ge 1\f$
 * \param[in]  t   coefficient table for degree \f$n\f$
 * \param[in]  a   \f$\alpha\f$
 * \param[in]  b   \f$\beta\f$
 * \param[in]  xi  evaluation point
 * \param[out] p   \f$P_n(\xi)\f$
 * \param[out] dp  \f$P_n'(\xi)\f$
 */
template <typename T>
inline void value_deriv_table(int n, const RecTable<T>& t, T a, T b, T xi, T& p, T& dp)
{
    T pm = T(1), dm = T(0), pk = p1(a, b, xi), dk = T(0.5) * (a + b + T(2));
    const T* A = t.A(); const T* B = t.B(); const T* C = t.C();
    for (int k = 1; k < n; ++k) {
        const T axi = mad(B[k], xi, A[k]);
        const T q   = mad(axi, pk, -(C[k] * pm));
        const T dq  = mad(axi, dk, mad(B[k], pk, -(C[k] * dm)));
        pm = pk;  pk = q;
        dm = dk;  dk = dq;
    }
    p = pk; dp = dk;
}

/**
 * \brief  Deflation sum \f$s = \sum_{j<k} 1/(r-\xi_j)\f$ of the Newton step (D.7).
 *
 * The \f$k\f$ divisions are independent: eight partial sums let the compiler
 * use SIMD divisions, instead of one long chain of dependent additions.
 *
 * \tparam    T  floating-point type
 * \param[in] r  current Newton iterate
 * \param[in] z  zeros already found
 * \param[in] k  number of zeros already found
 * \return    \f$s\f$
 */
template <typename T>
inline T deflation_sum(T r, const T* z, int k)
{
    constexpr int L = 8;
    T acc[L] = {};
    int j = 0;
    for (; j + L <= k; j += L)
        for (int l = 0; l < L; ++l) acc[l] += T(1) / (r - z[j + l]);
    T s = T(0);
    for (; j < k; ++j) s += T(1) / (r - z[j]);
    for (int l = 0; l < L; ++l) s += acc[l];
    return s;
}

} // namespace

// ===========================================================================
// Single point
//
// The coefficients are computed on the fly with rec_abc(): the divisions are
// outside the dependency chain of the recurrence (the book's form divides
// the new value by a1, which puts the division's ~14-cycle latency on the
// chain). Same operations as the block kernels, hence bitwise identical.
// ===========================================================================
template <typename T>
T jacobi(int n, double alpha, double beta, T xi)
{
    check_args(n, alpha, beta);
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);

    // Starting values of (A.1.3)
    T pm = T(1);                                         // P_0
    if (n == 0) return pm;
    T pk = p1(a, b, xi);                                 // P_1

    // (A.1.3') with the book's n = k: P_{k+1} from P_k and P_{k-1}, k = 1 .. n-1
    for (int k = 1; k < n; ++k) {
        const RecABC<T> c = rec_abc(k, a, b);
        const T q = mad(mad(c.B, xi, c.A), pk, -(c.C * pm));
        pm = pk;
        pk = q;
    }
    return pk;                                           // P_n
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
    T pm = T(1), dm = T(0);                              // P_0, P_0'
    T pk = p1(a, b, xi), dk = T(0.5) * (a + b + T(2));   // P_1, P_1'

    // Book's n = k, k = 1 .. n-1. Same operations, in the same order, as the
    // array version, so the results are bitwise identical.
    for (int k = 1; k < n; ++k) {
        const RecABC<T> c = rec_abc(k, a, b);
        const T axi = mad(c.B, xi, c.A);
        const T q   = mad(axi, pk, -(c.C * pm));
        const T dq  = mad(axi, dk, mad(c.B, pk, -(c.C * dm)));
        pm = pk;  pk = q;
        dm = dk;  dk = dq;
    }
    p  = pk;
    dp = dk;
}

// ===========================================================================
// Arrays of points: register-blocked (see values_block and kBlock above)
// ===========================================================================
template <typename T>
void jacobi(int n, double alpha, double beta,
            const T* xi, std::size_t np, T* p)
{
    check_args(n, alpha, beta);
    if (np == 0) return;
    if (n == 0) {
        for (std::size_t i = 0; i < np; ++i) p[i] = T(1);
        return;
    }
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);
    const RecTable<T> t(n, a, b);                        // A_k, B_k, C_k: once per call
    for_each_block(np, [&](std::size_t i0, auto m) {
        values_block(n, t, a, b, xi + i0, m, T(1), p + i0);
    });
}

template <typename T>
void jacobi_deriv(int n, double alpha, double beta,
                  const T* xi, std::size_t np, T* dp)
{
    check_args(n, alpha, beta);
    if (np == 0) return;
    if (n == 0) {
        for (std::size_t i = 0; i < np; ++i) dp[i] = T(0);
        return;
    }
    // (A.1.8): recurrence of degree n-1 with (a+1, b+1), scaled inside the
    // block kernel (no second pass over dp)
    const T a1 = static_cast<T>(alpha + 1.0), b1 = static_cast<T>(beta + 1.0);
    const T s  = T(0.5) * (T(n) + static_cast<T>(alpha) + static_cast<T>(beta) + T(1));
    if (n == 1) {                                        // P_0^{a+1,b+1} = 1
        for (std::size_t i = 0; i < np; ++i) dp[i] = s * T(1);
        return;
    }
    const RecTable<T> t(n - 1, a1, b1);
    for_each_block(np, [&](std::size_t i0, auto m) {
        values_block(n - 1, t, a1, b1, xi + i0, m, s, dp + i0);
    });
}

template <typename T>
void jacobi_and_deriv(int n, double alpha, double beta,
                      const T* xi, std::size_t np,
                      detail::nondeduced_t<T>* p, detail::nondeduced_t<T>* dp)
{
    check_args(n, alpha, beta);
    if (np == 0) return;
    if (n == 0) {
        if (p)  for (std::size_t i = 0; i < np; ++i) p[i]  = T(1);
        if (dp) for (std::size_t i = 0; i < np; ++i) dp[i] = T(0);
        return;
    }
    // Values from (A.1.3') and derivatives from its xi-derivative (D.2'), in
    // the same register block: four recurrence variables per point, no arrays
    // in memory (this replaces the earlier L1 cache blocking of 256 points).
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);
    const RecTable<T> t(n, a, b);
    for_each_block(np, [&](std::size_t i0, auto m) {
        values_derivs_block(n, t, a, b, xi + i0, m, p ? p + i0 : nullptr, dp ? dp + i0 : nullptr);
    });
}

template <typename T>
void jacobi_all(int nmax, double alpha, double beta,
                const T* xi, std::size_t np, T* V)
{
    check_args(nmax, alpha, beta);
    if (np == 0) return;
    // (D.4): V[i*ld + k] = P_k(xi_i), row-major, ld = nmax + 1 columns per row
    if (nmax == 0) {
        for (std::size_t i = 0; i < np; ++i) V[i] = T(1);
        return;
    }
    // Register blocks of points: each block fills its own rows of V, so every
    // row is written while it is in L1 (the earlier version wrote V column by
    // column, touching every row at every degree: 7x slower on 10^6 points).
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);
    const RecTable<T> t(nmax, a, b);
    const std::size_t ld = static_cast<std::size_t>(nmax) + 1;
    for_each_block(np, [&](std::size_t i0, auto m) {
        all_block(nmax, t, a, b, xi + i0, m, V + i0 * ld);
    });
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

    // The degree and parameters are the same for every Newton step of every
    // zero: compute the recurrence coefficients A_k, B_k, C_k once.
    const T a = static_cast<T>(alpha), b = static_cast<T>(beta);
    const RecTable<T> t(m, a, b);

    for (int k = 0; k < m; ++k) {
        // (D.8): Chebyshev zero  -cos((2k+1) pi / 2m), averaged with the
        // previous root (book, B.2: "a better approximation for the initial guess")
        T r = -std::cos(T(2 * k + 1) * dth);
        if (k > 0) r = T(0.5) * (r + z[k - 1]);

        bool converged = false;
        for (int it = 0; it < maxit; ++it) {
            T p, dp;
            value_deriv_table(m, t, a, b, r, p, dp);     // (A.1.3') + (D.2'), one pass

            // Deflation: s = sum_{j<k} 1/(r - xi_j) divides out the known roots
            const T s = deflation_sum(r, z, k);

            // (D.7): Newton step for f_{m-k} = P_m / prod_{j<k} (xi - xi_j)
            const T delta = -p / (dp - p * s);
            r += delta;

            // (D.9): |delta| <= 4 eps (the book's test, with the absolute value)
            if (std::fabs(delta) <= tol) { converged = true; break; }
        }
        if (!converged || !std::isfinite(r) || std::fabs(r) >= T(1))
            internal::throw_newton_failure<T>(k, m);
        z[k] = r;
    }

    internal::sort_zeros(m, z);                       // increasing, distinct
    if (alpha == beta) internal::symmetrize_zeros(m, z);   // exact symmetry, (A.1.6)

    // (D.11) Certification, with the m+1 cut points evaluated in one call of
    // the register-blocked array routine
    const std::vector<T> c = internal::cut_points(m, z);
    std::vector<T> pc(c.size());
    jacobi(m, alpha, beta, c.data(), c.size(), pc.data());
    internal::certify_zeros(m, pc.data());
}

// ===========================================================================
// Explicit instantiations: float, double, long double
// ===========================================================================
} // namespace optimized

/// \cond INTERNAL
DG_INSTANTIATE_JACOBI_IMPL(optimized, float)
DG_INSTANTIATE_JACOBI_IMPL(optimized, double)
DG_INSTANTIATE_JACOBI_IMPL(optimized, long double)
/// \endcond

} // namespace dg
