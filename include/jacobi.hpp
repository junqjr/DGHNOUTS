// jacobi.hpp - Jacobi polynomials P_n^{alpha,beta}(x) on [-1,1]
//
// Reference: Karniadakis & Sherwin, Spectral/hp Element Methods for CFD,
//            2nd ed., Appendix A.1
//
//   values      : three-term recurrence                              (A.1.3)
//   derivatives : dP_n^{a,b}/dx = 1/2 (n+a+b+1) P_{n-1}^{a+1,b+1}      (A.1.8)
//                 or the x-derivative of (A.1.3) itself (jacobi_and_deriv)
//                 -> both valid on the closed interval, including x = +/-1
//                    (unlike (A.1.4), which divides by 1 - x^2)
//   norm        : int (1-x)^a (1+x)^b [P_n^{a,b}]^2 dx                   (A.1.7)
//
// Requirements: n >= 0, alpha > -1, beta > -1.
//
// Array routines take raw pointers  (x, np)  so they work on any contiguous
// storage: std::vector (.data()), C arrays, Kokkos host views, ...
// Their loops run over the degree k outside and over the points inside:
// the inner loop has no dependency and is vectorised by the compiler.
#pragma once

#include <cstddef>
#include <vector>

namespace dg {

// ---- single point --------------------------------------------------------

// P_n^{a,b}(x)
double jacobi(int n, double alpha, double beta, double x);

// dP_n^{a,b}/dx   via (A.1.8)
double jacobi_deriv(int n, double alpha, double beta, double x);

// ---- np points: x[0..np-1] -> out[0..np-1] -------------------------------

// p[i] = P_n^{a,b}(x[i])
void jacobi(int n, double alpha, double beta,
            const double* x, std::size_t np, double* p);

// dp[i] = dP_n^{a,b}/dx (x[i])   via (A.1.8)
void jacobi_deriv(int n, double alpha, double beta,
                  const double* x, std::size_t np, double* dp);

// p[i] = P_n(x[i])  and  dp[i] = P_n'(x[i])  in ONE recurrence pass,
// using the x-derivative of (A.1.3):
//   a1 P'_{k+1} = a3 P_k + (a2 + a3 x) P'_k - a4 P'_{k-1}
// (p or dp may be nullptr if not needed)
void jacobi_and_deriv(int n, double alpha, double beta,
                      const double* x, std::size_t np, double* p, double* dp);

// All degrees 0..nmax in ONE recurrence pass (generalised Vandermonde matrix):
//   V[i*(nmax+1) + k] = P_k^{a,b}(x[i]),   i = 0..np-1,  k = 0..nmax
// (row-major: one row per point, one column per degree; V has np*(nmax+1) entries)
void jacobi_all(int nmax, double alpha, double beta,
                const double* x, std::size_t np, double* V);

// ---- std::vector convenience wrappers (out is resized) -------------------

void jacobi(int n, double alpha, double beta,
            const std::vector<double>& x, std::vector<double>& p);

void jacobi_deriv(int n, double alpha, double beta,
                  const std::vector<double>& x, std::vector<double>& dp);

// ---- norm ----------------------------------------------------------------

// h_n = int_{-1}^{1} (1-x)^a (1+x)^b [P_n^{a,b}(x)]^2 dx   (A.1.7)
double jacobi_norm2(int n, double alpha, double beta);

} // namespace dg
