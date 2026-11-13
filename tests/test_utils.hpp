/**
 * \file    test_utils.hpp
 * \brief   Small helpers shared by the test programs: PASS/FAIL reporting and
 *          relative errors.
 *
 * \details Each test program calls report() for every check and returns
 *          summary(), which is 0 if all checks passed and 1 otherwise, so
 *          that `ctest` reports failures.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>

/// Helpers for the DGHNOUTS test programs.
namespace dgtest {

inline int n_pass = 0;   ///< number of passed checks
inline int n_fail = 0;   ///< number of failed checks

/**
 * \brief  Relative error with a floor of 1, so values near zero are compared absolutely.
 * \param[in] a    computed value
 * \param[in] ref  reference value
 * \return    \f$|a-\mathrm{ref}| / \max(1,|\mathrm{ref}|)\f$
 */
inline long double rel_err(long double a, long double ref)
{
    return std::fabs(a - ref) / std::max(1.0L, std::fabs(ref));
}

/**
 * \brief  Print one PASS/FAIL line and update the counters.
 * \param[in] name     description of the check
 * \param[in] max_err  largest error found
 * \param[in] tol      tolerance; the check passes if max_err <= tol
 */
inline void report(const char* name, long double max_err, double tol)
{
    const bool ok = max_err <= tol;
    std::printf("  [%s] %-58s max err = %.2Le (tol %.0e)\n", ok ? "PASS" : "FAIL", name, max_err, tol);
    ok ? ++n_pass : ++n_fail;
}

/**
 * \brief  Print one PASS/FAIL line for a yes/no check and update the counters.
 * \param[in] name  description of the check
 * \param[in] ok    result of the check
 */
inline void report_bool(const char* name, bool ok)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    ok ? ++n_pass : ++n_fail;
}

/**
 * \brief  Print the totals.
 * \return 0 if every check passed, 1 otherwise (the program's exit code).
 */
inline int summary()
{
    std::printf("\n%d passed, %d failed\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}

/// A pair of Jacobi weight parameters.
struct Params {
    double a;   ///< \f$\alpha\f$
    double b;   ///< \f$\beta\f$
};

} // namespace dgtest
