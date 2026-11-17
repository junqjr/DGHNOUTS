Optimisations of the Jacobi module {#optimizations}
==================================

This page explains the computing techniques used in jacobi_optimized.cpp, why
each one was chosen, which alternatives were tried, and what each one gains.
The reference implementation (jacobi_reference.cpp) computes the same
quantities with the book's equations as written, one point at a time; read it
first. The mathematics is the same in both; only the order and the form of the
operations differ.

All the numbers on this page come from `bench_choices` (bench_choices.cpp),
which times each technique against its alternatives, and `bench_jacobi`
(bench_jacobi.cpp). They were measured on an Intel Xeon with AVX-512 and FMA
(GCC 13, `-O3 -march=native`, i.e. `-DDGPOLY_NATIVE=ON`), in `double`, for
\f$n=16\f$ and \f$\alpha=\beta=0\f$, on \f$n_p=4096\f$ points (fits in cache)
and \f$n_p=2^{20}\f$ points (does not). The machine is shared: differences
below about 20 % are noise. Run the two programs on your own machine before
relying on a number:

```bash
cmake --preset gcc -DDGPOLY_NATIVE=ON
cmake --build --preset gcc --target bench_choices bench_jacobi
./build/gcc/bench_choices      # optional argument: the degree n
./build/gcc/bench_jacobi
```

[TOC]

Where the time goes {#opt-cost}
===================

Every routine of the module evaluates the three-term recurrence (A.1.3),
\f[
  a_k^1\,P_{k+1}^{\alpha,\beta}(\xi)
  = \bigl(a_k^2 + a_k^3\,\xi\bigr)\,P_k^{\alpha,\beta}(\xi) - a_k^4\,P_{k-1}^{\alpha,\beta}(\xi),
  \qquad k = 1,\dots,n-1,
\f]
\f$n-1\f$ times per point. Two facts decide the performance.

1. **For one point, the recurrence is a dependency chain.** \f$P_{k+1}\f$
   needs \f$P_k\f$, which needs \f$P_{k-1}\f$: step \f$k+1\f$ cannot start
   before step \f$k\f$ has finished. Its duration is set by the *latency* of
   the operations on the chain: about 4 cycles for a multiply-add, 8 for a
   multiplication followed by an addition, and 13 to 20 for a division.
2. **Different points are independent.** The recurrences at \f$\xi_0\f$,
   \f$\xi_1\f$, … can run side by side. A core can start two multiply-adds
   per cycle, each on a SIMD register of 4 (AVX2) or 8 (AVX-512) doubles.

So a single point runs at the speed of the chain's latency and uses a tiny
fraction of the processor, while many points can keep every SIMD lane busy,
provided that the points are processed together and that the data stay close
to the processor. All the techniques below serve one of these two aims:
**shorten the chain** (no division on it, fused multiply-add) or **process
enough independent points together, without going back to memory**
(vectorisation across points, register blocking).

Summary {#opt-summary}
=======

| Technique | Where | Aim | Measured gain |
|---|---|---|---|
| \ref opt-loop-order "Vectorise across points" | array routines | use all SIMD lanes | ×6.5 (values, vs one point at a time) |
| \ref opt-normalised "Normalised recurrence (A.1.3')" | every routine | no division on the chain | ×1.4–1.7 in cache; none when memory-bound |
| \ref opt-table "Coefficient table" | array routines, zeros | compute \f$A_k,B_k,C_k\f$ once | ×6 per single-point evaluation in Newton |
| \ref opt-blocking "Register blocking, 32 points" | array routines | stay in registers, hide latency | ×2.4 in cache, ×5.5 out of cache (vs whole arrays) |
| \ref opt-fma "Fused multiply-add" | every routine | halve the latency of a step | ×2 together with wider SIMD (`-march=native`) |
| \ref opt-deriv "Scaling inside the kernel" | array `jacobi_deriv` | no second pass over the output | one pass instead of two |
| \ref opt-vandermonde "Row blocks of V" | `jacobi_all` | write \f$\mathbf{V}\f$ while it is in L1 | ×2 in cache, ×2.7 out of cache (vs columns) |
| \ref opt-zeros "Zeros: table, array certification" | `jacobi_zeros` | reuse work across Newton steps | ×3–3.6 per call |

Overall, against the reference implementation (\f$n=16\f$, nanoseconds per
point and per degree, from `bench_choices`):

| Routine | Reference | Optimised, `-march=native` | Optimised, portable (SSE2, no FMA) |
|---|---|---|---|
| `jacobi` (values) | 3.6–4.3 | 0.16–0.19 (×22) | 0.30–0.32 (×11) |
| `jacobi_deriv` (`bench_jacobi`) | 3.8–4.1 | 0.10–0.17 (×22–40) | |
| `jacobi_and_deriv` | 5.1–6.1 | 0.32–0.36 (×16) | 0.59–0.63 (×8) |
| `jacobi_all` | 3.5–4.2 | 0.68–0.72 in cache (×5.5), 2.1–2.4 out of cache (×1.7) | 0.63 / 2.4 |
| `jacobi_zeros`, Gauss–Lobatto, \f$Q=32\f$ | 67 µs | 17–20 µs (×3.6) | 18 µs |

The gains are about twice smaller without `-march=native`, which gives the
compiler only 2-wide SSE2 registers and no FMA. On a cluster, compile on a
compute node or name the target processor (see the README).


Vectorise across points, not across degrees {#opt-loop-order}
===========================================

**Problem.** The reference implementation loops over the points outside and
over the degrees inside: each point runs its own dependency chain, one after
the other, and the SIMD registers hold a single value.

**Choice.** Swap the loops: degrees outside, points inside,
```
for k = 1 .. n-1                       // degrees: sequential (dependency chain)
    for i = 0 .. np-1                  // points: independent -> SIMD
        P_{k+1}[i] = (A_k + B_k xi[i]) P_k[i] - C_k P_{k-1}[i]
```
The inner loop has no dependency between iterations, so the compiler turns it
into SIMD instructions, and the chains of different points overlap.

**Measured.** Values: 4.3 → 0.65 ns per point per degree, ×6.5, with the
book's form still dividing by \f$a_k^1\f$.

**Limit.** In this form the arrays \f$P_k\f$, \f$P_{k-1}\f$ of all the points
are read and written once per degree. For \f$2^{20}\f$ points (8 MB per array) they do
not fit in cache, and the loop is limited by memory bandwidth: 1.0 ns instead
of 0.65. Register blocking (below) removes this limit.

**GPU.** On a GPU the same choice is natural: one thread per point, the
recurrence in that thread's registers.


Normalised recurrence (A.1.3') {#opt-normalised}
==============================

**Problem.** The book's form computes
\f$P_{k+1} = [(a_k^2 + a_k^3\xi)P_k - a_k^4P_{k-1}]/a_k^1\f$. The division is
on the dependency chain: every step waits 13 to 20 cycles for it, and
divisions also have a low throughput.

**Choice.** Divide the coefficients, not the polynomials:
\f[
  P_{k+1}^{\alpha,\beta}(\xi)
  = \bigl(A_k + B_k\,\xi\bigr)\,P_k^{\alpha,\beta}(\xi) - C_k\,P_{k-1}^{\alpha,\beta}(\xi),
  \qquad
  A_k = \frac{a_k^2}{a_k^1},\quad B_k = \frac{a_k^3}{a_k^1},\quad C_k = \frac{a_k^4}{a_k^1}.
\f]
A step is then two multiply-adds and one multiplication, and only the last
multiply-add, the one that uses \f$P_k\f$, is on the chain:
\f$A_k+B_k\xi\f$ does not depend on the recurrence at all, and
\f$C_kP_{k-1}\f$ uses a value known one step earlier, so both are computed
while the previous step finishes. The coefficients do not depend on the point, so their
divisions are done once per degree, outside the loop over the points (or once
per call, see the next section). Code: `rec_abc()`, `RecABC`.

**Accuracy choice: three true divisions.** \f$A_k, B_k, C_k\f$ are computed as
`a2/a1`, `a3/a1`, `a4/a1`, each rounded once. Computing \f$r = 1/a_k^1\f$
first and then `a2*r`, `a3*r`, `a4*r` saves two divisions but rounds each
coefficient twice. For integer \f$\alpha,\beta\f$ the book's \f$a_k^j\f$ are
exact integers, so the reference implementation has no coefficient rounding
at all, and the double rounding made the optimised version about twice less
accurate where the recurrence cancels (e.g. \f$\alpha=2\f$, \f$n=24\f$,
\f$\xi\to1\f$). The divisions cost nothing measurable, since they are outside
the loop over the points; test_impl.cpp guards against the regression.

**Measured.** Degree-outer loop, in cache: 0.63–0.66 → 0.38–0.46 ns,
×1.4–1.7. Out of cache: no gain (memory-bound). Single point, coefficients
computed at every step: 60–65 → 56–58 ns, small, because there the
divisions of the coefficients are still done at every step (they are off the
chain, but they still occupy the divider).


Coefficient table {#opt-table}
=================

**Problem.** \f$A_k, B_k, C_k\f$ cost three divisions and about fifteen other
operations per degree. A routine that evaluates the same degree many times
(an array of points, or the Newton iterations of `jacobi_zeros`) would
recompute them every time.

**Choice.** `RecTable` computes \f$A_k, B_k, C_k\f$, \f$k=1,\dots,n-1\f$, once
per call; every block of points or every Newton step then reads them. For
\f$n\le128\f$ the table is an array on the stack (3 × 128 values); larger
\f$n\f$ use a `std::vector`.

**Why the stack.** A heap allocation and release costs about 13 ns, as much as
evaluating a single point with the table (10 ns at \f$n=16\f$), and the array
routines are often called on a handful of points. Degrees above 128 are far
beyond DG use, and the heap cost is then negligible.

**Why the single-point routines do not use it.** `jacobi(n, alpha, beta, xi)`
evaluates the degree once: filling a table costs the same divisions as
computing the coefficients on the fly, plus the stores. The table pays off
only when it is reused, which a single call cannot do.

**Measured.** Single-point evaluation, \f$n=16\f$: coefficients on the fly
56–58 ns, from the table 10 ns, ×6. This is the main gain of `jacobi_zeros`,
which evaluates \f$P_m\f$ and \f$P_m'\f$ at one point per Newton step.


Register blocking {#opt-blocking}
=================

**Problem.** The degree-outer loop vectorises, but it sweeps the arrays
\f$P_k\f$, \f$P_{k-1}\f$ of all the points once per degree, and is limited by
memory as soon as they leave the cache.

**Choice.** Process the points in blocks of `kBlock` = 32. For each block, the
whole recurrence runs with \f$\xi_i\f$, \f$P_k(\xi_i)\f$ and
\f$P_{k-1}(\xi_i)\f$ held in local arrays of 32 values, which the compiler
keeps in SIMD registers:
```
for each block of 32 points                 // xi, P_k, P_{k-1} in registers
    for k = 1 .. n-1                         // degree: sequential
        for i in block                       // points: independent -> SIMD
            q = (A_k + B_k xi[i]) P_k[i] - C_k P_{k-1}[i]
            P_{k-1}[i] = P_k[i];  P_k[i] = q
    write P_n for the block
```
Memory traffic falls to one read of \f$\xi_i\f$ and one write of the result
per point, whatever \f$n\f$. Code: `values_block()`,
`values_derivs_block()`, `all_block()`, `for_each_block()`.

**Why 32 points.** A block must contain enough independent chains to hide the
latency of a step: with a multiply-add latency of 4 cycles and two of them
started per cycle, about 8 SIMD operations must be in flight. 32 doubles are
8 AVX2 registers or 4 AVX-512 registers per recurrence variable.

| Points per block | 8 | 16 | **32** | 64 |
|---|---|---|---|---|
| `jacobi`, in cache (ns) | 0.46–0.53 | 0.39–0.47 | **0.20** | 0.21 |
| `jacobi`, out of cache (ns) | 0.46–0.55 | 0.44–0.48 | **0.17–0.21** | 0.20–0.22 |

8 and 16 points are latency-bound, about 2.5 times slower. 64 brings no
further gain and doubles the registers needed; `jacobi_and_deriv` keeps four
variables per point (\f$P_k, P_{k-1}\f$ and their derivatives), so 64 points
would need 32 AVX2 registers, more than exist, and the compiler would spill
them to memory. 32 is the smallest block on the plateau.

**Full blocks with a constant size.** `for_each_block()` calls the kernel with
`std::integral_constant<int, 32>` for the full blocks, so the compiler knows
the trip count of the inner loop at compile time and unrolls it completely;
only the last, partial block uses a run-time size.

**Alternative: L1-cache blocking.** Blocks of 256 points kept in small arrays
in the L1 cache, with the degree loop outside, are as fast on this CPU:

| | Whole arrays | L1 blocks of 256 | Register blocks of 32 |
|---|---|---|---|
| `jacobi`, in cache / out of cache (ns) | 0.38–0.46 / 0.95–1.07 | 0.15–0.18 / 0.18–0.20 | 0.16–0.19 / 0.17–0.19 |
| `jacobi_and_deriv`, in / out of cache (ns) | 1.5 / 2.0–2.1 | 0.25–0.33 / 0.37–0.39 | 0.35–0.36 / 0.32 |

Both remove the memory limit, and the differences between them are within
the noise of the machine. Register blocking was kept because it needs no
scratch arrays (the four arrays of 256 values of `jacobi_and_deriv` are 8 KB of
stack per call), because its fixed, small block lets the compiler unroll the
point loop, and because it has the shape of a GPU kernel: one point per
thread (or SIMD lane), the whole recurrence in registers. (An earlier version
of this documentation reported register blocking 3 to 4 times faster than an
L1-blocked version of `jacobi_and_deriv`; the measurements with
`bench_choices` do not reproduce that gain against a clean L1-blocked
version.)


Fused multiply-add {#opt-fma}
==================

**Problem.** In a step \f$(A_k + B_k\xi)P_k - C_kP_{k-1}\f$, the operation
on the chain is the multiplication by \f$P_k\f$ followed by the subtraction
(see \ref opt-normalised "above"). Written separately, they cost about 8
cycles of latency per step.

**Choice.** The helper `mad(x, y, z)` returns \f$xy+z\f$ computed by
`std::fma`, one instruction of about 4 cycles with a single rounding, when the
hardware has it; otherwise it returns `x*y + z`. A step is then
\f[
  q = \operatorname{mad}\bigl(\operatorname{mad}(B_k,\,\xi,\,A_k),\ P_k,\ -C_k\,P_{k-1}\bigr).
\f]
`std::fma` is used only if the compiler defines `FP_FAST_FMA` (`double`) or
`FP_FAST_FMAF` (`float`), i.e. when it generates one instruction, which is the
case with `-march=native` (`-DDGPOLY_NATIVE=ON`) on any recent x86 or Arm
processor. Without hardware support, `std::fma` is a slow library call that
computes the exact result in software; it is then avoided. `long double` never
uses it (x87 has no FMA).

**Bitwise reproducibility.** Every routine computes a step with `mad()` in the
same order, so the single-point routines, the array routines and the Newton
iteration give bitwise identical results on a given machine and build
(checked by test_impl.cpp). This is also why the library refuses
`-ffast-math`: the compiler would be free to reorder the operations
differently in each routine.

**Measured.** FMA is not measured alone: `-march=native` also widens the SIMD
registers (SSE2: 2 doubles, AVX-512: 8). Together: `jacobi` 0.30–0.32 →
0.16–0.19 ns, `jacobi_and_deriv` 0.59–0.63 → 0.32–0.36 ns, about ×2.


Derivatives: scaling inside the kernel {#opt-deriv}
======================================

`jacobi_deriv` uses (A.1.8),
\f[
  \frac{d}{d\xi}P_n^{\alpha,\beta}(\xi)
  = \frac12\,(n+\alpha+\beta+1)\,P_{n-1}^{\alpha+1,\beta+1}(\xi).
\f]
The array version runs `values_block()` with degree \f$n-1\f$ and parameters
\f$(\alpha+1,\beta+1)\f$ and multiplies by the factor
\f$s=\tfrac12(n+\alpha+\beta+1)\f$ when it writes each block's results
(argument `scale`), instead of calling the array `jacobi()` and then
multiplying the whole output array in a second pass, which would read and
write it again.

`jacobi_and_deriv` computes the value and the derivative in the same register
block, from (A.1.3') and its \f$\xi\f$-derivative (D.2'): the two recurrences
share \f$A_k + B_k\xi\f$, and one pass gives both results. Either output
pointer may be `nullptr`.


Vandermonde matrix: row blocks {#opt-vandermonde}
==============================

**Problem.** \f$\mathbf{V}_{ik}=P_k^{\alpha,\beta}(\xi_i)\f$ is stored row by
row (`V[i*(N+1) + k]`). Filled degree by degree (column by column), every
degree touches every row: for many points the whole matrix passes through the
cache \f$N\f$ times.

**Choice.** `all_block()` fills the \f$N+1\f$ values of the 32 rows of a block
together, while those rows are in L1, and moves on.

**Measured.** In cache: columns 1.4–1.5 ns, row blocks 0.68–0.72 ns, ×2. Out
of cache: columns 6.1 ns, row blocks 2.1–2.3 ns, ×2.7. The routine remains
the slowest per value because it must write its whole output, \f$N+1\f$
values per point (136 MB for \f$2^{20}\f$ points at \f$N=16\f$): out of
cache it is limited by the memory bandwidth, and the reference
implementation, which writes one row at a time, is only 1.7 times slower.


Zeros {#opt-zeros}
=====

`jacobi_zeros` is Newton's method with deflation (Appendix B.2). Each Newton
step evaluates \f$P_m\f$ and \f$P_m'\f$ at a single point: one dependency
chain of \f$m\f$ steps, which no vectorisation across points can help. The
gain therefore comes from doing less work per step, and from the part of the
algorithm that does have many points.

| Technique | Measured |
|---|---|
| Coefficient table, built once per call and read by every Newton step (`value_deriv_table()`) | single-point evaluation 56–58 → 10 ns |
| Certification (D.11): the \f$m+1\f$ cut points evaluated in one call of the register-blocked array `jacobi()` | \f$m=32\f$: 4.0 → 0.2–0.25 µs; \f$m=64\f$: 15–16 → 0.7–0.8 µs |
| Deflation sum \f$\sum_{j<k}1/(r-\xi_j)\f$ with eight partial sums (`deflation_sum()`) | no measurable gain (\f$k=64\f$: 38–40 ns either way) |
| Whole routine, Gauss–Lobatto nodes | \f$Q=16\f$: 13.4 → 4.3–4.5 µs; \f$Q=32\f$: 67 → 17–20 µs; \f$Q=64\f$: 307 → 84–88 µs |

**About the deflation sum.** The eight partial sums were meant to break the
chain of additions so that the compiler could use SIMD divisions. The
measurement shows no gain: the \f$k\f$ divisions themselves, not the
additions, set the time, and the divider is shared by all SIMD lanes. The
technique is harmless but does not pay for its complexity; it is a candidate
for simplification.

The reflection (D.10), the exact symmetrisation and the certification (D.11)
make the routine robust; they are not speed optimisations, and the reference
implementation has them too.

**In a solver** the zeros are computed once, at start-up: tens of
microseconds either way. The optimisations matter for the array routines,
when evaluating at many points (interpolation onto fine output grids, error
norms, plotting).


What was not done, and why {#opt-not-done}
==========================

- **Hand-written SIMD intrinsics** (AVX2, AVX-512, SVE). The compiler
  vectorises the point loops of the blocks by itself, and intrinsics would
  tie the code to one instruction set: Jean-Zay and Adastra hosts are x86,
  Rhea (Alice Recoque) is Arm.
- **Threads** (OpenMP). The routines run once at start-up on a few points, or
  are called from code that is itself parallel. Threads would add a dependency
  for no gain in the solver.
- **GPU kernels.** Planned with Kokkos in the solver, not in this
  preprocessing library. Register blocking already has the GPU shape: one point
  per thread, the recurrence in registers, \f$A_k,B_k,C_k\f$ shared by all
  threads.
- **Computing in `float`.** Single precision keeps only 4 to 5 correct digits
  after the recurrence; nodes and matrices for a single-precision solver
  should be computed in `double` and rounded (see \ref precision "Precision").
- **Relaxed floating-point rules** (`-ffast-math`, `-Ofast`). They would allow
  reassociation, break the bitwise agreement between routines, and delete the
  NaN checks of `jacobi_zeros`; the library refuses to compile with them.
