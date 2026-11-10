# DGHNOUTS

**D**iscontinuous **G**alerkin **H**igh-order **N**avier–Stokes **O**ptimized **U**nstructured **T**hreaded **S**olver

Author: Carlos Junqueira Junior (DynFluid, Arts et Métiers – ENSAM)

DGHNOUTS is a new high-order discontinuous Galerkin (DG) solver for the
three-dimensional compressible Navier–Stokes equations, aimed at LES and DNS on
GPU supercomputers. It is designed to run from a single source on NVIDIA and
AMD GPUs (Jean-Zay, Adastra, Alice Recoque) and on CPUs.

> **Status: early development.** The repository currently contains the first
> building block of the spectral/hp library: Jacobi polynomials. The solver
> itself does not exist yet. See the [roadmap](#roadmap).

---

## Contents

- [Roadmap](#roadmap)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Building](#building)
- [Testing](#testing)
- [Documentation](#documentation)
- [Choosing the compiler](#choosing-the-compiler)
- [The Jacobi module](#the-jacobi-module)
- [References](#references)

---

## Roadmap

**Spectral/hp library** (plain C++17, CPU; runs once at start-up)

- [x] Jacobi polynomials P<sub>n</sub><sup>(α,β)</sup>: values, derivatives, norms
- [ ] Zeros of Jacobi polynomials
- [ ] Gauss, Gauss–Radau and Gauss–Lobatto quadrature (Jacobi weights)
- [ ] Lagrange interpolation and differentiation matrices
- [ ] Vandermonde matrices (nodal ↔ modal), modal filters

**Solver** (C++17 + Kokkos; CUDA, HIP and OpenMP backends)

- [ ] Mesh and element data structures (hexahedra, curved elements)
- [ ] DGSEM operators: volume and surface terms, numerical fluxes
- [ ] Split-form / entropy-stable formulation for under-resolved turbulence
- [ ] Viscous terms
- [ ] Time integration (low-storage Runge–Kutta)
- [ ] MPI parallelism (GPU-aware halo exchange)
- [ ] I/O (HDF5)

## Repository layout

```
dghnouts/
├── CMakeLists.txt        build definition, compiler-specific flags
├── CMakePresets.json     one preset per compiler (gcc, nvhpc, intel, clang, gcc-debug)
├── docs/
│   ├── Doxyfile          Doxygen configuration (API reference)
│   ├── mainpage.md       front page of the API reference
│   └── theory/
│       └── dghnouts_theory.tex   mathematical documentation (LaTeX)
├── include/
│   └── jacobi.hpp        public interface of the Jacobi module
├── src/
│   └── jacobi.cpp        implementation
└── tests/
    └── test_jacobi.cpp   unit tests against the identities of Appendix A
```

## Requirements

- A C++17 compiler: GCC ≥ 9, Clang ≥ 10, NVIDIA HPC SDK `nvc++`, or Intel oneAPI `icpx`
- CMake ≥ 3.21

No other dependency for now. Kokkos, MPI and HDF5 will be added with the solver.

## Building

```bash
git clone git@github.com:junqjr/DGHNOUTS.git
cd DGHNOUTS

cmake --preset gcc            # configure (build directory: build/gcc)
cmake --build --preset gcc    # compile
```

Each preset builds in its own directory under `build/`, so several compilers and
configurations can coexist. `cmake --list-presets` shows all of them.

## Testing

```bash
ctest --preset gcc            # or run the test executable directly:
./build/gcc/test_jacobi
```

The tests check the Jacobi module against the identities of Appendix A.1 of
Karniadakis & Sherwin (end-point values, symmetry, the Sturm–Liouville equation,
derivatives at x = ±1), and the array routines against the single-point ones.

**Optional: comparison with Polylib.** If you have Polylib (`polylib.cpp` and
`polylib.h`, distributed with the book), point CMake at its folder to enable the
additional comparison and orthogonality tests:

```bash
cmake --preset gcc -DPOLYLIB_DIR=/path/to/polylib
```

Polylib is used only as a reference in the tests; it is never part of the library.

## Documentation

Two complementary documents:

- **Theory** (`docs/theory/dghnouts_theory.tex`): the mathematics of every
  routine, in the notation and equation numbering of Karniadakis & Sherwin.
- **API reference** (Doxygen): generated from the comments in the code. Each
  routine states the equation it implements and links to the theory PDF.

Both are built by one target:

```bash
cmake --preset gcc
cmake --build --preset gcc --target doc
# open build/gcc/doc/html/index.html
```

Requirements: Doxygen ≥ 1.9 and a LaTeX installation with `latexmk`, `dvips` and
Ghostscript (`gs`). Formulas are rendered offline, so the pages can be read
without an internet connection. Graphviz (`dot`) is optional and adds call graphs.

## Choosing the compiler

| Preset | Compiler | Notes |
|---|---|---|
| `gcc` | GNU `g++` | default for development |
| `clang` | LLVM `clang++` | |
| `nvhpc` | NVIDIA HPC SDK `nvc++` | load the `nvidia-compilers` module on Jean-Zay first |
| `intel` | Intel oneAPI `icpx` | |
| `gcc-debug` | GNU `g++` | `-g` with AddressSanitizer and UBSan |

Add `-DDGPOLY_NATIVE=ON` to optimise for the CPU of the build machine
(`-march=native`, `-xHost` or `-tp=native` depending on the compiler).

If you change the compiler of an existing build directory, delete it first: CMake
caches the compiler on the first configure.

## The Jacobi module

`include/jacobi.hpp`, namespace `dg`. Requires n ≥ 0 and α, β > −1.

| Function | Computes | Book equation |
|---|---|---|
| `jacobi(n, a, b, x)` | P<sub>n</sub><sup>(α,β)</sup>(x) | recurrence (A.1.3) |
| `jacobi_deriv(n, a, b, x)` | dP<sub>n</sub><sup>(α,β)</sup>/dx | (A.1.8), valid at x = ±1 |
| `jacobi(n, a, b, x, np, p)` | P<sub>n</sub> at np points | (A.1.3), vectorised over points |
| `jacobi_deriv(n, a, b, x, np, dp)` | P′<sub>n</sub> at np points | (A.1.8) |
| `jacobi_and_deriv(n, a, b, x, np, p, dp)` | P<sub>n</sub> and P′<sub>n</sub> in one pass | (A.1.3) and its x-derivative |
| `jacobi_all(nmax, a, b, x, np, V)` | P<sub>0</sub> … P<sub>nmax</sub> (Vandermonde matrix) | (A.1.3), each degree once |
| `jacobi_norm2(n, a, b)` | ‖P<sub>n</sub><sup>(α,β)</sup>‖² with the Jacobi weight | (A.1.7), via `std::lgamma` |

Array routines take raw pointers (`const double* x, std::size_t np`) so they work
with any contiguous storage. `std::vector` overloads are also provided.

Example:

```cpp
#include "jacobi.hpp"
#include <vector>

std::vector<double> x = {-1.0, -0.5, 0.0, 0.5, 1.0};
std::vector<double> p(x.size()), dp(x.size());

// Legendre polynomial L_4 and its derivative at all points, in one pass
dg::jacobi_and_deriv(4, 0.0, 0.0, x.data(), x.size(), p.data(), dp.data());
```

## References

- G. Karniadakis and S. Sherwin, *Spectral/hp Element Methods for Computational
  Fluid Dynamics*, 2nd ed., Oxford University Press, 2005. Equation numbers in
  the code refer to this book (Appendices A and B).
- D. A. Kopriva, *Implementing Spectral Methods for Partial Differential
  Equations*, Springer, 2009.
- [Kokkos](https://kokkos.org): performance-portable programming model planned
  for the solver kernels.
