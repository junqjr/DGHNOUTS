DGHNOUTS API documentation {#mainpage}
==========================

**D**iscontinuous **G**alerkin **H**igh-order **N**avier–Stokes **O**ptimized
**U**nstructured **T**hreaded **S**olver: a high-order discontinuous Galerkin
solver for the three-dimensional compressible Navier–Stokes equations, aimed at
LES and DNS on GPU supercomputers.

This is the API reference, generated from the comments in the source code. The
mathematics behind each routine is in the theory document
\theorypdf, which follows the notation and equation numbering of Karniadakis &
Sherwin, *Spectral/hp Element Methods for Computational Fluid Dynamics*, 2nd ed.

How to read the two together
----------------------------

- Every routine's documentation states the equation it implements, with the
  book's number, e.g. (A.1.3), or (D.n) for an equation derived in the theory
  document, and the section of \theorypdf where it is discussed.
- Inside the function bodies (see the source browser, or `src/`), each step is
  commented with the equation it computes.

Modules
-------

| Module | Header | Theory |
|---|---|---|
| \ref jacobi "Jacobi polynomials" | jacobi.hpp | \theorypdf, Section 2 |

Tests
-----

test_jacobi.cpp lists every identity checked and its tolerance
(\theorypdf, Section 3).

Building, testing and choosing the compiler are described in the
`README.md` at the root of the repository.

Building this documentation
---------------------------

```bash
cmake --preset gcc
cmake --build --preset gcc --target doc
# open build/gcc/doc/html/index.html
```

The `doc` target runs Doxygen and, if `latexmk` is installed, builds the theory
PDF and copies it next to the HTML pages so that the \theorypdf links work.
Formulas are rendered by MathJax, loaded from a CDN, so viewing them needs an
internet connection.
