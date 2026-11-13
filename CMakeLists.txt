cmake_minimum_required(VERSION 3.21)
project(dgpoly LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})")

# ---------------------------------------------------------------------------
# Compiler-specific flags
#   CMAKE_CXX_COMPILER_ID is GNU (g++), NVHPC (nvc++), IntelLLVM (icpx),
#   Clang (clang++), ...  The choice of compiler itself is made at configure
#   time (see CMakePresets.json or -DCMAKE_CXX_COMPILER=...), never here.
# ---------------------------------------------------------------------------
option(DGPOLY_NATIVE "Optimise for the CPU of the build machine" OFF)

set(DGPOLY_WARNINGS "")
set(DGPOLY_ARCH "")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
  set(DGPOLY_WARNINGS -Wall -Wextra -Wpedantic)
  if(DGPOLY_NATIVE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
      set(DGPOLY_ARCH -xHost)
    else()
      set(DGPOLY_ARCH -march=native)
    endif()
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  set(DGPOLY_WARNINGS -Wall)            # nvc++ has no -Wextra / -Wpedantic
  if(DGPOLY_NATIVE)
    set(DGPOLY_ARCH -tp=native)
  endif()
endif()

# --- the library: Jacobi polynomials (later: zeros, quadrature, D, Vandermonde)
add_library(dgpoly src/jacobi.cpp)
target_include_directories(dgpoly PUBLIC include)
target_compile_options(dgpoly PRIVATE ${DGPOLY_WARNINGS} ${DGPOLY_ARCH})

# --- Optional: Polylib as an external reference for the tests
#     cmake --preset gcc -DPOLYLIB_DIR=/path/to/folder/with/polylib.cpp
#     Without it, the tests that compare against Polylib are skipped.
set(POLYLIB_DIR "" CACHE PATH "Folder containing polylib.cpp and polylib.h (optional)")

# --- tests: one executable per module, Polylib linked when available
enable_testing()
set(DGPOLY_TESTS test_jacobi test_zeros)
foreach(t IN LISTS DGPOLY_TESTS)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE dgpoly)
  target_compile_options(${t} PRIVATE ${DGPOLY_WARNINGS})
endforeach()

if(POLYLIB_DIR)
  if(NOT EXISTS "${POLYLIB_DIR}/polylib.cpp" OR NOT EXISTS "${POLYLIB_DIR}/polylib.h")
    message(FATAL_ERROR "POLYLIB_DIR=${POLYLIB_DIR} does not contain polylib.cpp and polylib.h")
  endif()
  add_library(polylib_ref STATIC "${POLYLIB_DIR}/polylib.cpp")
  target_include_directories(polylib_ref PUBLIC "${POLYLIB_DIR}")
  target_compile_options(polylib_ref PRIVATE -w)                  # legacy code: silence warnings
  set_target_properties(polylib_ref PROPERTIES CXX_STANDARD 14)   # uses 'register'
  foreach(t IN LISTS DGPOLY_TESTS)
    target_link_libraries(${t} PRIVATE polylib_ref)
    target_compile_definitions(${t} PRIVATE DGPOLY_HAVE_POLYLIB)
  endforeach()
  message(STATUS "Polylib reference: ${POLYLIB_DIR}")
else()
  message(STATUS "Polylib reference: not set, Polylib comparison tests skipped")
endif()

foreach(t IN LISTS DGPOLY_TESTS)
  add_test(NAME ${t} COMMAND ${t})
endforeach()

# ---------------------------------------------------------------------------
# Documentation:  cmake --build --preset gcc --target doc
#   HTML API documentation (Doxygen) in <build>/doc/html, with the theory PDF
#   (docs/theory, built with latexmk if available) copied next to it so that
#   the \theorypdf links in the code comments work.
# ---------------------------------------------------------------------------
find_package(Doxygen QUIET)
find_program(LATEXMK_EXECUTABLE latexmk)

if(DOXYGEN_FOUND)
  set(DOC_DIR "${CMAKE_BINARY_DIR}/doc")
  if(DOXYGEN_DOT_FOUND)
    set(DOC_HAVE_DOT YES)
  else()
    set(DOC_HAVE_DOT NO)
  endif()

  set(DOC_COMMANDS
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DOC_DIR}"
    COMMAND ${CMAKE_COMMAND} -E env DGHNOUTS_DOC_DIR=${DOC_DIR} DGHNOUTS_HAVE_DOT=${DOC_HAVE_DOT}
            ${DOXYGEN_EXECUTABLE} docs/Doxyfile)

  if(LATEXMK_EXECUTABLE)
    list(APPEND DOC_COMMANDS
      COMMAND ${LATEXMK_EXECUTABLE} -pdf -interaction=nonstopmode -halt-on-error -quiet
              -output-directory=${DOC_DIR}/theory
              ${CMAKE_SOURCE_DIR}/docs/theory/dghnouts_theory.tex
      COMMAND ${CMAKE_COMMAND} -E copy ${DOC_DIR}/theory/dghnouts_theory.pdf ${DOC_DIR}/html/)
  else()
    message(STATUS "latexmk not found: 'doc' builds the HTML only, without the theory PDF")
  endif()

  add_custom_target(doc ${DOC_COMMANDS}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Building documentation in ${DOC_DIR}/html (open index.html)"
    VERBATIM)
else()
  message(STATUS "Doxygen not found: 'doc' target not available")
endif()

