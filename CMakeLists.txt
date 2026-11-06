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

# --- tests
enable_testing()
add_executable(test_jacobi tests/test_jacobi.cpp)
target_link_libraries(test_jacobi PRIVATE dgpoly)
target_compile_options(test_jacobi PRIVATE ${DGPOLY_WARNINGS})

if(POLYLIB_DIR)
  if(NOT EXISTS "${POLYLIB_DIR}/polylib.cpp" OR NOT EXISTS "${POLYLIB_DIR}/polylib.h")
    message(FATAL_ERROR "POLYLIB_DIR=${POLYLIB_DIR} does not contain polylib.cpp and polylib.h")
  endif()
  add_library(polylib_ref STATIC "${POLYLIB_DIR}/polylib.cpp")
  target_include_directories(polylib_ref PUBLIC "${POLYLIB_DIR}")
  target_compile_options(polylib_ref PRIVATE -w)                  # legacy code: silence warnings
  set_target_properties(polylib_ref PROPERTIES CXX_STANDARD 14)   # uses 'register'
  target_link_libraries(test_jacobi PRIVATE polylib_ref)
  target_compile_definitions(test_jacobi PRIVATE DGPOLY_HAVE_POLYLIB)
  message(STATUS "Polylib reference: ${POLYLIB_DIR}")
else()
  message(STATUS "Polylib reference: not set, Polylib comparison tests skipped")
endif()

add_test(NAME jacobi COMMAND test_jacobi)
