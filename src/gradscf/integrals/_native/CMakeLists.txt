cmake_minimum_required(VERSION 3.20)
project(gradscf_native LANGUAGES C CXX)

if(NOT JAX_INCLUDE_DIR)
  message(FATAL_ERROR "Set JAX_INCLUDE_DIR using jax.ffi.include_dir()")
endif()
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# The vendor tree is read-only source; generated headers stay in the build tree.
set(cint_VERSION 6.1.3)
set(cint_SOVERSION 6)
set(HAVE_EXPL 1)
set(HAVE_SQRTL 1)
set(HAVE_FABSL 1)
configure_file(vendor/libcint/include/cint.h.in generated/cint.h)
configure_file(vendor/libcint/src/cint_config.h.in generated/cint_config.h)
configure_file(include/config.h generated/config.h COPYONLY)

set(CINT_SOURCES
  cart2sph cint1e cint2e cint2c2e cint3c2e cint_bas fblas g1e g2e misc optimizer fmt
  rys_wheeler eigh rys_roots find_roots g1e_grids g3c1e g2c2e g3c2e)
list(TRANSFORM CINT_SOURCES PREPEND "vendor/libcint/src/")
list(TRANSFORM CINT_SOURCES APPEND ".c")
add_library(gradscf_integrals SHARED
  ${CINT_SOURCES} vendor/libcint/src/autocode/intor1.c
  vendor/libcint/src/autocode/int3c2e.c
  vendor/libcint/src/autocode/grad1.c vendor/libcint/src/autocode/grad2.c
  vendor/libcint/src/autocode/hess.c
  vendor/pyscf/pyscf/lib/gto/fill_int2e.c csrc/ffi.cc csrc/geometry.cc csrc/geometry_hessian.cc csrc/geometry_higher.cc csrc/coefficients.cc csrc/exponents.cc csrc/compact.cc
  csrc/packed_eri.cc csrc/density_fitting.cc csrc/direct_jk.cc csrc/direct_jk_geometry.cc csrc/packed_jk.cc
  vendor/pyscf_ecp/gto/nr_ecp.c vendor/pyscf_ecp/np_helper/np_helper.c csrc/ecp.cc)
target_include_directories(gradscf_integrals PRIVATE
  "${JAX_INCLUDE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/generated"
  vendor/libcint/src vendor/libcint/include vendor/pyscf_ecp)
find_package(Threads REQUIRED)
target_link_libraries(gradscf_integrals PRIVATE Threads::Threads)

if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  target_link_libraries(gradscf_integrals PRIVATE m ${ACCELERATE_FRAMEWORK})
else()
  if(GRADSCF_BLAS_LIBRARY)
    if(NOT EXISTS "${GRADSCF_BLAS_LIBRARY}")
      message(FATAL_ERROR "GRADSCF_BLAS_LIBRARY does not exist")
    endif()
    set(BLAS_LIBRARIES "${GRADSCF_BLAS_LIBRARY}")
  else()
    find_package(BLAS REQUIRED)
  endif()
  target_link_libraries(gradscf_integrals PRIVATE m ${BLAS_LIBRARIES})
endif()
target_compile_options(gradscf_integrals PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-Wno-deprecated-non-prototype>)
# Link-local vendor symbols cannot interpose with a separately imported PySCF.
if(APPLE)
  target_link_options(gradscf_integrals PRIVATE
    "LINKER:-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.macos")
elseif(UNIX)
  target_link_options(gradscf_integrals PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
    "LINKER:-z,defs")
else()
  message(FATAL_ERROR "Native integral build currently supports Linux and macOS")
endif()
if(GRADSCF_OUTPUT_DIR)
  set_target_properties(gradscf_integrals PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${GRADSCF_OUTPUT_DIR}")
endif()
