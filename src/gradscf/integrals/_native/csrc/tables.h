#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>
#include "xla/ffi/api/ffi.h"
extern "C" {
#include "cint.h"
#include "cint_funcs.h"
}
#undef atm
#undef bas

namespace ffi = xla::ffi;
using Intor = int (*)(double*, int*, int*, int*, int, int*, int,
                     double*, CINTOpt*, double*);

// Explicit scalar comparison also supports older compilers with newer SDK headers.
struct ShellShapeLess {
  bool operator()(const std::array<int,13>& a,const std::array<int,13>& b) const {
    for (size_t i=0;i<a.size();++i) if (a[i]!=b[i]) return a[i]<b[i];
    return false;
  }
};

// Owned libcint buffers shared by value and derivative drivers.
struct IntegralTables {
  int natm, nbas, n;
  std::vector<int> atm, bas, ao;
  std::vector<double> env;
  Intor cached_intor=nullptr;
  std::map<std::array<int,13>,int,ShellShapeLess> cache_sizes;
  IntegralTables(ffi::BufferR2<ffi::S32> atoms, ffi::BufferR2<ffi::S32> basis,
                 ffi::BufferR1<ffi::F64> environment, int op, int cart) {
    if (atoms.dimensions()[1] != ATM_SLOTS || basis.dimensions()[1] != BAS_SLOTS ||
        environment.element_count() < 20 || op < 0 || op > 4 || (cart != 0 && cart != 1))
      throw std::invalid_argument("Invalid libcint table dimensions or operator");
    natm = atoms.dimensions()[0]; nbas = basis.dimensions()[0];
    atm.assign(atoms.typed_data(), atoms.typed_data()+atoms.element_count());
    bas.assign(basis.typed_data(), basis.typed_data()+basis.element_count());
    env.assign(environment.typed_data(), environment.typed_data()+environment.element_count());
    for (double v : env)
      if (!std::isfinite(v)) throw std::invalid_argument("Native integral env must be finite");
    if (env[PTR_RANGE_OMEGA] != 0.)
      throw std::invalid_argument("Range-separated integrals are not supported");
    auto inside = [&](int ptr, int64_t count) {
      return ptr >= PTR_ENV_START && count > 0 && int64_t(ptr)+count <= int64_t(env.size());
    };
    for (int a=0; a<natm; ++a) {
      const int* row = atm.data()+a*ATM_SLOTS;
      if (!inside(row[PTR_COORD], 3) || row[NUC_MOD_OF] != 1)
        throw std::invalid_argument("Native backend requires valid point nuclei");
    }
    ao.assign(nbas+1, 0);
    for (int s=0; s<nbas; ++s) {
      const int* row = bas.data()+s*BAS_SLOTS;
      int l=row[ANG_OF], np=row[NPRIM_OF], nc=row[NCTR_OF];
      if (row[ATOM_OF]<0 || row[ATOM_OF]>=natm || l<0 || l>12 ||
          np<1 || np>64 || nc<1 || nc>64 || row[KAPPA_OF]!=0 ||
          !inside(row[PTR_EXP],np) || !inside(row[PTR_COEFF],int64_t(np)*nc))
        throw std::invalid_argument("Invalid or unsupported libcint basis table");
      for (int p=0; p<np; ++p)
        if (env[row[PTR_EXP]+p]<=0.) throw std::invalid_argument("Gaussian exponents must be positive");
      ao[s+1]=ao[s]+nc*(cart ? (l+1)*(l+2)/2 : 2*l+1);
    }
    n=ao.back();
    if (n<1) throw std::invalid_argument("Invalid AO count");
  }
  int coord_ptr(int shell) const { return atm[bas[shell*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD]; }
  void check_shape(const ffi::Buffer<ffi::F64>& buffer, int op) const {
    const int rank=op==4 ? 4 : (op==3 ? 3 : 2);
    if (buffer.dimensions().size()!=rank) throw std::invalid_argument("Invalid integral tensor rank");
    for (int i=0; i<rank; ++i)
      if (buffer.dimensions()[i]!=(op==3 && i==0 ? 3 : n))
        throw std::invalid_argument("AO count does not match basis table");
  }
  void shell(Intor intor, int* shells, size_t size, std::vector<double>& block,
             std::vector<double>& cache, int legs=0) {
    if (size>size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Derivative shell workspace exceeds int32 indexing");
    // Opt in only for real 1e/3c parameter products. With fixed global ENV
    // controls, libcint scratch size depends on the intor and shell shapes,
    // not on coordinates, exponents, coefficients, or their storage pointers.
    // Other derivative drivers retain their original per-call size query.
    std::array<int,13> key{};
    if (legs) {
      if (legs!=2 && legs!=3) throw std::invalid_argument("Invalid cached shell rank");
      if (cached_intor!=intor) {cache_sizes.clear();cached_intor=intor;}
      key[0]=legs;
      for (int a=0;a<legs;++a) {
        const int* row=bas.data()+shells[a]*BAS_SLOTS;
        key[1+4*a]=row[ANG_OF];key[2+4*a]=row[NPRIM_OF];
        key[3+4*a]=row[NCTR_OF];key[4+4*a]=row[KAPPA_OF];
      }
    }
    auto found=cache_sizes.find(key);
    int nc=legs && found!=cache_sizes.end() ? found->second
      : intor(nullptr,nullptr,shells,atm.data(),natm,bas.data(),nbas,env.data(),nullptr,nullptr);
    if (nc<=0) throw std::invalid_argument("libcint derivative cache query failed or exceeded int32 indexing");
    if (legs) cache_sizes.emplace(key,nc);
    block.assign(size,0.); cache.resize(nc);
    intor(block.data(),nullptr,shells,atm.data(),natm,bas.data(),nbas,env.data(),nullptr,cache.data());
  }
};
