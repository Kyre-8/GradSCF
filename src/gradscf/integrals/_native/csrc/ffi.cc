// CPU float64 JAX FFI adapter. No Python callbacks or process-global optimizer.
#include "coulomb_shells.h"
#include "eri_window.h"
extern "C" {
using LegacyIntor = int (*)();
void GTOnr2e_fill_s1(LegacyIntor, LegacyIntor, double*, double*, int, int, int,
                    int*, int*, CINTOpt*, int*, int, int*, int, double*);
}
static int NoPrescreen() { return 1; }

static ffi::Error Integrals(ffi::BufferR2<ffi::S32> atoms,
                           ffi::BufferR2<ffi::S32> basis,
                           ffi::BufferR1<ffi::F64> environment,
                           ffi::ResultBuffer<ffi::F64> result,
                           int32_t op, int32_t cart) {
  try {
    IntegralTables tables(atoms, basis, environment, op, cart, true);
    const int natm=tables.natm, nbas=tables.nbas;
    auto& atm=tables.atm;
    auto& bas=tables.bas;
    auto& env=tables.env;
    auto& ao=tables.ao;
    const int n = ao.back(), components = op == 3 ? 3 : 1;
    const int rank = op == 4 ? 4 : (op == 3 ? 3 : 2);
    if (op < 0 || op > 4 || (cart != 0 && cart != 1) || n < 1 ||
        result->dimensions().size() != rank)
      return ffi::Error::InvalidArgument("Invalid native integral output shape or operator");
    for (int i = 0; i < rank; ++i)
      if (result->dimensions()[i] != (op == 3 && i == 0 ? 3 : n))
        return ffi::Error::InvalidArgument("AO count does not match basis table");
    Intor cart_funcs[] = {int1e_ovlp_cart, int1e_kin_cart, int1e_nuc_cart,
                          int1e_r_cart, int2e_cart};
    Intor sph_funcs[] = {int1e_ovlp_sph, int1e_kin_sph, int1e_nuc_sph,
                         int1e_r_sph, int2e_sph};
    Intor intor = cart ? cart_funcs[op] : sph_funcs[op];
    double* output = result->typed_data();
    std::fill(output, output + result->element_count(), 0.);
    if (op == 4) {
      int slices[] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
      int maxdim = 0, maxcache = 0;
      for (int s = 0; s < nbas; ++s) {
        maxdim = std::max(maxdim, ao[s+1] - ao[s]);
        int shells[] = {s, s, s, s};
        const int cache_size = intor(nullptr, nullptr, shells, atm.data(), natm,
                                     bas.data(), nbas, env.data(), nullptr, nullptr);
        // libcint's 32-bit cache ABI returns zero when its internal Cartesian
        // recurrence/contraction workspace overflows, including sph kernels.
        // It is not a valid empty cache. Never enter fill with that sentinel.
        if (cache_size <= 0)
          return ffi::Error::InvalidArgument(
              "ERI libcint cache query failed or exceeded int32 indexing");
        maxcache = std::max(maxcache, cache_size);
      }
      const int64_t blocksize = int64_t(maxdim)*maxdim*maxdim*maxdim;
      if (blocksize > std::numeric_limits<int>::max())
        return ffi::Error::InvalidArgument("ERI shell workspace exceeds int32 indexing");
      // Own allocation in C++ so failure is returned to JAX, not dereferenced by C.
      std::vector<double> scratch(size_t(blocksize) + maxcache);
      for (int i = 0; i < nbas; ++i)
        for (int j = 0; j < nbas; ++j)
          GTOnr2e_fill_s1(reinterpret_cast<LegacyIntor>(intor), NoPrescreen,
                         output, scratch.data(), 1, i, j, slices, ao.data(),
                         nullptr, atm.data(), natm, bas.data(), nbas, env.data());
    } else {
      for (int i = 0; i < nbas; ++i) {
        for (int j = 0; j < nbas; ++j) {
          int shells[] = {i, j};
          const int di = ao[i+1] - ao[i], dj = ao[j+1] - ao[j];
          const size_t size = size_t(di)*dj;
          std::vector<double> block(size * components, 0.);
          intor(block.data(), nullptr, shells, atm.data(), natm,
                bas.data(), nbas, env.data(), nullptr, nullptr);
          // libcint shell blocks have the first AO index contiguous.
          for (int c = 0; c < components; ++c)
            for (int p = 0; p < di; ++p)
              for (int q = 0; q < dj; ++q)
                output[(size_t(c)*n + ao[i]+p)*n + ao[j]+q] =
                    block[size_t(c)*size + q*di + p];
        }
      }
    }
    return ffi::Error::Success();
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

extern "C" __attribute__((visibility("default")))
XLA_FFI_Error* GradSCFIntegrals(XLA_FFI_CallFrame*);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GradSCFIntegrals, Integrals,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("operator")
        .Attr<int32_t>("cart"));

static ffi::Error ERISlice(ffi::BufferR2<ffi::S32> atoms,
                           ffi::BufferR2<ffi::S32> basis,
                           ffi::BufferR1<ffi::F64> environment,
                           ffi::ResultBuffer<ffi::F64> result,
                           int32_t cart,ffi::Span<const int32_t> shell_slice) {
  try {
    IntegralTables t(atoms,basis,environment,4,cart,true);
    ERIWindow window(t,shell_slice.begin(),shell_slice.size());
    window.check_shape(*result);
    Intor intor=cart ? int2e_cart : int2e_sph;
    LocalOptimizer optimizer(t,4);
    int maxcache=0;
    int64_t maxblock=1;
    std::vector<bool> selected(t.nbas,false);
    for (int a=0;a<4;++a) {
      int maxdim=0;
      for (int s=window.shells[2*a];s<window.shells[2*a+1];++s) {
        selected[s]=true;
        maxdim=std::max(maxdim,t.ao[s+1]-t.ao[s]);
      }
      maxblock*=maxdim;
    }
    if (maxblock>std::numeric_limits<int>::max())
      throw std::invalid_argument("ERI shell workspace exceeds int32 indexing");
    // Match the established full driver and vendor GTOmax_cache_size bound,
    // querying only the selected shell union, never all shell quartets.
    for (int s=0;s<t.nbas;++s) if (selected[s]) {
      int shells[]={s,s,s,s};
      int count=intor(nullptr,nullptr,shells,t.atm.data(),t.natm,t.bas.data(),
                       t.nbas,t.env.data(),optimizer.value,nullptr);
      if (count<=0)
        throw std::invalid_argument("ERI libcint cache query failed or exceeded int32 indexing");
      maxcache=std::max(maxcache,count);
    }
    std::vector<double> scratch(size_t(maxblock)+maxcache);
    double* output=result->typed_data();
    std::fill(output,output+result->element_count(),0.);
    for (int i=0;i<window.shells[1]-window.shells[0];++i)
      for (int j=0;j<window.shells[3]-window.shells[2];++j)
        GTOnr2e_fill_s1(reinterpret_cast<LegacyIntor>(intor),NoPrescreen,
                        output,scratch.data(),1,i,j,window.shells.data(),t.ao.data(),
                        optimizer.value,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data());
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFERISlice(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GradSCFERISlice, ERISlice,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
        .Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("cart")
        .Attr<ffi::Span<const int32_t>>("shell_slice"));
