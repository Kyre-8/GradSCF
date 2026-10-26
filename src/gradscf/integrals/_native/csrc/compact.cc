// Stable compact-integral FFI ABI dispatch.
#include "compact.h"

static ffi::Error Compact(ffi::BufferR2<ffi::S32> atoms, ffi::BufferR2<ffi::S32> basis,
    ffi::BufferR1<ffi::F64> env, ffi::ResultBuffer<ffi::F64> result,
    int32_t layout, int32_t cart, int32_t split) {
  try {
    IntegralTables t(atoms,basis,env,4,cart);
    double* out=result->typed_data();std::fill(out,out+result->element_count(),0.);
    if (layout==0 || layout==1) {
      packed_eri(t, layout, cart, out, result->element_count());
    } else if (layout==2 || layout==3) {
      density_fitting(t, layout, cart, split, out, result->element_count());
    } else throw std::invalid_argument("Unknown compact integral layout");
    return ffi::Error::Success();
  } catch (const std::exception& e) {return ffi::Error::Internal(e.what());}
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFCompact(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFCompact,Compact,
    ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
      .Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("layout").Attr<int32_t>("cart").Attr<int32_t>("split"));
