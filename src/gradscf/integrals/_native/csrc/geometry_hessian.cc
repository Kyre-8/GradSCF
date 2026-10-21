// Analytic second coordinate products, evaluated and contracted shell by shell.
// No coordinate Jacobian/Hessian, Python callback, or finite difference is formed.
#include "geometry_contraction.h"

template<bool transpose>
static ffi::Error GeometryHessianImpl(ffi::BufferR2<ffi::S32> atoms,
                                  ffi::BufferR2<ffi::S32> basis,
                                  ffi::BufferR1<ffi::F64> environment,
                                  ffi::Buffer<ffi::F64> direction,
                                  ffi::Buffer<ffi::F64> vector,
                                  ffi::ResultBuffer<ffi::F64> result,
                                  int32_t op, int32_t cart,
                                  const int32_t* slice=nullptr,size_t count=0) {
  try {
    IntegralTables t(atoms,basis,environment,op,cart,true);
    ERIWindow window(t,slice,count);
    if (slice && op!=4) throw std::invalid_argument("shell_slice applies only to ERI");
    auto check_shape=[&](const ffi::Buffer<ffi::F64>& buffer) {
      if (slice) window.check_shape(buffer); else t.check_shape(buffer,op);
    };
    std::vector<bool> allowed(t.env.size(),false);
    for (int a=0; a<t.natm; ++a)
      for (int x=0; x<3; ++x) allowed[t.atm[a*ATM_SLOTS+PTR_COORD]+x]=true;
    if (op==3) for (int x=0; x<3; ++x) allowed[PTR_COMMON_ORIG+x]=true;
    auto check_direction=[&](const ffi::Buffer<ffi::F64>& buffer) {
      if (buffer.dimensions().size()!=1 || buffer.element_count()!=t.env.size())
        throw std::invalid_argument("Invalid geometry Hessian direction shape");
      for (size_t i=0; i<t.env.size(); ++i) {
        if (!std::isfinite(buffer.typed_data()[i]))
          throw std::invalid_argument("Derivative vector must be finite");
        if (!allowed[i] && buffer.typed_data()[i]!=0.)
          throw std::invalid_argument("Native Hessian supports coordinates/origin only, not basis exponents/coefficients");
      }
    };
    check_direction(direction);
    if constexpr (transpose) {
      check_shape(vector);
      if (result->dimensions().size()!=1 || result->element_count()!=t.env.size())
        throw std::invalid_argument("Invalid geometry Hessian VJP output shape");
      for (size_t i=0; i<vector.element_count(); ++i)
        if (!std::isfinite(vector.typed_data()[i]))
          throw std::invalid_argument("Derivative vector must be finite");
    } else {
      check_shape(*result);
      check_direction(vector);
    }
    const double* u=direction.typed_data();
    const double* input=vector.typed_data();
    double* output=result->typed_data();
    std::fill(output,output+result->element_count(),0.);
    auto evaluate=[&](Intor kernel,int* shells,size_t size,
                      std::vector<double>& block,std::vector<double>& cache,int) {
      t.shell(kernel,shells,size,block,cache);
    };
    contract_geometry_hessian<transpose>(t,op,cart,u,input,output,evaluate,slice ? &window : nullptr);
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

template<bool transpose>
static ffi::Error GeometryHessian(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
                                  ffi::BufferR1<ffi::F64> env,ffi::Buffer<ffi::F64> direction,
                                  ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,
                                  int32_t op,int32_t cart) {
  return GeometryHessianImpl<transpose>(atoms,basis,env,direction,vector,result,op,cart);
}

template<bool transpose>
static ffi::Error ERISliceGeometryHessian(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
                                          ffi::BufferR1<ffi::F64> env,ffi::Buffer<ffi::F64> direction,
                                          ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,
                                          int32_t op,int32_t cart,ffi::Span<const int32_t> slice) {
  return GeometryHessianImpl<transpose>(atoms,basis,env,direction,vector,result,op,cart,slice.begin(),slice.size());
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFGeometryHessianJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFGeometryHessianVJP(XLA_FFI_CallFrame*);
#define GEOMETRY_HESSIAN_BIND ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>() \
    .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>() \
    .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>() \
    .Ret<ffi::Buffer<ffi::F64>>().Attr<int32_t>("operator").Attr<int32_t>("cart")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFGeometryHessianJVP, GeometryHessian<false>, GEOMETRY_HESSIAN_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFGeometryHessianVJP, GeometryHessian<true>, GEOMETRY_HESSIAN_BIND);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFERISliceGeometryHessianJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFERISliceGeometryHessianVJP(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceGeometryHessianJVP, ERISliceGeometryHessian<false>,
                              GEOMETRY_HESSIAN_BIND.Attr<ffi::Span<const int32_t>>("shell_slice"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceGeometryHessianVJP, ERISliceGeometryHessian<true>,
                              GEOMETRY_HESSIAN_BIND.Attr<ffi::Span<const int32_t>>("shell_slice"));
