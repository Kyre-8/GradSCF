// Analytic coordinate products from libcint derivative kernels. No Python,
// PySCF bridge, numerical differences, or full coordinate Jacobian.
#include "eri_geometry_entries.h"

template<bool transpose>
static ffi::Error GeometryImpl(ffi::BufferR2<ffi::S32> atoms,
                           ffi::BufferR2<ffi::S32> basis,
                           ffi::BufferR1<ffi::F64> environment,
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
    if constexpr (transpose) {
      check_shape(vector);
      if (result->dimensions().size()!=1 || result->element_count()!=t.env.size())
        throw std::invalid_argument("Invalid geometry VJP output shape");
    } else {
      check_shape(*result);
      if (vector.dimensions().size()!=1 || vector.element_count()!=t.env.size())
        throw std::invalid_argument("Invalid geometry JVP direction shape");
      std::vector<bool> allowed(t.env.size(),false);
      for (int a=0; a<t.natm; ++a)
        for (int x=0; x<3; ++x) allowed[t.atm[a*ATM_SLOTS+PTR_COORD]+x]=true;
      if (op==3) for (int x=0; x<3; ++x) allowed[PTR_COMMON_ORIG+x]=true;
      for (size_t i=0; i<t.env.size(); ++i)
        if (!allowed[i] && vector.typed_data()[i]!=0.)
          throw std::invalid_argument("Native JVP supports coordinates/origin only, not basis exponents/coefficients");
    }
    const double* input=vector.typed_data();
    for (size_t i=0; i<vector.element_count(); ++i)
      if (!std::isfinite(input[i])) throw std::invalid_argument("Derivative vector must be finite");
    double* output=result->typed_data();
    std::fill(output,output+result->element_count(),0.);
    auto add=[&](size_t integral_index, int env_index, double derivative) {
      if (integral_index==ERIWindow::outside) return;
      if constexpr (transpose) output[env_index]+=derivative*input[integral_index];
      else output[integral_index]+=derivative*input[env_index];
    };
    const int n=t.n;
    auto index2=[&](int c,int i,int j) { return (size_t(c)*n+i)*n+j; };
    std::vector<double> block,cache;
    if (op==4) {
      for_each_eri_geometry_entry(t,cart,add,slice ? &window : nullptr);
    } else {
      Intor ips_cart[]={int1e_ipovlp_cart,int1e_ipkin_cart,int1e_ipnuc_cart,int1e_irp_cart};
      Intor ips_sph[]={int1e_ipovlp_sph,int1e_ipkin_sph,int1e_ipnuc_sph,int1e_irp_sph};
      Intor ip=cart ? ips_cart[op] : ips_sph[op];
      int components=op==3 ? 3 : 1;
      for (int i=0; i<t.nbas; ++i) for (int j=0; j<t.nbas; ++j) {
        int shells[]={i,j};
        int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
        size_t size=size_t(di)*dj;
        t.shell(ip,shells,3*components*size,block,cache);
        for (int c=0; c<components; ++c) for (int x=0; x<3; ++x)
          for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p) {
            double v=-block[(c*3+x)*size+q*di+p];
            int a=t.ao[i]+p,b=t.ao[j]+q;
            // irp differentiates ket; the other kernels differentiate bra.
            int ptr=t.coord_ptr(op==3 ? j : i)+x;
            add(index2(c,a,b),ptr,v); add(index2(c,b,a),ptr,v);
          }
        if (op==2) {
          // Derivative of the attraction operator itself, independently of AO centers.
          for (int atom=0; atom<t.natm; ++atom) {
            int z=t.atm[atom*ATM_SLOTS+CHARGE_OF];
            if (!z) continue;
            int ptr=t.atm[atom*ATM_SLOTS+PTR_COORD];
            for (int x=0; x<3; ++x) t.env[PTR_RINV_ORIG+x]=t.env[ptr+x];
            t.shell(cart ? int1e_iprinv_cart : int1e_iprinv_sph,shells,3*size,block,cache);
            for (int x=0; x<3; ++x) for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p) {
              double v=-z*block[x*size+q*di+p];
              int a=t.ao[i]+p,b=t.ao[j]+q;
              add(index2(0,a,b),ptr+x,v); add(index2(0,b,a),ptr+x,v);
            }
          }
        } else if (op==3) {
          t.shell(cart ? int1e_ovlp_cart : int1e_ovlp_sph,shells,size,block,cache);
          for (int c=0; c<3; ++c) for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p)
            add(index2(c,t.ao[i]+p,t.ao[j]+q),PTR_COMMON_ORIG+c,-block[q*di+p]);
        }
      }
    }
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

template<bool transpose>
static ffi::Error Geometry(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
                           ffi::BufferR1<ffi::F64> env,ffi::Buffer<ffi::F64> vector,
                           ffi::ResultBuffer<ffi::F64> result,int32_t op,int32_t cart) {
  return GeometryImpl<transpose>(atoms,basis,env,vector,result,op,cart);
}

template<bool transpose>
static ffi::Error ERISliceGeometry(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
                                   ffi::BufferR1<ffi::F64> env,ffi::Buffer<ffi::F64> vector,
                                   ffi::ResultBuffer<ffi::F64> result,int32_t op,int32_t cart,
                                   ffi::Span<const int32_t> slice) {
  return GeometryImpl<transpose>(atoms,basis,env,vector,result,op,cart,slice.begin(),slice.size());
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFGeometryJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFGeometryVJP(XLA_FFI_CallFrame*);
#define GEOMETRY_BIND ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>() \
    .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>() \
    .Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() \
    .Attr<int32_t>("operator").Attr<int32_t>("cart")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFGeometryJVP, Geometry<false>, GEOMETRY_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFGeometryVJP, Geometry<true>, GEOMETRY_BIND);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFERISliceGeometryJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFERISliceGeometryVJP(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceGeometryJVP, ERISliceGeometry<false>,
                              GEOMETRY_BIND.Attr<ffi::Span<const int32_t>>("shell_slice"));
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFERISliceGeometryVJP, ERISliceGeometry<true>,
                              GEOMETRY_BIND.Attr<ffi::Span<const int32_t>>("shell_slice"));
