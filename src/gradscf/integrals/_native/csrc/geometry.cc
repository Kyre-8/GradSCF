// Analytic coordinate products from libcint derivative kernels. No Python,
// PySCF bridge, numerical differences, or full coordinate Jacobian.
#include "tables.h"

template<bool transpose>
static ffi::Error Geometry(ffi::BufferR2<ffi::S32> atoms,
                           ffi::BufferR2<ffi::S32> basis,
                           ffi::BufferR1<ffi::F64> environment,
                           ffi::Buffer<ffi::F64> vector,
                           ffi::ResultBuffer<ffi::F64> result,
                           int32_t op, int32_t cart) {
  try {
    IntegralTables t(atoms,basis,environment,op,cart,true);
    if constexpr (transpose) {
      t.check_shape(vector,op);
      if (result->dimensions().size()!=1 || result->element_count()!=t.env.size())
        throw std::invalid_argument("Invalid geometry VJP output shape");
    } else {
      t.check_shape(*result,op);
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
      if constexpr (transpose) output[env_index]+=derivative*input[integral_index];
      else output[integral_index]+=derivative*input[env_index];
    };
    const int n=t.n;
    auto index2=[&](int c,int i,int j) { return (size_t(c)*n+i)*n+j; };
    auto index4=[&](int i,int j,int k,int l) { return ((size_t(i)*n+j)*n+k)*n+l; };
    std::vector<double> block,cache;
    if (op==4) {
      Intor ip=cart ? int2e_ip1_cart : int2e_ip1_sph;
      for (int i=0; i<t.nbas; ++i) for (int j=0; j<t.nbas; ++j)
        for (int k=0; k<t.nbas; ++k) for (int l=0; l<t.nbas; ++l) {
          int shells[]={i,j,k,l};
          int di=t.ao[i+1]-t.ao[i], dj=t.ao[j+1]-t.ao[j];
          int dk=t.ao[k+1]-t.ao[k], dl=t.ao[l+1]-t.ao[l];
          size_t size=size_t(di)*dj*dk*dl;
          t.shell(ip,shells,3*size,block,cache);
          for (int x=0; x<3; ++x) for (int s=0; s<dl; ++s)
            for (int r=0; r<dk; ++r) for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p) {
              // d/d(center) = -d/d(electron coordinate); first AO contiguous.
              double v=-block[x*size+((size_t(s)*dk+r)*dj+q)*di+p];
              int a=t.ao[i]+p,b=t.ao[j]+q,c=t.ao[k]+r,d=t.ao[l]+s;
              int ptr=t.coord_ptr(i)+x;
              add(index4(a,b,c,d),ptr,v); add(index4(b,a,c,d),ptr,v);
              add(index4(c,d,a,b),ptr,v); add(index4(c,d,b,a),ptr,v);
            }
        }
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

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFGeometryJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFGeometryVJP(XLA_FFI_CallFrame*);
#define GEOMETRY_BIND ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>() \
    .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>() \
    .Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() \
    .Attr<int32_t>("operator").Attr<int32_t>("cart")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFGeometryJVP, Geometry<false>, GEOMETRY_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFGeometryVJP, Geometry<true>, GEOMETRY_BIND);
