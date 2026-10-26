// Analytic products in the raw, normalized libcint ENV contraction coefficients.
// Each AO leg has an independent shadow shell. VJPs uncontract one shell leg
// locally and consume its block immediately, without a primitive-pair cache.
#include "tables.h"
#include "parallel.h"
extern "C" {
extern CINTIntegralFunction int3c2e_cart, int3c2e_sph;
}

struct CoefficientTables {
  IntegralTables t;
  int first_shell=0, shell_stride=1;
  int op, cart, active, orbital_ao, auxiliary_ao, components;
  int shadow_shell[2], shadow_coeff[2];
  size_t count=0, pairs=0, shadow_capacity=0;
  std::vector<size_t> offsets;
  std::vector<int> angular, primitives, contractions;
  std::vector<double> block, cache;
  Intor intor;

  CoefficientTables(ffi::BufferR2<ffi::S32> atoms,
                    ffi::BufferR2<ffi::S32> basis,
                    ffi::BufferR1<ffi::F64> environment,
                    ffi::BufferR1<ffi::F64> coefficients,
                    int operation, int cartesian, int split)
      : t(atoms,basis,environment,operation==6 ? 4:operation,cartesian),
        op(operation), cart(cartesian) {
    if (op!=6 && (op<0 || op>3))
      throw std::invalid_argument("Native coefficient products support operators 0..3 and 6 only");
    if (op==6 ? (split<1 || split>=t.nbas) : split!=0)
      throw std::invalid_argument("Invalid coefficient orbital/auxiliary shell split");
    active=op==6 ? split:t.nbas;
    orbital_ao=t.ao[active]; auxiliary_ao=op==6 ? t.n-orbital_ao:0;
    components=op==3 ? 3:1;
    pairs=size_t(orbital_ao)*(orbital_ao+1)/2;
    for (int s=0; s<active; ++s) {
      const int* row=t.bas.data()+s*BAS_SLOTS;
      int np=row[NPRIM_OF], nc=row[NCTR_OF], l=row[ANG_OF];
      offsets.push_back(count); primitives.push_back(np); contractions.push_back(nc);
      angular.push_back(cart ? (l+1)*(l+2)/2:2*l+1);
      count+=size_t(np)*nc;
      shadow_capacity=std::max(shadow_capacity,size_t(np)*std::max(np,nc));
    }
    if (coefficients.element_count()!=count)
      throw std::invalid_argument("Native coefficient vector does not match active shells");
    check_finite(coefficients.typed_data(),count);
    if (t.env.size()+count+2*shadow_capacity>size_t(std::numeric_limits<int>::max()) ||
        t.nbas>std::numeric_limits<int>::max()-2)
      throw std::invalid_argument("Native coefficient workspace exceeds int32 indexing");
    // Restore each active shell into its own owned segment, including when
    // input BAS rows happen to share a PTR_COEFF in the fixed ENV.
    for (int s=0; s<active; ++s) {
      int* row=t.bas.data()+s*BAS_SLOTS;
      row[PTR_COEFF]=int(t.env.size());
      const double* input=coefficients.typed_data()+offsets[s];
      t.env.insert(t.env.end(),input,input+size_t(primitives[s])*contractions[s]);
    }
    const int original_shells=t.nbas;
    t.bas.resize(size_t(original_shells+2)*BAS_SLOTS);
    for (int leg=0; leg<2; ++leg) {
      shadow_shell[leg]=original_shells+leg;
      shadow_coeff[leg]=int(t.env.size());
      t.env.resize(t.env.size()+shadow_capacity,0.);
    }
    t.nbas+=2;
    Intor cart_funcs[]={int1e_ovlp_cart,int1e_kin_cart,int1e_nuc_cart,int1e_r_cart};
    Intor sph_funcs[]={int1e_ovlp_sph,int1e_kin_sph,int1e_nuc_sph,int1e_r_sph};
    intor=op==6 ? (cart ? int3c2e_cart:int3c2e_sph)
               : (cart ? cart_funcs[op]:sph_funcs[op]);
  }

  static void check_finite(const double* input, size_t size) {
    for (size_t i=0; i<size; ++i)
      if (!std::isfinite(input[i]))
        throw std::invalid_argument("Native coefficient product inputs must be finite");
  }
  void check_vector(const ffi::Buffer<ffi::F64>& vector) const {
    if (vector.dimensions().size()!=1 || vector.element_count()!=count)
      throw std::invalid_argument("Invalid native coefficient direction or output shape");
  }
  void check_integrals(const ffi::Buffer<ffi::F64>& tensor) const {
    if (op==6) {
      if (tensor.dimensions().size()!=2 || tensor.dimensions()[0]!=auxiliary_ao ||
          tensor.dimensions()[1]!=pairs)
        throw std::invalid_argument("Invalid packed three-center coefficient product shape");
    } else t.check_shape(tensor,op);
  }
  int shadow(int leg, int shell, const double* direction, bool identity=false) {
    const int* original=t.bas.data()+shell*BAS_SLOTS;
    int* row=t.bas.data()+shadow_shell[leg]*BAS_SLOTS;
    std::copy(original,original+BAS_SLOTS,row);
    row[PTR_COEFF]=shadow_coeff[leg];
    double* values=t.env.data()+shadow_coeff[leg];
    const int np=primitives[shell], nc=contractions[shell];
    if (identity) {
      row[NCTR_OF]=np;
      std::fill(values,values+size_t(np)*np,0.);
      for (int p=0; p<np; ++p) values[size_t(p)*np+p]=1.;
    } else {
      const double* input=direction ? direction+offsets[shell]
                                    : t.env.data()+original[PTR_COEFF];
      std::copy(input,input+size_t(np)*nc,values);
    }
    return shadow_shell[leg];
  }
  template<class Consume>
  void shell_pairs(Consume consume) {
    for (int i=first_shell; i<active; i+=shell_stride) for (int j=0; j<(op==6 ? i+1:active); ++j) {
      if (op==6) for (int k=active; k<int(t.ao.size())-1; ++k) consume(i,j,k);
      else consume(i,j,-1);
    }
  }
  size_t integral_index(int component, int a, int b, int auxiliary) const {
    if (op==6) return size_t(auxiliary-orbital_ao)*pairs+size_t(a)*(a+1)/2+b;
    return (size_t(component)*orbital_ao+a)*orbital_ao+b;
  }
  void evaluate(int i, int j, int k, const double* left, const double* right,
                int identity_leg, int& di, int& dj, int& dk) {
    int shells[]={shadow(0,i,left,identity_leg==0),
                  shadow(1,j,right,identity_leg==1),k};
    di=(identity_leg==0 ? primitives[i]:contractions[i])*angular[i];
    dj=(identity_leg==1 ? primitives[j]:contractions[j])*angular[j];
    dk=k<0 ? 1:t.ao[k+1]-t.ao[k];
    if (op==6) {
      // libcint also contracts spherical results in Cartesian space. Its
      // int32 cache query can wrap back to a positive value; reject the
      // guaranteed cache overflow before querying, including identity legs.
      int64_t cartesian_block=1;
      for (int leg=0; leg<3; ++leg) {
        const int* row=t.bas.data()+shells[leg]*BAS_SLOTS;
        const int64_t l=row[ANG_OF],nc=row[NCTR_OF];
        cartesian_block*=nc*(l+1)*(l+2)/2;
      }
      if (3*cartesian_block>=std::numeric_limits<int>::max())
        throw std::invalid_argument("Three-center internal Cartesian cache exceeds int32 indexing");
    }
    // No optimizer: its nonzero-contraction masks would be stale after a
    // direction/identity substitution and can erase derivatives at zero.
    t.shell(intor,shells,size_t(components)*di*dj*dk,block,cache,op==6 ? 3:2);
  }
  void directional(const double* left, const double* right, double* output) {
    shell_pairs([&](int i,int j,int k) {
      int di,dj,dk;
      evaluate(i,j,k,left,right,-1,di,dj,dk);
      const size_t size=size_t(di)*dj*dk;
      for (int c=0; c<components; ++c) for (int r=0; r<dk; ++r)
        for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p) {
          int a=t.ao[i]+p,b=t.ao[j]+q;
          if (op==6 && a<b) continue;
          output[integral_index(c,a,b,k<0 ? 0:t.ao[k]+r)]+=
            block[c*size+(size_t(r)*dj+q)*di+p];
        }
    });
  }
  void transpose(const double* direction, const double* cotangent, double* output) {
    shell_pairs([&](int i,int j,int k) {
      for (int leg=0; leg<2; ++leg) {
        int di,dj,dk;
        evaluate(i,j,k,leg==0 ? nullptr:direction,leg==1 ? nullptr:direction,
                 leg,di,dj,dk);
        const int shell=leg==0 ? i:j;
        const int np=primitives[shell],nc=contractions[shell],na=angular[shell];
        const int other=leg==0 ? dj:di;
        const size_t size=size_t(di)*dj*dk;
        for (int c=0; c<components; ++c) for (int r=0; r<dk; ++r)
          for (int p=0; p<np; ++p) for (int angular_index=0; angular_index<na; ++angular_index)
            for (int contraction=0; contraction<nc; ++contraction) {
              double value=0.;
              for (int q=0; q<other; ++q) {
                int a=t.ao[i]+(leg==0 ? contraction*na+angular_index:q);
                int b=t.ao[j]+(leg==1 ? contraction*na+angular_index:q);
                if (op==6 && a<b) continue;
                const int bp=leg==0 ? p*na+angular_index:q;
                const int bq=leg==1 ? p*na+angular_index:q;
                value+=block[c*size+(size_t(r)*dj+bq)*di+bp]
                      *cotangent[integral_index(c,a,b,k<0 ? 0:t.ao[k]+r)];
              }
              output[offsets[shell]+size_t(contraction)*np+p]+=value;
            }
      }
    });
  }
};

template<bool transpose>
static ffi::Error CoefficientProduct(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis, ffi::BufferR1<ffi::F64> environment,
    ffi::BufferR1<ffi::F64> coefficients, ffi::Buffer<ffi::F64> vector,
    ffi::ResultBuffer<ffi::F64> result, int32_t op, int32_t cart, int32_t split) {
  try {
    CoefficientTables driver(atoms,basis,environment,coefficients,op,cart,split);
    if constexpr (transpose) {
      driver.check_integrals(vector); driver.check_vector(*result);
    } else {
      driver.check_vector(vector); driver.check_integrals(*result);
    }
    CoefficientTables::check_finite(vector.typed_data(),vector.element_count());
    double* output=result->typed_data();
    std::fill(output,output+result->element_count(),0.);
    if constexpr (transpose) parallel_vjp(driver,output,[&](auto& local,double* out) {
      local.transpose(nullptr,vector.typed_data(),out);
    });
    else {
      driver.directional(vector.typed_data(),nullptr,output);
      driver.directional(nullptr,vector.typed_data(),output);
    }
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

template<bool transpose>
static ffi::Error CoefficientHessianProduct(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis, ffi::BufferR1<ffi::F64> environment,
    ffi::BufferR1<ffi::F64> coefficients, ffi::Buffer<ffi::F64> direction,
    ffi::Buffer<ffi::F64> vector, ffi::ResultBuffer<ffi::F64> result,
    int32_t op, int32_t cart, int32_t split) {
  try {
    CoefficientTables driver(atoms,basis,environment,coefficients,op,cart,split);
    driver.check_vector(direction);
    if constexpr (transpose) {
      driver.check_integrals(vector); driver.check_vector(*result);
    } else {
      driver.check_vector(vector); driver.check_integrals(*result);
    }
    CoefficientTables::check_finite(direction.typed_data(),direction.element_count());
    CoefficientTables::check_finite(vector.typed_data(),vector.element_count());
    double* output=result->typed_data();
    std::fill(output,output+result->element_count(),0.);
    if constexpr (transpose) parallel_vjp(driver,output,[&](auto& local,double* out) {
      local.transpose(direction.typed_data(),vector.typed_data(),out);
    });
    else {
      driver.directional(direction.typed_data(),vector.typed_data(),output);
      driver.directional(vector.typed_data(),direction.typed_data(),output);
    }
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch (const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFCoefficientJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFCoefficientVJP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFCoefficientHessianJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFCoefficientHessianVJP(XLA_FFI_CallFrame*);
#define COEFFICIENT_ARGS ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>() \
    .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>() \
    .Arg<ffi::BufferR1<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>()
#define COEFFICIENT_RESULT .Ret<ffi::Buffer<ffi::F64>>() \
    .Attr<int32_t>("operator").Attr<int32_t>("cart").Attr<int32_t>("split")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFCoefficientJVP,CoefficientProduct<false>,
    COEFFICIENT_ARGS COEFFICIENT_RESULT);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFCoefficientVJP,CoefficientProduct<true>,
    COEFFICIENT_ARGS COEFFICIENT_RESULT);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFCoefficientHessianJVP,CoefficientHessianProduct<false>,
    COEFFICIENT_ARGS .Arg<ffi::Buffer<ffi::F64>>() COEFFICIENT_RESULT);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFCoefficientHessianVJP,CoefficientHessianProduct<true>,
    COEFFICIENT_ARGS .Arg<ffi::Buffer<ffi::F64>>() COEFFICIENT_RESULT);
