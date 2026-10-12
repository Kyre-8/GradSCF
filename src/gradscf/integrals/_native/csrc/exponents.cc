// Analytic Gaussian exponent products: d exp(-alpha*r^2)/d alpha=-r^2 exp(-alpha*r^2).
// Cartesian angular raising is reduced to the original shell before its
// original Cartesian/spherical transformation. No finite differences or Jacobian.
#include "tables.h"
#include "parallel.h"
#include <array>
extern "C" {
extern CINTIntegralFunction int3c2e_cart;
double *CINTc2s_ket_sph(double*, int, double*, int);
double CINTcommon_fac_sp(int);
}

static int ncart(int l) { return (l+1)*(l+2)/2; }
static std::vector<std::array<int,3>> powers(int l) {
  std::vector<std::array<int,3>> out;
  for (int x=l; x>=0; --x) for (int y=l-x; y>=0; --y)
    out.push_back({x,y,l-x-y});
  return out;
}

struct ExponentTables {
  IntegralTables t;
  int first_shell=0,shell_stride=1;
  int op,cart,active,original_shells,shadow,coefficient_ptr,n,m,components;
  size_t count=0,pairs;
  std::vector<size_t> offsets;
  std::array<std::vector<double>,13> transformations;
  std::vector<double> block,cache,reduced,spherical_scratch;
  std::array<std::vector<std::array<int,3>>,11> raise_indices;
  Intor intor;

  ExponentTables(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
                 ffi::BufferR1<ffi::F64> env,int operation,int representation,int split)
      :t(atoms,basis,env,operation==6 ? 4:operation,representation),op(operation),cart(representation) {
    if (op!=6 && (op<0 || op>3)) throw std::invalid_argument("Unsupported exponent integral operator");
    if (op==6 ? split<1 || split>=t.nbas : split!=0)
      throw std::invalid_argument("Invalid exponent orbital/auxiliary split");
    original_shells=t.nbas;active=op==6 ? split:t.nbas;
    n=t.ao[active];m=t.n-n;pairs=size_t(n)*(n+1)/2;components=op==3 ? 3:1;
    for (int s=0;s<active;++s) {
      if (t.bas[s*BAS_SLOTS+ANG_OF]>10)
        throw std::invalid_argument("Exponent angular raising requires orbital l <= 10");
      offsets.push_back(count);count+=t.bas[s*BAS_SLOTS+NPRIM_OF];
    }
    if (t.env.size()+64>=size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Exponent workspace exceeds int32 indexing");
    shadow=t.nbas++;coefficient_ptr=int(t.env.size());
    t.bas.resize(size_t(t.nbas)*BAS_SLOTS);t.env.resize(t.env.size()+64);
    Intor functions[]={int1e_ovlp_cart,int1e_kin_cart,int1e_nuc_cart,int1e_r_cart};
    intor=op==6 ? int3c2e_cart:functions[op];
  }
  void check_integrals(const ffi::Buffer<ffi::F64>& x) const {
    if (op==6) {
      if (x.dimensions().size()!=2 || x.dimensions()[0]!=m || x.dimensions()[1]!=pairs)
        throw std::invalid_argument("Invalid exponent packed integral shape");
    } else t.check_shape(x,op);
  }
  void check_vector(const ffi::Buffer<ffi::F64>& x) const {
    if (x.dimensions().size()!=1 || x.element_count()!=count)
      throw std::invalid_argument("Invalid exponent vector shape");
  }
  const std::vector<double>& transform(int l) {
    auto& matrix=transformations[l];
    if (matrix.empty()) {
      int nc=ncart(l),ns=2*l+1;
      std::vector<double> identity(size_t(nc)*nc,0.),out(size_t(nc)*ns);
      for (int i=0;i<nc;++i) identity[size_t(i)*nc+i]=1.;
      double* result=CINTc2s_ket_sph(out.data(),nc,identity.data(),l);
      matrix.assign(result,result+size_t(nc)*ns);
    }
    return matrix;
  }
  void spherical(std::array<int,3>& dims,const std::array<int,3>& shells,int legs) {
    size_t stride=1;
    for (int axis=0;axis<legs;++axis) {
      int l=t.bas[shells[axis]*BAS_SLOTS+ANG_OF],nc=ncart(l),ns=2*l+1;
      int contractions=t.bas[shells[axis]*BAS_SLOTS+NCTR_OF];
      int old=dims[axis],next=contractions*ns;
      size_t outer=reduced.size()/(stride*old);
      spherical_scratch.assign(outer*stride*next,0.);
      auto& out=spherical_scratch;
      const auto& matrix=transform(l);
      for (size_t o=0;o<outer;++o) for (int c=0;c<contractions;++c)
        for (int s=0;s<ns;++s) for (int p=0;p<nc;++p) {
          double weight=matrix[size_t(s)*nc+p];
          if (weight==0.) continue;
          for (size_t q=0;q<stride;++q)
            out[(o*next+c*ns+s)*stride+q]+=weight*reduced[(o*old+c*nc+p)*stride+q];
        }
      reduced.swap(out);dims[axis]=next;stride*=next;
    }
  }
  std::array<int,3> evaluate(int i,int j,int k,int leg,int primitive) {
    const std::array<int,3> original={i,j,k};
    int s=original[leg];const int* source=t.bas.data()+s*BAS_SLOTS;
    const int l=source[ANG_OF],np=source[NPRIM_OF],nc=source[NCTR_OF];
    int* row=t.bas.data()+shadow*BAS_SLOTS;
    std::copy(source,source+BAS_SLOTS,row);
    row[ANG_OF]=l+2;row[NPRIM_OF]=1;row[PTR_EXP]+=primitive;row[PTR_COEFF]=coefficient_ptr;
    const double ratio=CINTcommon_fac_sp(l)/CINTcommon_fac_sp(l+2);
    for (int c=0;c<nc;++c)
      t.env[coefficient_ptr+c]=ratio*t.env[source[PTR_COEFF]+c*np+primitive];
    std::array<int,3> shells=original; shells[leg]=shadow;
    const int legs=op==6 ? 3:2;
    std::array<int,3> dims={1,1,1},raised={1,1,1};
    int64_t workspace=components;
    for (int a=0;a<legs;++a) {
      const int* base=t.bas.data()+original[a]*BAS_SLOTS;
      dims[a]=ncart(base[ANG_OF])*base[NCTR_OF];
      const int* sh=t.bas.data()+shells[a]*BAS_SLOTS;
      raised[a]=ncart(sh[ANG_OF])*sh[NCTR_OF];workspace*=raised[a];
    }
    if ((op==6 ? 3*workspace:workspace)>=std::numeric_limits<int>::max())
      throw std::invalid_argument("Exponent raised Cartesian cache exceeds int32 indexing");
    t.shell(intor,shells.data(),size_t(workspace),block,cache,legs);
    auto& index=raise_indices[l];
    if (index.empty()) {
      auto lower=powers(l),higher=powers(l+2);
      index.resize(lower.size());
      for (size_t p=0;p<lower.size();++p) for (int axis=0;axis<3;++axis) {
        auto monomial=lower[p];monomial[axis]+=2;
        index[p][axis]=int(std::find(higher.begin(),higher.end(),monomial)-higher.begin());
      }
    }
    size_t size=size_t(dims[0])*dims[1]*dims[2];
    reduced.assign(size*components,0.);
    for (int c=0;c<components;++c) for (int r=0;r<dims[2];++r)
      for (int q=0;q<dims[1];++q) for (int p=0;p<dims[0];++p) {
        int a=leg==0 ? p:q;int contraction=a/ncart(l),angular=a%ncart(l);
        for (int axis=0;axis<3;++axis) {
          int shifted=contraction*ncart(l+2)+index[angular][axis];
          int pp=leg==0 ? shifted:p,qq=leg==1 ? shifted:q;
          reduced[c*size+(size_t(r)*dims[1]+q)*dims[0]+p]-=
            block[((size_t(c)*raised[2]+r)*raised[1]+qq)*raised[0]+pp];
        }
      }
    if (!cart) spherical(dims,original,legs);
    return dims;
  }
  template<bool transpose>
  void apply(const double* input,double* output) {
    for (int i=first_shell;i<active;i+=shell_stride) for (int j=0;j<(op==6 ? i+1:active);++j)
      for (int kk=op==6 ? active:0;kk<(op==6 ? original_shells:1);++kk)
        for (int leg=0;leg<2;++leg) {
          int shell=leg==0 ? i:j,np=t.bas[shell*BAS_SLOTS+NPRIM_OF];
          for (int primitive=0;primitive<np;++primitive) {
            size_t parameter=offsets[shell]+primitive;
            if constexpr (!transpose) if (input[parameter]==0.) continue;
            auto d=evaluate(i,j,op==6 ? kk:-1,leg,primitive);
            double sum=0.;size_t size=size_t(d[0])*d[1]*d[2];
            for (int c=0;c<components;++c) for (int r=0;r<d[2];++r)
              for (int q=0;q<d[1];++q) for (int p=0;p<d[0];++p) {
                int a=t.ao[i]+p,b=t.ao[j]+q;if (op==6 && a<b) continue;
                size_t target=op==6 ? size_t(t.ao[kk]+r-n)*pairs+size_t(a)*(a+1)/2+b
                                            : (size_t(c)*n+a)*n+b;
                double derivative=reduced[c*size+(size_t(r)*d[1]+q)*d[0]+p];
                if constexpr (transpose) sum+=input[target]*derivative;
                else output[target]+=input[parameter]*derivative;
              }
            if constexpr (transpose) output[parameter]+=sum;
          }
        }
  }
};

template<bool transpose>
static ffi::Error ExponentProduct(ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
    ffi::BufferR1<ffi::F64> environment,ffi::Buffer<ffi::F64> vector,
    ffi::ResultBuffer<ffi::F64> result,int32_t op,int32_t cart,int32_t split) {
  try {
    ExponentTables driver(atoms,basis,environment,op,cart,split);
    if constexpr (transpose) {driver.check_integrals(vector);driver.check_vector(*result);}
    else {driver.check_vector(vector);driver.check_integrals(*result);}
    for (size_t i=0;i<vector.element_count();++i)
      if (!std::isfinite(vector.typed_data()[i])) throw std::invalid_argument("Nonfinite exponent derivative vector");
    std::fill(result->typed_data(),result->typed_data()+result->element_count(),0.);
    if constexpr (transpose) parallel_vjp(driver,result->typed_data(),[&](auto& local,double* out) {
      local.template apply<true>(vector.typed_data(),out);
    });
    else driver.apply<false>(vector.typed_data(),result->typed_data());
    return ffi::Error::Success();
  } catch (const std::invalid_argument& e) {return ffi::Error::InvalidArgument(e.what());}
    catch (const std::exception& e) {return ffi::Error::Internal(e.what());}
}
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFExponentJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFExponentVJP(XLA_FFI_CallFrame*);
#define EXPONENT_BIND ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>() \
 .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>() \
 .Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() \
 .Attr<int32_t>("operator").Attr<int32_t>("cart").Attr<int32_t>("split")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFExponentJVP,ExponentProduct<false>,EXPONENT_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFExponentVJP,ExponentProduct<true>,EXPONENT_BIND);
