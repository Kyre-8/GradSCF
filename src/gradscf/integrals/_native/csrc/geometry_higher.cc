// Cubic/quartic coordinate products: one/two exact Gaussian center derivatives
// of the existing second-derivative kernels. No displaced integrals or jets.
#include "geometry_contraction.h"

extern "C" double CINTcommon_fac_sp(int);

namespace {
template<size_t N> struct IndexLess {
  bool operator()(const std::array<int,N>& a,const std::array<int,N>& b) const {
    for (size_t i=0;i<N;++i) if (a[i]!=b[i]) return a[i]<b[i];
    return false;
  }
};

using Angular=std::array<int,3>;
using Term=std::array<int,16>;  // Four Cartesian powers, then four alpha powers.
using Descriptor=std::array<int,8>;  // Shifted l and alpha power for each leg.

int cart_count(int l) { return (l+1)*(l+2)/2; }
int cart_index(const Angular& a) {
  const int m=a[1]+a[2];
  return m*(m+1)/2+a[2];  // libcint order: descending x, then descending y.
}

struct Reduction {
  std::array<int,4> original, shifted;
  double weight;
};

class ShiftedShells {
  IntegralTables& t;
  const int legs, extra_order;
  const size_t original_env_size;
  const double* extra;
  std::array<int,4> shadows{}, coefficient_ptr{};
  std::array<std::vector<Angular>,13> angular_powers;
  std::vector<double> raw;

  const std::vector<Angular>& powers(int l) {
    auto& values=angular_powers[l];
    if (values.empty()) for (int x=l;x>=0;--x) for (int y=l-x;y>=0;--y)
      values.push_back({x,y,l-x-y});
    return values;
  }

  int shadow(int leg,int original,int shifted_l,int alpha_power) {
    const int* source=t.bas.data()+original*BAS_SLOTS;
    int* target=t.bas.data()+shadows[leg]*BAS_SLOTS;
    std::copy(source,source+BAS_SLOTS,target);
    target[ANG_OF]=shifted_l;target[PTR_COEFF]=coefficient_ptr[leg];
    const int np=source[NPRIM_OF],nc=source[NCTR_OF];
    const double ratio=CINTcommon_fac_sp(source[ANG_OF])/CINTcommon_fac_sp(shifted_l);
    for (int c=0;c<nc;++c) for (int p=0;p<np;++p) {
      double value=t.env[source[PTR_COEFF]+c*np+p]*ratio;
      for (int k=0;k<alpha_power;++k) value*=t.env[source[PTR_EXP]+p];
      if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite shifted primitive coefficient");
      t.env[coefficient_ptr[leg]+c*np+p]=value;
    }
    return shadows[leg];
  }

public:
  ShiftedShells(IntegralTables& tables,int op,int order,const double* directions)
      :t(tables),legs(op==4 ? 4:2),extra_order(order-2),
       original_env_size(t.env.size()),extra(directions) {
    const int original=t.nbas;
    size_t capacity=0;
    for (int s=0;s<original;++s) {
      const int* row=t.bas.data()+s*BAS_SLOTS;
      if (row[ANG_OF]+2>12)
        throw std::invalid_argument("Cubic/quartic geometry shifts require original l <= 10");
      capacity=std::max(capacity,size_t(row[NPRIM_OF])*row[NCTR_OF]);
    }
    if (original>std::numeric_limits<int>::max()-legs ||
        t.env.size()+legs*capacity>=size_t(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Higher geometry shadow workspace exceeds int32 indexing");
    t.bas.resize(size_t(original+legs)*BAS_SLOTS);
    for (int leg=0;leg<legs;++leg) {
      shadows[leg]=original+leg;
      coefficient_ptr[leg]=int(original_env_size+leg*capacity);
    }
    t.env.resize(original_env_size+legs*capacity);
    t.nbas+=legs;
  }

  void operator()(Intor kernel,int* shells,size_t size,std::vector<double>& block,
                  std::vector<double>& cache,int nucleus) {
    std::array<int,4> counts{1,1,1,1},contractions{1,1,1,1},dims{1,1,1,1};
    std::array<const std::vector<Angular>*,4> monomials{};
    size_t angular_size=1,original_size=1,contraction_size=1;
    for (int leg=0;leg<legs;++leg) {
      const int* row=t.bas.data()+shells[leg]*BAS_SLOTS;
      counts[leg]=cart_count(row[ANG_OF]);contractions[leg]=row[NCTR_OF];
      dims[leg]=counts[leg]*contractions[leg];monomials[leg]=&powers(row[ANG_OF]);
      angular_size*=counts[leg];original_size*=dims[leg];contraction_size*=contractions[leg];
    }
    if (!original_size || size%original_size)
      throw std::invalid_argument("Invalid higher geometry shell component layout");
    const size_t components=size/original_size;
    block.assign(size,0.);
    double directions[2][4][3]{};
    for (int order=0;order<extra_order;++order) {
      const double* direction=extra+size_t(order)*original_env_size;
      for (int leg=0;leg<legs;++leg) for (int axis=0;axis<3;++axis) {
        // Each rinv term depends on A-C and B-C. For translation-invariant
        // S/T/ERI kernels eliminate the last AO center instead.
        const int reference=nucleus>=0 ? nucleus:t.coord_ptr(shells[legs-1]);
        directions[order][leg][axis]=direction[t.coord_ptr(shells[leg])+axis]-direction[reference+axis];
      }
    }

    std::map<Descriptor,std::vector<Reduction>,IndexLess<8>> groups;
    for (size_t flat=0;flat<angular_size;++flat) {
      size_t rest=flat;
      std::array<int,4> original{};
      Term initial{};
      for (int leg=0;leg<legs;++leg) {
        original[leg]=int(rest%counts[leg]);rest/=counts[leg];
        const Angular& a=(*monomials[leg])[original[leg]];
        for (int axis=0;axis<3;++axis) initial[3*leg+axis]=a[axis];
      }
      std::map<Term,double,IndexLess<16>> terms{{initial,1.}};
      for (int order=0;order<extra_order;++order) {
        std::map<Term,double,IndexLess<16>> next;
        for (const auto& entry:terms) for (int leg=0;leg<legs;++leg)
          for (int axis=0;axis<3;++axis) {
            const double direction=directions[order][leg][axis];
            if (direction==0. || entry.second==0.) continue;
            const int index=3*leg+axis;
            Term raised=entry.first;raised[index]++;raised[12+leg]++;
            next[raised]+=2.*direction*entry.second;
            const int power=entry.first[index];
            if (power) {
              Term lowered=entry.first;lowered[index]--;
              next[lowered]-=power*direction*entry.second;
            }
          }
        terms.swap(next);
      }
      for (const auto& entry:terms) {
        if (entry.second==0.) continue;
        Descriptor descriptor{};
        Reduction reduction{original,{},entry.second};
        for (int leg=0;leg<legs;++leg) {
          Angular a{entry.first[3*leg],entry.first[3*leg+1],entry.first[3*leg+2]};
          const int l=a[0]+a[1]+a[2];
          if (l<0 || l>12) throw std::invalid_argument("Invalid shifted angular momentum");
          descriptor[2*leg]=l;descriptor[2*leg+1]=entry.first[12+leg];
          reduction.shifted[leg]=cart_index(a);
        }
        groups[descriptor].push_back(reduction);
      }
    }

    for (const auto& group:groups) {
      int shifted_shells[4]{};
      std::array<int,4> shifted_counts{1,1,1,1},shifted_dims{1,1,1,1};
      size_t shifted_size=1;
      for (int leg=0;leg<legs;++leg) {
        shifted_shells[leg]=shadow(leg,shells[leg],group.first[2*leg],group.first[2*leg+1]);
        shifted_counts[leg]=cart_count(group.first[2*leg]);
        shifted_dims[leg]=shifted_counts[leg]*contractions[leg];
        if (shifted_size>size_t(std::numeric_limits<int>::max())/shifted_dims[leg])
          throw std::invalid_argument("Higher geometry Cartesian workspace exceeds int32 indexing");
        shifted_size*=shifted_dims[leg];
      }
      if (components*shifted_size>=size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Higher geometry component workspace exceeds int32 indexing");
      // Reuse scratch storage, not the 2/3-leg query cache. All four ERI legs
      // would have to enter its key before four-leg cache reuse is valid.
      t.shell(kernel,shifted_shells,components*shifted_size,raw,cache);
      for (const Reduction& entry:group.second) for (size_t contraction=0;contraction<contraction_size;++contraction) {
        size_t rest=contraction,original_index=0,shifted_index=0,original_stride=1,shifted_stride=1;
        for (int leg=0;leg<legs;++leg) {
          const int column=int(rest%contractions[leg]);rest/=contractions[leg];
          original_index+=(column*counts[leg]+entry.original[leg])*original_stride;
          shifted_index+=(column*shifted_counts[leg]+entry.shifted[leg])*shifted_stride;
          original_stride*=dims[leg];shifted_stride*=shifted_dims[leg];
        }
        for (size_t component=0;component<components;++component)
          block[component*original_size+original_index]+=
            entry.weight*raw[component*shifted_size+shifted_index];
      }
    }
  }
};
}  // namespace

template<bool transpose>
static ffi::Error GeometryHigher(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> environment,
    ffi::Buffer<ffi::F64> direction,ffi::Buffer<ffi::F64> vector,
    ffi::BufferR2<ffi::F64> extra,ffi::ResultBuffer<ffi::F64> result,
    int32_t op,int32_t cart,int32_t order) {
  try {
    if ((order!=3 && order!=4) || cart!=1 || (op!=0 && op!=1 && op!=2 && op!=4))
      throw std::invalid_argument("Higher geometry supports Cartesian S/T/nuclear/full-ERI orders 3 and 4 only");
    IntegralTables t(atoms,basis,environment,op,cart,true);
    if (t.env[PTR_RANGE_OMEGA]!=0.)
      throw std::invalid_argument("Third/fourth geometry derivatives currently require omega=0");
    const size_t env_size=t.env.size();
    std::vector<bool> allowed(env_size,false);
    for (int atom=0;atom<t.natm;++atom) for (int axis=0;axis<3;++axis)
      allowed[t.atm[atom*ATM_SLOTS+PTR_COORD]+axis]=true;
    auto check_direction=[&](const double* values) {
      for (size_t i=0;i<env_size;++i)
        if (!std::isfinite(values[i]) || (!allowed[i] && values[i]!=0.))
          throw std::invalid_argument("Higher geometry requires finite coordinate-only directions");
    };
    if (direction.dimensions().size()!=1 || direction.element_count()!=env_size ||
        extra.dimensions()[0]!=order-2 || extra.dimensions()[1]!=int64_t(env_size))
      throw std::invalid_argument("Invalid higher geometry direction shape");
    check_direction(direction.typed_data());
    for (int i=0;i<order-2;++i) check_direction(extra.typed_data()+size_t(i)*env_size);
    if constexpr (transpose) {
      t.check_shape(vector,op);
      if (result->dimensions().size()!=1 || result->element_count()!=env_size)
        throw std::invalid_argument("Invalid higher geometry adjoint shape");
      for (size_t i=0;i<vector.element_count();++i) if (!std::isfinite(vector.typed_data()[i]))
        throw std::invalid_argument("Higher geometry cotangent must be finite");
    } else {
      t.check_shape(*result,op);
      if (vector.dimensions().size()!=1 || vector.element_count()!=env_size)
        throw std::invalid_argument("Invalid higher geometry direction shape");
      check_direction(vector.typed_data());
    }
    double* output=result->typed_data();
    std::fill(output,output+result->element_count(),0.);
    ShiftedShells evaluator(t,op,order,extra.typed_data());
    contract_geometry_hessian<transpose>(t,op,cart,direction.typed_data(),vector.typed_data(),output,evaluator);
    for (size_t i=0;i<result->element_count();++i) if (!std::isfinite(output[i]))
      throw std::invalid_argument("Higher geometry derivative is nonfinite");
    return ffi::Error::Success();
  } catch (const std::invalid_argument& error) { return ffi::Error::InvalidArgument(error.what()); }
    catch (const std::exception& error) { return ffi::Error::Internal(error.what()); }
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFGeometryHigherJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFGeometryHigherVJP(XLA_FFI_CallFrame*);
#define GEOMETRY_HIGHER_BIND ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>() \
    .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>() \
    .Arg<ffi::Buffer<ffi::F64>>().Arg<ffi::Buffer<ffi::F64>>() \
    .Arg<ffi::BufferR2<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() \
    .Attr<int32_t>("operator").Attr<int32_t>("cart").Attr<int32_t>("order")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFGeometryHigherJVP,GeometryHigher<false>,GEOMETRY_HIGHER_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFGeometryHigherVJP,GeometryHigher<true>,GEOMETRY_HIGHER_BIND);
