// Coordinate products of J/K, contracted before leaving each ERI shell block.
#include "eri_geometry_entries.h"
#include "geometry_contraction.h"
#include <atomic>
#include <memory>
#include <string>

template<bool transpose,class Output>
static void DirectJKGeometryCompute(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> environment,
    ffi::BufferR3<ffi::F64> density,ffi::BufferR1<ffi::F64> direction,
    ffi::Buffer<ffi::F64> vector,Output result,
    int32_t order,int32_t cart,int32_t with_j,int32_t with_k,
    int first_shell=0,int last_shell=-1) {
    if ((order!=1 && order!=2) || (with_j!=0 && with_j!=1) ||
        (with_k!=0 && with_k!=1) || !(with_j || with_k))
      throw std::invalid_argument("Invalid direct J/K geometry product");
    IntegralTables t(atoms,basis,environment,4,cart,true);
    const int n=t.n,nb=density.dimensions()[0],nops=with_j+with_k;
    const size_t nn=size_t(n)*n;
    if (density.dimensions()[1]!=n || density.dimensions()[2]!=n)
      throw std::invalid_argument("Direct J/K geometry density shape mismatch");
    auto check_jk=[&](const auto& buffer) {
      if (buffer.dimensions().size()!=4 || buffer.dimensions()[0]!=nops ||
          buffer.dimensions()[1]!=nb || buffer.dimensions()[2]!=n || buffer.dimensions()[3]!=n)
        throw std::invalid_argument("Direct J/K geometry matrix shape mismatch");
    };
    auto check_finite=[](const auto& buffer) {
      for (size_t i=0;i<buffer.element_count();++i)
        if (!std::isfinite(buffer.typed_data()[i]))
          throw std::invalid_argument("Direct J/K geometry input must be finite");
    };
    std::vector<bool> allowed(t.env.size(),false);
    for (int atom=0;atom<t.natm;++atom) for (int axis=0;axis<3;++axis)
      allowed[t.atm[atom*ATM_SLOTS+PTR_COORD]+axis]=true;
    auto check_direction=[&](const auto& buffer) {
      if (buffer.dimensions().size()!=1 || buffer.element_count()!=t.env.size())
        throw std::invalid_argument("Direct J/K geometry direction shape mismatch");
      check_finite(buffer);
      bool active=false;
      for (size_t i=0;i<t.env.size();++i) if (buffer.typed_data()[i]!=0.) {
        if (!allowed[i])
          throw std::invalid_argument("Direct J/K geometry supports coordinates only, not basis exponents/coefficients");
        active=true;
      }
      return active;
    };
    check_finite(density);
    double* output=result->typed_data();
    std::fill(output,output+result->element_count(),0.);
    bool active=true;
    if constexpr (transpose) {
      check_jk(vector);check_finite(vector);
      if (result->dimensions().size()!=1 || result->element_count()!=t.env.size())
        throw std::invalid_argument("Direct J/K geometry adjoint shape mismatch");
      if (order==2) active=check_direction(direction);
    } else {
      check_jk(*result);
      active=check_direction(direction);
      if (order==2) active=check_direction(vector) && active;
    }
    // Numerical zero directions also arise from custom_root closure constants.
    // This shortcut is inside the native linear operation, not a nonlinear AD rule.
    if (!active) return;
    const double* dm=density.typed_data();
    const double* u=direction.typed_data();
    const double* input=vector.typed_data();
    auto consume=[&](size_t index,int coordinate,double value) {
      const int d=int(index%n);index/=n;
      const int c=int(index%n);index/=n;
      const int b=int(index%n),a=int(index/n);
      double sum=0.;
      for (int batch=0;batch<nb;++batch) {
        const double* density_matrix=dm+size_t(batch)*nn;
        if constexpr (transpose) {
          if (with_j) sum+=input[size_t(batch)*nn+size_t(a)*n+b]*density_matrix[size_t(c)*n+d];
          if (with_k) sum+=input[size_t(with_j*nb+batch)*nn+size_t(a)*n+c]*density_matrix[size_t(b)*n+d];
        } else {
          if (with_j) output[size_t(batch)*nn+size_t(a)*n+b]+=value*density_matrix[size_t(c)*n+d];
          if (with_k) output[size_t(with_j*nb+batch)*nn+size_t(a)*n+c]+=value*density_matrix[size_t(b)*n+d];
        }
      }
      if constexpr (transpose) output[coordinate]+=value*sum;
    };
    if (order==1) {
      auto add=[&](size_t index,int coordinate,double value) {
        consume(index,coordinate,transpose ? value:value*u[coordinate]);
      };
      for_each_eri_geometry_entry(t,cart,add,nullptr,first_shell,last_shell);
    } else {
      auto evaluate=[&](Intor kernel,int* shells,size_t size,
                        std::vector<double>& block,std::vector<double>& cache,int) {
        t.shell(kernel,shells,size,block,cache);
      };
      auto add=[&](size_t index,int a,int b,double value) {
        consume(index,a,transpose ? value*u[b]:value*u[a]*input[b]);
      };
      for_each_geometry_hessian_entry(t,4,cart,evaluate,add,nullptr,first_shell,last_shell);
    }
}

// Retain the original v1 ABI and serial behavior for existing binaries/callers.
template<bool transpose>
static ffi::Error DirectJKGeometry(ffi::BufferR2<ffi::S32> atoms,
    ffi::BufferR2<ffi::S32> basis,ffi::BufferR1<ffi::F64> environment,
    ffi::BufferR3<ffi::F64> density,ffi::BufferR1<ffi::F64> direction,
    ffi::Buffer<ffi::F64> vector,ffi::ResultBuffer<ffi::F64> result,
    int32_t order,int32_t cart,int32_t with_j,int32_t with_k) {
  try {
    DirectJKGeometryCompute<transpose>(atoms,basis,environment,density,direction,
        vector,result,order,cart,with_j,with_k);
    return ffi::Error::Success();
  } catch(const std::invalid_argument& error) {
    return ffi::Error::InvalidArgument(error.what());
  } catch(const std::exception& error) {
    return ffi::Error::Internal(error.what());
  }
}

// A coordinate VJP returns only ENV, so worker-private reductions stay linear
// in the basis metadata. No worker owns a full J/K matrix or an AO ERI tensor.
static ffi::Future ParallelDirectJKGeometryVJP(ffi::ThreadPool pool,
    ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
    ffi::BufferR1<ffi::F64> environment,ffi::BufferR3<ffi::F64> density,
    ffi::BufferR1<ffi::F64> direction,ffi::Buffer<ffi::F64> vector,
    ffi::ResultBuffer<ffi::F64> result,int32_t order,int32_t cart,
    int32_t with_j,int32_t with_k,int32_t requested_workers) {
  ffi::Promise promise;
  ffi::Future future(promise);
  try {
    if (requested_workers<1)
      throw std::invalid_argument("Direct J/K geometry workers must be positive");
    const int shells=basis.dimensions()[0];
    const int workers=int(std::max<int64_t>(1,std::min<int64_t>(
        requested_workers,std::min<int64_t>(pool.num_threads(),shells))));
    const size_t size=result->element_count();
    auto partials=std::make_shared<std::vector<std::vector<double>>>(
        workers,std::vector<double>(size));
    struct Failure {bool invalid=false;std::string message;};
    auto errors=std::make_shared<std::vector<Failure>>(workers);
    auto remaining=std::make_shared<std::atomic<int>>(workers);
    for (int worker=0;worker<workers;++worker) {
      auto task=[=]() mutable {
        try {
          struct LocalOutput {
            ffi::Buffer<ffi::F64> layout;
            double* data;
            auto dimensions() const {return layout.dimensions();}
            auto element_count() const {return layout.element_count();}
            double* typed_data() const {return data;}
          } local{*result,(*partials)[worker].data()};
          DirectJKGeometryCompute<true>(atoms,basis,environment,density,direction,
              vector,&local,order,cart,with_j,with_k,
              worker*shells/workers,(worker+1)*shells/workers);
        } catch(const std::invalid_argument& error) {
          (*errors)[worker]={true,error.what()};
        } catch(const std::exception& error) {
          (*errors)[worker]={false,error.what()};
        }
        if (remaining->fetch_sub(1,std::memory_order_acq_rel)==1) {
          for (const auto& error:*errors) if (!error.message.empty()) {
            promise.SetError(error.invalid ? ffi::Error::InvalidArgument(error.message)
                                           : ffi::Error::Internal(error.message));
            return;
          }
          double* output=result->typed_data();
          std::fill(output,output+size,0.);
          // Fixed partition/reduction order makes each worker count repeatable.
          for (const auto& part:*partials)
            for (size_t i=0;i<size;++i) output[i]+=part[i];
          promise.SetAvailable();
        }
      };
      if (workers==1) task(); else pool.Schedule(std::move(task));
    }
  } catch(const std::invalid_argument& error) {
    promise.SetError(ffi::Error::InvalidArgument(error.what()));
  } catch(const std::exception& error) {
    promise.SetError(ffi::Error::Internal(error.what()));
  }
  return future;
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFDirectJKGeometryJVP(XLA_FFI_CallFrame*);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFDirectJKGeometryVJP(XLA_FFI_CallFrame*);
#define DIRECT_GEOMETRY_BIND ffi::Ffi::Bind().Arg<ffi::BufferR2<ffi::S32>>() \
    .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>() \
    .Arg<ffi::BufferR3<ffi::F64>>().Arg<ffi::BufferR1<ffi::F64>>() \
    .Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>() \
    .Attr<int32_t>("order").Attr<int32_t>("cart") \
    .Attr<int32_t>("with_j").Attr<int32_t>("with_k")
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJKGeometryJVP,DirectJKGeometry<false>,DIRECT_GEOMETRY_BIND);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJKGeometryVJP,DirectJKGeometry<true>,DIRECT_GEOMETRY_BIND);
extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFDirectJKGeometryVJPParallel(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJKGeometryVJPParallel,ParallelDirectJKGeometryVJP,
    ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().Arg<ffi::BufferR2<ffi::S32>>()
      .Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR1<ffi::F64>>()
      .Arg<ffi::BufferR3<ffi::F64>>().Arg<ffi::BufferR1<ffi::F64>>()
      .Arg<ffi::Buffer<ffi::F64>>().Ret<ffi::Buffer<ffi::F64>>()
      .Attr<int32_t>("order").Attr<int32_t>("cart")
      .Attr<int32_t>("with_j").Attr<int32_t>("with_k").Attr<int32_t>("workers"));
