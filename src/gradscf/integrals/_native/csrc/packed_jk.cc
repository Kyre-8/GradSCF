// Packed-ERI J/K and its stable asynchronous FFI entry point.
#include "coulomb_shells.h"
#include <atomic>
#include <memory>

static ffi::Future PackedJK(ffi::ThreadPool pool, ffi::Buffer<ffi::F64> eri,
    ffi::BufferR3<ffi::F64> density, ffi::ResultBufferR4<ffi::F64> result,
    int32_t with_j,int32_t with_k) {
  ffi::Promise promise;
  ffi::Future future(promise);
  try {
    if ((with_j!=0 && with_j!=1) || (with_k!=0 && with_k!=1) || !(with_j || with_k))
      throw std::invalid_argument("Packed J/K requires at least one selected operator");
    const int nb=density.dimensions()[0],n=density.dimensions()[1];
    const size_t nn=size_t(n)*n,np=size_t(n)*(n+1)/2;
    bool s8=eri.dimensions().size()==1;
    if (n<1 || density.dimensions()[2]!=n ||
        (s8 ? eri.element_count()!=np*(np+1)/2 :
          (eri.dimensions().size()!=2 || eri.dimensions()[0]!=np || eri.dimensions()[1]!=np)) ||
        result->dimensions()[0]!=with_j+with_k || result->dimensions()[1]!=nb || result->dimensions()[2]!=n || result->dimensions()[3]!=n)
      throw std::invalid_argument("Packed J/K dimensions do not match");
    const double* integrals=eri.typed_data();const double* input=density.typed_data();
    double* output=result->typed_data();const size_t size=result->element_count();
    // Bounded worker-local accumulators avoid atomics in the integral loop.
    // A Future prevents blocking an XLA worker while waiting on its own pool.
    const int workers=np<1000 ? 1 : int(std::max<int64_t>(1,std::min<int64_t>(8,pool.num_threads())));
    auto partials=std::make_shared<std::vector<double>>(size*workers,0.);
    auto remaining=std::make_shared<std::atomic<int>>(workers);
    for (int worker=0;worker<workers;++worker) {
      // Quartet count grows quadratically in the outer pair index.
      size_t begin=size_t(np*std::sqrt(double(worker)/workers));
      size_t end=worker+1==workers ? np : size_t(np*std::sqrt(double(worker+1)/workers));
      auto task=[=]() mutable {
        double* local=partials->data()+size*worker;
        for (int a=0;a<n;++a) for (int b=0;b<=a;++b) {
          size_t ab=pair_index(a,b),offset=s8 ? ab*(ab+1)/2 : ab*np;
          if(ab<begin || ab>=end) continue;
          for (int c=0;c<=a;++c) for (int d=0;d<=(c==a ? b:c);++d) {
            size_t cd=pair_index(c,d);double v=integrals[offset+cd];
            for (int batch=0;batch<nb;++batch) {
              const double* dm=input+size_t(batch)*nn;
              if (with_j) {
                double* j=local+size_t(batch)*nn;
                double j_ab=v*(dm[size_t(c)*n+d]+(c!=d ? dm[size_t(d)*n+c]:0.));
                j[size_t(a)*n+b]+=j_ab;if(a!=b)j[size_t(b)*n+a]+=j_ab;
                if(ab!=cd) {
                  double j_cd=v*(dm[size_t(a)*n+b]+(a!=b ? dm[size_t(b)*n+a]:0.));
                  j[size_t(c)*n+d]+=j_cd;if(c!=d)j[size_t(d)*n+c]+=j_cd;
                }
              }
              if (with_k) {
                double* k=local+size_t(with_j*nb+batch)*nn;
                k[size_t(a)*n+c]+=v*dm[size_t(b)*n+d];
                if(c!=d)k[size_t(a)*n+d]+=v*dm[size_t(b)*n+c];
                if(a!=b)k[size_t(b)*n+c]+=v*dm[size_t(a)*n+d];
                if(a!=b && c!=d)k[size_t(b)*n+d]+=v*dm[size_t(a)*n+c];
                if(ab!=cd) {
                  k[size_t(c)*n+a]+=v*dm[size_t(d)*n+b];
                  if(a!=b)k[size_t(c)*n+b]+=v*dm[size_t(d)*n+a];
                  if(c!=d)k[size_t(d)*n+a]+=v*dm[size_t(c)*n+b];
                  if(a!=b && c!=d)k[size_t(d)*n+b]+=v*dm[size_t(c)*n+a];
                }
              }
            }
          }
        }
        if(remaining->fetch_sub(1,std::memory_order_acq_rel)==1) {
          // Fixed reduction order keeps a fixed worker count reproducible.
          std::copy(partials->begin(),partials->begin()+size,output);
          for(int w=1;w<workers;++w) for(size_t i=0;i<size;++i)
            output[i]+=(*partials)[size*w+i];
          promise.SetAvailable();
        }
      };
      if(workers==1) task(); else pool.Schedule(std::move(task));
    }
  } catch(const std::exception& e) {promise.SetError(ffi::Error::Internal(e.what()));}
  return future;
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFPackedJK(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFPackedJK,PackedJK,
    ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().Arg<ffi::Buffer<ffi::F64>>()
      .Arg<ffi::BufferR3<ffi::F64>>().Ret<ffi::BufferR4<ffi::F64>>()
      .Attr<int32_t>("with_j").Attr<int32_t>("with_k"));
