// Shell-direct J/K with disjoint output-row ownership.
#include "coulomb_shells.h"
#include <atomic>
#include <memory>
#include <string>

static ffi::Future DirectJK(ffi::ThreadPool pool,
    ffi::BufferR2<ffi::S32> atoms,ffi::BufferR2<ffi::S32> basis,
    ffi::BufferR1<ffi::F64> env,ffi::BufferR3<ffi::F64> density,
    ffi::ResultBufferR4<ffi::F64> result,int32_t cart,double cutoff,int32_t requested_workers,
    int32_t with_j,int32_t with_k) {
  ffi::Promise promise;
  ffi::Future future(promise);
  try {
    if (requested_workers<1) throw std::invalid_argument("Direct J/K workers must be positive");
    if ((with_j!=0 && with_j!=1) || (with_k!=0 && with_k!=1) || !(with_j || with_k))
      throw std::invalid_argument("Direct J/K requires at least one selected operator");
    auto tables=std::make_shared<IntegralTables>(atoms,basis,env,4,cart,true);
    const int n=tables->n,nb=density.dimensions()[0];
    if (density.dimensions()[1]!=n || density.dimensions()[2]!=n ||
        result->dimensions()[0]!=with_j+with_k || result->dimensions()[1]!=nb || result->dimensions()[2]!=n || result->dimensions()[3]!=n)
      throw std::invalid_argument("Direct J/K shape mismatch");
    const double* d=density.typed_data();double* out=result->typed_data();const size_t nn=size_t(n)*n;
    for (size_t i=0;i<density.element_count();++i) if (!std::isfinite(d[i])) throw std::invalid_argument("Nonfinite density");
    const size_t size=result->element_count();
    const int workers=int(std::max<int64_t>(1,std::min<int64_t>(
        requested_workers,std::min<int64_t>(pool.num_threads(),tables->nbas))));
    std::fill(out,out+size,0.);
    // Ordinary int2e kernels read CINTOpt. One call-scoped owner retains its
    // primitive-pair data until every worker has completed its output rows.
    auto optimizer=std::make_shared<LocalOptimizer>(*tables,4);
    auto bounds=std::make_shared<std::vector<double>>(
        eri_schwarz_bounds(*tables,cart,cutoff,optimizer->value));
    auto remaining=std::make_shared<std::atomic<int>>(workers);
    auto errors=std::make_shared<std::vector<std::string>>(workers);
    for (int worker=0;worker<workers;++worker) {
      auto task=[=]() mutable {
        try {
          // Tables and shell scratch are private; optimizer data is read-only.
          IntegralTables local_tables=*tables;
          const int shell_begin=worker*tables->nbas/workers;
          const int shell_end=(worker+1)*tables->nbas/workers;
          const int row_begin=tables->ao[shell_begin],row_end=tables->ao[shell_end];
          auto owns=[&](int row) {return row_begin<=row && row<row_end;};
          unique_eri(local_tables,cart,cutoff,[&](int a,int b,int c,int e,double v) {
            const size_t ab=pair_index(a,b),ce=pair_index(c,e);
            const bool own_a=owns(a),own_b=owns(b),own_c=owns(c),own_e=owns(e);
            for (int batch=0;batch<nb;++batch) {
              const double* dm=d+size_t(batch)*nn;
              if (with_j) {
                double* j=out+size_t(batch)*nn;
                if (own_a || own_b) {
                  double j_ab=v*(dm[size_t(c)*n+e]+(c!=e ? dm[size_t(e)*n+c]:0.));
                  if(own_a)j[size_t(a)*n+b]+=j_ab;
                  if(a!=b && own_b)j[size_t(b)*n+a]+=j_ab;
                }
                if(ab!=ce && (own_c || own_e)) {
                  double j_ce=v*(dm[size_t(a)*n+b]+(a!=b ? dm[size_t(b)*n+a]:0.));
                  if(own_c)j[size_t(c)*n+e]+=j_ce;
                  if(c!=e && own_e)j[size_t(e)*n+c]+=j_ce;
                }
              }
              if (with_k) {
                double* k=out+size_t(with_j*nb+batch)*nn;
                if(own_a)k[size_t(a)*n+c]+=v*dm[size_t(b)*n+e];
                if(c!=e && own_a)k[size_t(a)*n+e]+=v*dm[size_t(b)*n+c];
                if(a!=b && own_b)k[size_t(b)*n+c]+=v*dm[size_t(a)*n+e];
                if(a!=b && c!=e && own_b)k[size_t(b)*n+e]+=v*dm[size_t(a)*n+c];
                if(ab!=ce) {
                  if(own_c)k[size_t(c)*n+a]+=v*dm[size_t(e)*n+b];
                  if(a!=b && own_c)k[size_t(c)*n+b]+=v*dm[size_t(e)*n+a];
                  if(c!=e && own_e)k[size_t(e)*n+a]+=v*dm[size_t(c)*n+b];
                  if(a!=b && c!=e && own_e)k[size_t(e)*n+b]+=v*dm[size_t(c)*n+a];
                }
              }
            }
          },shell_begin,shell_end,bounds.get(),optimizer->value);
        } catch(const std::exception& e) {
          (*errors)[worker]=e.what();
        }
        if (remaining->fetch_sub(1,std::memory_order_acq_rel)==1) {
          for (const auto& error:*errors) if (!error.empty()) {
            promise.SetError(ffi::Error::Internal(error));
            return;
          }
          // Output rows are disjoint; no full worker matrices or reduction.
          promise.SetAvailable();
        }
      };
      // Do not block an XLA worker waiting for work on its own thread pool.
      if (workers==1) task(); else pool.Schedule(std::move(task));
    }
  } catch(const std::exception& e) {promise.SetError(ffi::Error::Internal(e.what()));}
  return future;
}

extern "C" __attribute__((visibility("default"))) XLA_FFI_Error* GradSCFDirectJK(XLA_FFI_CallFrame*);
XLA_FFI_DEFINE_HANDLER_SYMBOL(GradSCFDirectJK,DirectJK,
    ffi::Ffi::Bind().Ctx<ffi::ThreadPool>().Arg<ffi::BufferR2<ffi::S32>>().Arg<ffi::BufferR2<ffi::S32>>()
      .Arg<ffi::BufferR1<ffi::F64>>().Arg<ffi::BufferR3<ffi::F64>>().Ret<ffi::BufferR4<ffi::F64>>()
      .Attr<int32_t>("cart").Attr<double>("cutoff").Attr<int32_t>("workers")
      .Attr<int32_t>("with_j").Attr<int32_t>("with_k"));
