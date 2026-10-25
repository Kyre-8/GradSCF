// Shared shell-local Coulomb traversal; no dense AO ERI allocation.
#pragma once
#include "tables.h"

extern "C" {
extern CINTIntegralFunction int2c2e_cart, int2c2e_sph, int3c2e_cart, int3c2e_sph;
extern CINTOptimizerFunction int2c2e_optimizer, int3c2e_optimizer;
}

static size_t pair_index(size_t a, size_t b) {
  if (a < b) std::swap(a,b);
  return a*(a+1)/2+b;
}

struct LocalOptimizer {
  CINTOpt* value=nullptr;
  bool owns=true;
  LocalOptimizer(IntegralTables& t, int centers, CINTOpt* prepared=nullptr)
      : value(prepared),owns(prepared==nullptr) {
    if (owns) {
      auto f=centers==4 ? int2e_optimizer : (centers==3 ? int3c2e_optimizer : int2c2e_optimizer);
      f(&value,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data());
    }
  }
  LocalOptimizer(const LocalOptimizer&)=delete;
  LocalOptimizer& operator=(const LocalOptimizer&)=delete;
  ~LocalOptimizer() { if (owns) CINTdel_optimizer(&value); }
};

using ERICacheSizes = std::map<std::array<int,17>,int>;

static void block(IntegralTables& t, Intor f, CINTOpt* opt, int* shells,
                  size_t count, std::vector<double>& data, std::vector<double>& cache,
                  ERICacheSizes* sizes=nullptr) {
  if (count > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Integral shell block exceeds int32 indexing");
  std::array<int,17> key{};
  int length=0;
  if (sizes) {
    // Only ordinary four-center callers opt in. With fixed ENV controls,
    // scratch size depends on the intor and shell shapes, not AO centers.
    key[0]=f==int2e_cart;
    for (int a=0;a<4;++a) {
      const int* row=t.bas.data()+shells[a]*BAS_SLOTS;
      key[1+4*a]=row[ANG_OF];key[2+4*a]=row[NPRIM_OF];
      key[3+4*a]=row[NCTR_OF];key[4+4*a]=row[KAPPA_OF];
    }
    auto found=sizes->find(key);
    if (found!=sizes->end()) length=found->second;
  }
  if (!length) {
    length=f(nullptr,nullptr,shells,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data(),opt,nullptr);
    if (length<=0) throw std::invalid_argument("libcint cache query failed");
    if (sizes) sizes->emplace(key,length);
  }
  data.assign(count,0.); cache.resize(length);
  f(data.data(),nullptr,shells,t.atm.data(),t.natm,t.bas.data(),t.nbas,t.env.data(),opt,cache.data());
}

static std::vector<double> eri_schwarz_bounds(IntegralTables& t, int cart, double cutoff,
                                              CINTOpt* prepared_optimizer=nullptr) {
  if (!std::isfinite(cutoff) || cutoff<0) throw std::invalid_argument("Invalid Schwarz cutoff");
  if (!cutoff) return {};
  Intor f=cart ? int2e_cart : int2e_sph;
  LocalOptimizer opt(t,4,prepared_optimizer);
  std::vector<double> data,cache,bounds(size_t(t.nbas)*t.nbas,0.);
  ERICacheSizes sizes;
  for (int i=0;i<t.nbas;++i) for (int j=0;j<=i;++j) {
    int sh[]={i,j,i,j};int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
    block(t,f,opt.value,sh,size_t(di)*dj*di*dj,data,cache,&sizes);
    double largest=0.;
    for (int p=0;p<di;++p) for (int q=0;q<dj;++q)
      largest=std::max(largest,std::abs(data[((size_t(q)*di+p)*dj+q)*di+p]));
    bounds[size_t(i)*t.nbas+j]=std::sqrt(largest);
  }
  return bounds;
}

template<class Consume>
static void unique_eri(IntegralTables& t, int cart, double cutoff, Consume consume,
                       int shell_begin=0, int shell_end=-1,
                       const std::vector<double>* prepared_bounds=nullptr,
                       CINTOpt* prepared_optimizer=nullptr) {
  if (!std::isfinite(cutoff) || cutoff<0) throw std::invalid_argument("Invalid Schwarz cutoff");
  if (shell_end<0) shell_end=t.nbas;
  if (shell_begin<0 || shell_begin>=shell_end || shell_end>t.nbas)
    throw std::invalid_argument("Invalid ERI output-shell range");
  Intor f=cart ? int2e_cart : int2e_sph;
  LocalOptimizer opt(t,4,prepared_optimizer);
  std::vector<double> data,cache,owned_bounds;
  if (!prepared_bounds) owned_bounds=eri_schwarz_bounds(t,cart,cutoff,opt.value);
  const auto& bounds=prepared_bounds ? *prepared_bounds : owned_bounds;
  if (cutoff>0 && bounds.size()!=size_t(t.nbas)*t.nbas)
    throw std::invalid_argument("Schwarz bounds do not match shell dimensions");
  const double largest_bound=cutoff>0 ? *std::max_element(bounds.begin(),bounds.end()) : 0.;
  ERICacheSizes sizes;
  auto owns=[&](int shell) {return shell_begin<=shell && shell<shell_end;};
  for (int i=shell_begin;i<t.nbas;++i) for (int j=0;j<=i;++j) {
    // This upper bound rejects only pairs whose every inner quartet already
    // fails the same basis-only Schwarz test used below.
    if (cutoff>0 && bounds[size_t(i)*t.nbas+j]*largest_bound<cutoff) continue;
    const bool outer_owned=owns(i) || owns(j);
    // Visit a quartet only if at least one of its four output shells is
    // owned. The retained i,j,k,l order is the original lexicographic order.
    for (int k=outer_owned ? 0:shell_begin;k<=i;++k) {
      const bool owned=outer_owned || owns(k);
      const int first=owned ? 0:shell_begin;
      const int last=owned ? k+1:std::min(k+1,shell_end);
      for (int l=first;l<last;++l) {
        if (cutoff>0 && bounds[size_t(i)*t.nbas+j]*bounds[size_t(k)*t.nbas+l]<cutoff) continue;
        const int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
        const int dk=t.ao[k+1]-t.ao[k],dl=t.ao[l+1]-t.ao[l];
        int sh[]={i,j,k,l};
        block(t,f,opt.value,sh,size_t(di)*dj*dk*dl,data,cache,&sizes);
        for (int p=0;p<di;++p) for (int q=0;q<dj;++q) {
          const int a=t.ao[i]+p,b=t.ao[j]+q;
          if (a<b) continue;
          const size_t ab=pair_index(a,b);
          for (int r=0;r<dk;++r) for (int s=0;s<dl;++s) {
            const int c=t.ao[k]+r,d=t.ao[l]+s;
            if (c<d || ab<pair_index(c,d)) continue;
            consume(a,b,c,d,data[((size_t(s)*dk+r)*dj+q)*di+p]);
          }
        }
      }
    }
  }
}
