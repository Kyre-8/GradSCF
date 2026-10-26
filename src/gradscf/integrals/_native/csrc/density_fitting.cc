// Two-center metric and three-center orbital/auxiliary Coulomb values.
#include "compact.h"
#include "coulomb_shells.h"

void density_fitting(IntegralTables& t, int layout, int cart, int split, double* out, size_t output_size) {
      if (split<1 || split>=t.nbas) throw std::invalid_argument("Invalid orbital/auxiliary shell split");
      const int n=t.ao[split],m=t.n-n;
      const size_t np=size_t(n)*(n+1)/2;
      if (output_size!=(layout==2 ? size_t(m)*m : size_t(m)*np))
        throw std::invalid_argument("Auxiliary integral shape mismatch");
      Intor f=layout==2 ? (cart ? int2c2e_cart:int2c2e_sph) : (cart ? int3c2e_cart:int3c2e_sph);
      LocalOptimizer opt(t,layout==2 ? 2:3);
      std::vector<double> data,cache;
      if (layout==2) {
        for (int i=split;i<t.nbas;++i) for (int j=split;j<=i;++j) {
          int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j],sh[]={i,j};
          block(t,f,opt.value,sh,size_t(di)*dj,data,cache);
          for (int p=0;p<di;++p) for (int q=0;q<dj;++q) {
            int a=t.ao[i]+p-n,b=t.ao[j]+q-n;
            out[size_t(a)*m+b]=out[size_t(b)*m+a]=data[size_t(q)*di+p];
          }
        }
      } else {
        for (int i=0;i<split;++i) for (int j=0;j<=i;++j) for (int k=split;k<t.nbas;++k) {
          int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j],dk=t.ao[k+1]-t.ao[k],sh[]={i,j,k};
          block(t,f,opt.value,sh,size_t(di)*dj*dk,data,cache);
          for (int p=0;p<di;++p) for (int q=0;q<dj;++q) {
            int a=t.ao[i]+p,b=t.ao[j]+q;if (a<b) continue;
            for (int r=0;r<dk;++r)
              out[size_t(t.ao[k]+r-n)*np+pair_index(a,b)]=data[(size_t(r)*dj+q)*di+p];
          }
        }
      }
}
