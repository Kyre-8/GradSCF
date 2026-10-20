// First-coordinate ERI entries shared by full-integral and direct contractions.
#pragma once
#include "tables.h"

template<class Accumulate>
void for_each_eri_geometry_entry(IntegralTables& t, int cart, Accumulate&& add) {
  const int n=t.n;
  auto index=[&](int i,int j,int k,int l) {return ((size_t(i)*n+j)*n+k)*n+l;};
  Intor ip=cart ? int2e_ip1_cart : int2e_ip1_sph;
  std::vector<double> block,cache;
  for (int i=0;i<t.nbas;++i) for (int j=0;j<t.nbas;++j)
    for (int k=0;k<t.nbas;++k) for (int l=0;l<t.nbas;++l) {
      int shells[]={i,j,k,l};
      const int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
      const int dk=t.ao[k+1]-t.ao[k],dl=t.ao[l+1]-t.ao[l];
      const size_t size=size_t(di)*dj*dk*dl;
      t.shell(ip,shells,3*size,block,cache);
      for (int x=0;x<3;++x) for (int s=0;s<dl;++s)
        for (int r=0;r<dk;++r) for (int q=0;q<dj;++q) for (int p=0;p<di;++p) {
          // A center derivative is the negative electron-coordinate derivative.
          const double value=-block[x*size+((size_t(s)*dk+r)*dj+q)*di+p];
          const int a=t.ao[i]+p,b=t.ao[j]+q,c=t.ao[k]+r,d=t.ao[l]+s;
          const int coordinate=t.coord_ptr(i)+x;
          add(index(a,b,c,d),coordinate,value);add(index(b,a,c,d),coordinate,value);
          add(index(c,d,a,b),coordinate,value);add(index(c,d,b,a),coordinate,value);
        }
    }
}
