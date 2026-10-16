// Shared second-coordinate contraction. Higher products differentiate only
// the shell evaluator; index permutations and nuclear-center terms stay shared.
#pragma once
#include "tables.h"

template<bool transpose, class Evaluate>
void contract_geometry_hessian(IntegralTables& t, int op, int cart,
                                const double* u, const double* input,
                                double* output, Evaluate&& evaluate) {
  // Shadow shells may have been appended, but never enter the physical loop.
  const int nshell=int(t.ao.size())-1;
  // Each call contributes one ordered Hessian entry. Cross-slot entries
  // explicitly contribute both coordinate orders, even for a shared atom.
  auto add=[&](size_t index,int a,int b,double value) {
    if constexpr (transpose) output[a]+=value*u[b]*input[index];
    else output[index]+=value*u[a]*input[b];
  };
  auto cross=[&](size_t index,int a,int b,double value) {
    add(index,a,b,value); add(index,b,a,value);
  };
  const int n=t.n;
  auto index2=[&](int c,int i,int j) { return (size_t(c)*n+i)*n+j; };
  auto index4=[&](int i,int j,int k,int l) { return ((size_t(i)*n+j)*n+k)*n+l; };
  std::vector<double> block,cache;
  if (op==4) {
    Intor kernels_cart[]={int2e_ipip1_cart,int2e_ipvip1_cart,int2e_ip1ip2_cart};
    Intor kernels_sph[]={int2e_ipip1_sph,int2e_ipvip1_sph,int2e_ip1ip2_sph};
    for (int i=0; i<nshell; ++i) for (int j=0; j<nshell; ++j)
      for (int k=0; k<nshell; ++k) for (int l=0; l<nshell; ++l) {
        int shells[]={i,j,k,l};
        int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
        int dk=t.ao[k+1]-t.ao[k],dl=t.ao[l+1]-t.ao[l];
        size_t size=size_t(di)*dj*dk*dl;
        for (int kind=0; kind<3; ++kind) {
          evaluate(cart ? kernels_cart[kind] : kernels_sph[kind],shells,9*size,block,cache,-1);
          for (int x=0; x<3; ++x) for (int y=0; y<3; ++y)
            for (int s=0; s<dl; ++s) for (int r=0; r<dk; ++r)
              for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p) {
                double value=block[(x*3+y)*size+((size_t(s)*dk+r)*dj+q)*di+p];
                int a=t.ao[i]+p,b=t.ao[j]+q,c=t.ao[k]+r,d=t.ao[l]+s;
                int px=t.coord_ptr(i)+x;
                if (kind==0) {
                  int py=t.coord_ptr(i)+y;
                  add(index4(a,b,c,d),px,py,value); add(index4(b,a,c,d),px,py,value);
                  add(index4(c,d,a,b),px,py,value); add(index4(c,d,b,a),px,py,value);
                } else if (kind==1) {
                  int py=t.coord_ptr(j)+y;
                  cross(index4(a,b,c,d),px,py,value);
                  cross(index4(c,d,a,b),px,py,value);
                } else {
                  int py=t.coord_ptr(k)+y;
                  cross(index4(a,b,c,d),px,py,value); cross(index4(b,a,c,d),px,py,value);
                  cross(index4(a,b,d,c),px,py,value); cross(index4(b,a,d,c),px,py,value);
                }
              }
        }
      }
  } else {
    Intor same_cart[]={int1e_ipipovlp_cart,int1e_ipipkin_cart,int1e_ipiprinv_cart,int1e_ipipr_cart};
    Intor same_sph[]={int1e_ipipovlp_sph,int1e_ipipkin_sph,int1e_ipiprinv_sph,int1e_ipipr_sph};
    Intor mixed_cart[]={int1e_ipovlpip_cart,int1e_ipkinip_cart,int1e_iprinvip_cart,int1e_iprip_cart};
    Intor mixed_sph[]={int1e_ipovlpip_sph,int1e_ipkinip_sph,int1e_iprinvip_sph,int1e_iprip_sph};
    const int components=op==3 ? 3 : 1;
    for (int i=0; i<nshell; ++i) for (int j=0; j<nshell; ++j) {
      int shells[]={i,j};
      int di=t.ao[i+1]-t.ao[i],dj=t.ao[j+1]-t.ao[j];
      size_t size=size_t(di)*dj;
      for (int atom=0; atom<(op==2 ? t.natm : 1); ++atom) {
        double factor=1.;
        int nucleus=-1;
        if (op==2) {
          factor=-t.atm[atom*ATM_SLOTS+CHARGE_OF];
          if (factor==0.) continue;
          nucleus=t.atm[atom*ATM_SLOTS+PTR_COORD];
          for (int x=0; x<3; ++x) t.env[PTR_RINV_ORIG+x]=t.env[nucleus+x];
        }
        // For each nuclear potential, translation invariance replaces an AO
        // coordinate derivative by (d/d R_AO - d/d R_nucleus). Expanding both
        // factors includes all operator/basis and operator/operator terms.
        auto add_relative=[&](size_t index,int a,int b,int x,int y,double value,bool both) {
          auto entry=[&](int first,int second,double v) {
            if (both) cross(index,first,second,v); else add(index,first,second,v);
          };
          entry(a+x,b+y,value);
          if (nucleus>=0) {
            entry(a+x,nucleus+y,-value);
            entry(nucleus+x,b+y,-value);
            entry(nucleus+x,nucleus+y,value);
          }
        };
        for (int kind=0; kind<2; ++kind) {
          Intor kernel=kind==0 ? (cart ? same_cart[op] : same_sph[op])
                               : (cart ? mixed_cart[op] : mixed_sph[op]);
          evaluate(kernel,shells,9*components*size,block,cache,nucleus);
          for (int c=0; c<components; ++c) for (int x=0; x<3; ++x) for (int y=0; y<3; ++y)
            for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p) {
              // libcint ipipr: (bra,bra,r); iprip: (bra,r,ket).
              int component=op==3 ? (kind==0 ? (x*3+y)*3+c : (x*3+c)*3+y) : x*3+y;
              double value=factor*block[component*size+q*di+p];
              int a=t.ao[i]+p,b=t.ao[j]+q;
              if (kind==0) {
                add_relative(index2(c,a,b),t.coord_ptr(i),t.coord_ptr(i),x,y,value,false);
                add_relative(index2(c,b,a),t.coord_ptr(i),t.coord_ptr(i),x,y,value,false);
              } else {
                add_relative(index2(c,a,b),t.coord_ptr(i),t.coord_ptr(j),x,y,value,true);
              }
            }
        }
      }
      if (op==3) {
        // d^2 <i|(r-O)_c|j> / dO_c dR = -dS/dR = +<nabla i|j>.
        evaluate(cart ? int1e_ipovlp_cart : int1e_ipovlp_sph,shells,3*size,block,cache,-1);
        for (int c=0; c<3; ++c) for (int x=0; x<3; ++x)
          for (int q=0; q<dj; ++q) for (int p=0; p<di; ++p) {
            double value=block[x*size+q*di+p];
            int a=t.ao[i]+p,b=t.ao[j]+q;
            cross(index2(c,a,b),t.coord_ptr(i)+x,PTR_COMMON_ORIG+c,value);
            cross(index2(c,b,a),t.coord_ptr(i)+x,PTR_COMMON_ORIG+c,value);
          }
      }
    }
  }
}
