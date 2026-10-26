// Packed four-center Coulomb values.
#include "compact.h"
#include "coulomb_shells.h"

void packed_eri(IntegralTables& t, int layout, int cart, double* out, size_t output_size) {
      size_t np=size_t(t.n)*(t.n+1)/2;
      size_t size=layout==0 ? np*np : np*(np+1)/2;
      if (output_size!=size) throw std::invalid_argument("Packed ERI shape mismatch");
      unique_eri(t,cart,0.,[&](int a,int b,int c,int d,double v) {
        size_t ab=pair_index(a,b),cd=pair_index(c,d);
        if (layout==0) {out[ab*np+cd]=v;out[cd*np+ab]=v;}
        else out[pair_index(ab,cd)]=v;
      });
}
