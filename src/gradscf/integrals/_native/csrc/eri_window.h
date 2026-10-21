// Shell-window traversal and local AO indexing for bounded ERI products.
#pragma once
#include "tables.h"

struct ERIWindow {
  static constexpr size_t outside=std::numeric_limits<size_t>::max();
  std::array<int,8> shells;
  std::array<int,4> starts,shape;

  ERIWindow(const IntegralTables& t,const int32_t* slice=nullptr,size_t count=0) {
    const int nshell=int(t.ao.size())-1;
    if (slice && count!=8) throw std::invalid_argument("shell_slice must have eight endpoints");
    for (int a=0;a<4;++a) {
      int first=slice ? slice[2*a] : 0,last=slice ? slice[2*a+1] : nshell;
      if (first<0 || first>=last || last>nshell)
        throw std::invalid_argument("shell_slice requires 0 <= start < stop <= nshell");
      shells[2*a]=first;shells[2*a+1]=last;
      starts[a]=t.ao[first];shape[a]=t.ao[last]-starts[a];
    }
    // The vendor s1 fill computes each pair product in signed int first.
    if (slice && (int64_t(shape[0])*shape[1]>std::numeric_limits<int>::max() ||
                  int64_t(shape[2])*shape[3]>std::numeric_limits<int>::max()))
      throw std::invalid_argument("ERI shell_slice pair dimensions exceed int32 indexing");
  }

  void check_shape(const ffi::Buffer<ffi::F64>& buffer) const {
    if (buffer.dimensions().size()!=4)
      throw std::invalid_argument("Invalid ERI shell_slice tensor rank");
    for (int a=0;a<4;++a) if (buffer.dimensions()[a]!=shape[a])
      throw std::invalid_argument("ERI shell_slice AO dimensions do not match basis table");
  }

  size_t index(int i,int j,int k,int l) const {
    const int ao[]={i,j,k,l};
    size_t result=0;
    for (int a=0;a<4;++a) {
      int local=ao[a]-starts[a];
      if (local<0 || local>=shape[a]) return outside;
      result=result*shape[a]+local;
    }
    return result;
  }

  template<class Consume>
  void for_each_shell(Consume&& consume) const {
    // Every existing derivative emission is one of these ERI permutations.
    // Their inverse boxes cover all contributing base quartets. Earlier boxes
    // own overlaps; emissions themselves retain their original multiplicity.
    constexpr int permutations[8][4]={{0,1,2,3},{1,0,2,3},{0,1,3,2},{1,0,3,2},
                                      {2,3,0,1},{3,2,0,1},{2,3,1,0},{3,2,1,0}};
    std::vector<std::array<int,8>> boxes;
    for (const auto& permutation:permutations) {
      std::array<int,8> box;
      for (int a=0;a<4;++a) {
        box[2*permutation[a]]=shells[2*a];
        box[2*permutation[a]+1]=shells[2*a+1];
      }
      if (std::find(boxes.begin(),boxes.end(),box)==boxes.end()) boxes.push_back(box);
    }
    for (size_t b=0;b<boxes.size();++b) {
      const auto& box=boxes[b];
      for (int i=box[0];i<box[1];++i) for (int j=box[2];j<box[3];++j)
        for (int k=box[4];k<box[5];++k) for (int l=box[6];l<box[7];++l) {
          const int quartet[]={i,j,k,l};
          bool owned=false;
          for (size_t earlier=0;earlier<b && !owned;++earlier) {
            bool inside=true;
            for (int a=0;a<4;++a)
              inside=inside && boxes[earlier][2*a]<=quartet[a] && quartet[a]<boxes[earlier][2*a+1];
            owned=inside;
          }
          if (!owned) consume(i,j,k,l);
        }
    }
  }
};

template<class Consume>
void for_each_eri_shell(const IntegralTables& t,const ERIWindow* window,Consume&& consume) {
  if (window) return window->for_each_shell(consume);
  const int nshell=int(t.ao.size())-1;
  for (int i=0;i<nshell;++i) for (int j=0;j<nshell;++j)
    for (int k=0;k<nshell;++k) for (int l=0;l<nshell;++l) consume(i,j,k,l);
}
