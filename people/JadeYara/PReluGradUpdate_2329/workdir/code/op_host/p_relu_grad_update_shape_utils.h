#ifndef JADEYARA_PRELU_SHAPE_UTILS_H
#define JADEYARA_PRELU_SHAPE_UTILS_H
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>
namespace JadeYaraShape {
using Dims=std::vector<int64_t>;
inline bool UnknownRank(const Dims& d){return d.size()==1 && d[0]==-2;}
inline bool Valid(const Dims& d,bool runtime){
 if(d.size()>8)return false;
 for(int64_t v:d)if(v<(runtime?0:-1))return false;
 return true;
}
inline bool Product(const Dims& d,uint64_t& n){
 n=1;for(int64_t v:d)if(v<0)return false;
 for(int64_t v:d)if(v==0){n=0;return true;}
 const uint64_t limit=static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
 for(int64_t v:d){if(n>limit/static_cast<uint64_t>(v))return false;n*=static_cast<uint64_t>(v);}
 return true;
}
inline bool Broadcast(const Dims& a,const Dims& b,Dims& out){
 const size_t rank=std::max(a.size(),b.size());out.assign(rank,1);
 for(size_t j=0;j<rank;++j){
  const int64_t x=j<rank-a.size()?1:a[j-(rank-a.size())];
  const int64_t y=j<rank-b.size()?1:b[j-(rank-b.size())];
  if(x==y)out[j]=x;else if(x==1)out[j]=y;else if(y==1)out[j]=x;
  else if(x<0)out[j]=y;else if(y<0)out[j]=x;else return false;
 }return true;
}
// Exact builtin broadcast_weight_shape branch order. InferShape uses origin
// shapes; runtime uses compact physical storage shapes, including NC1HWC0.
inline bool MapWeight(const Dims& x,const Dims& w,bool blocked,Dims& mapped){
 if(!x.empty() && w.size()==x.size()-1 && w.size()!=1){mapped=w;mapped.insert(mapped.begin(),1);}
 else if(x.size()==4){if(w.empty())return false;mapped={1,w[0],1,1};}
 else if(blocked && w.size()==5){mapped=w;}
 else if(blocked && w.size()==1 && w[0]!=1){
  const int64_t c1=w[0]<0?w[0]:w[0]/16+(w[0]%16!=0);mapped={1,c1,1,1,16};
 }else{if(x.size()<2 || w.empty())return false;mapped.assign(x.size(),1);mapped[1]=w[0];}
 return true;
}
inline bool BroadcastTo(const Dims& src,const Dims& dst){
 if(src.size()>dst.size())return false;const size_t lead=dst.size()-src.size();
 for(size_t j=0;j<src.size();++j){const int64_t s=src[j],t=dst[lead+j];if(s>=0&&t>=0&&s!=1&&s!=t)return false;}
 return true;
}
inline bool Normalize(const Dims& g,const Dims& x,const Dims& w,bool blocked,bool runtime,Dims& out,Dims& mapped){
 if(!Valid(g,runtime)||!Valid(x,runtime)||!Valid(w,runtime))return false;
 if(!Broadcast(g,x,out)||!MapWeight(x,w,blocked,mapped)||!BroadcastTo(mapped,out))return false;
 uint64_t oldN=0,newN=0;const bool oldKnown=Product(w,oldN),newKnown=Product(mapped,newN);
 if(oldKnown&&newKnown&&oldN!=newN)return false;
 return !runtime||(oldKnown&&newKnown);
}
}
#endif
