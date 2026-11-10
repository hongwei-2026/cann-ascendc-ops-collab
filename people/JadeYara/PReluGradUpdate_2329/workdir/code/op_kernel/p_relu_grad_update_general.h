#ifndef JADEYARA_GENERAL_BROADCAST_H
#define JADEYARA_GENERAL_BROADCAST_H
#include "kernel_operator.h"
#include "p_relu_grad_update_general_tiling.h"
#include <type_traits>
namespace JadeYaraGeneral {
using namespace AscendC;
template<class T> class PReluGradUpdate {
public:
 __aicore__ inline void Init(GM_ADDR g,GM_ADDR x,GM_ADDR w,GM_ADDR dx,GM_ADDR up,
                            const PReluGradUpdateGeneralTilingData* data,TPipe* pipe) {
  t_=data;p_=pipe;start_=0;end_=0;
  if(t_->totalNum==0)return;
  start_=static_cast<uint64_t>(GetBlockIdx())*t_->blockSpan;
  if(start_>=t_->totalNum)return;
  end_=start_+t_->blockSpan;if(end_>t_->totalNum)end_=t_->totalNum;
  gGm_.SetGlobalBuffer((__gm__ T*)g,t_->sourceElements[0]);
  xGm_.SetGlobalBuffer((__gm__ T*)x,t_->sourceElements[1]);
  wGm_.SetGlobalBuffer((__gm__ T*)w,t_->sourceElements[2]);
  dxGm_.SetGlobalBuffer((__gm__ T*)dx,t_->totalNum);upGm_.SetGlobalBuffer((__gm__ T*)up,t_->totalNum);
  const uint64_t bytes=t_->tileLen*4;
  // A phase: g,pos,compact (neg during math). B=A+256mod512: x,w,dx,up.
  // Each tile is 2048 lanes, so payload bytes are multiples of512.
  p_->InitBuffer(g_,bytes+256);p_->InitBuffer(x_,bytes);p_->InitBuffer(w_,bytes);
  p_->InitBuffer(dx_,bytes);p_->InitBuffer(up_,bytes+256);p_->InitBuffer(pos_,bytes);
  p_->InitBuffer(compact_,bytes);p_->InitBuffer(ramp_,bytes);p_->InitBuffer(lanes_,bytes);
  p_->InitBuffer(mask_,t_->tileLen);
  if constexpr(!std::is_same<T,float>::value){
   p_->InitBuffer(raw_,t_->tileLen*sizeof(T));p_->InitBuffer(dxRaw_,t_->tileLen*sizeof(T));p_->InitBuffer(upRaw_,t_->tileLen*sizeof(T));
  }
  CreateVecIndex(ramp_.Get<int32_t>(),static_cast<int32_t>(0),static_cast<uint32_t>(t_->tileLen));PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Process() {
  if(t_->totalNum==0 || start_>=end_)return;
  for(uint64_t z=start_;z<end_;){
   uint64_t valid=end_-z;if(valid>t_->tileLen)valid=t_->tileLen;
   const uint64_t toBoundary=t_->splitPeriod-z%t_->splitPeriod;if(valid>toBoundary)valid=toBoundary;
   const uint32_t n=static_cast<uint32_t>(valid),count=static_cast<uint32_t>(Align(valid,64));
   Sync<HardEvent::MTE3_V>();
   // Clamp BEFORE every index formula: virtual tail lanes repeat a valid lane.
   Mins(lanes_.Get<int32_t>(),ramp_.Get<int32_t>(),static_cast<int32_t>(n-1),count);PipeBarrier<PIPE_V>();
   uint64_t a[PRELU_GENERAL_RANK]={},b[PRELU_GENERAL_RANK]={};uint32_t first=t_->rank;
   for(uint32_t j=0;j<t_->rank;++j){
    a[j]=(z/t_->periods[j])%t_->dims[j];b[j]=((z+valid-1)/t_->periods[j])%t_->dims[j];
    if(first==t_->rank && a[j]!=b[j])first=j;
   }
   Expand(0,g_.Get<float>(),z,n,count,a,b,first);
   Expand(1,x_.Get<float>(),z,n,count,a,b,first);
   Expand(2,w_.Get<float>(),z,n,count,a,b,first);
   Compute(n,count);Sync<HardEvent::V_MTE3>();
   const DataCopyExtParams cp={1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
   DataCopyPad(dxGm_[z],DXOut(),cp);DataCopyPad(upGm_[z],UPOut(),cp);z+=valid;
  }
  Sync<HardEvent::MTE3_V>();
 }
private:
 __aicore__ inline uint64_t Align(uint64_t n,uint64_t a){return (n+a-1)/a*a;}
 template<HardEvent E> __aicore__ inline void Sync(){const event_t e=static_cast<event_t>(p_->FetchEventID(E));SetFlag<E>(e);WaitFlag<E>(e);}
 __aicore__ inline GlobalTensor<T>& Source(uint32_t q){if(q==0)return gGm_;if(q==1)return xGm_;return wGm_;}
 __aicore__ inline LocalTensor<T> DXOut(){if constexpr(std::is_same<T,float>::value)return dx_.Get<T>();else return dxRaw_.Get<T>();}
 __aicore__ inline LocalTensor<T> UPOut(){if constexpr(std::is_same<T,float>::value)return up_.Get<T>();else return upRaw_.Get<T>();}
 __aicore__ inline void Copy(LocalTensor<T> dst,GlobalTensor<T>& src,uint64_t offset,uint64_t n){
  T zero;if constexpr(sizeof(T)==4)*reinterpret_cast<uint32_t*>(&zero)=0;else *reinterpret_cast<uint16_t*>(&zero)=0;
  const DataCopyExtParams cp={1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
  const DataCopyPadExtParams<T> pad={true,0,static_cast<uint8_t>(Align(n,32/sizeof(T))-n),zero};
  DataCopyPad(dst,src[offset],cp,pad);
 }
 __aicore__ inline void Include(uint64_t l,uint64_t h,uint64_t& lo,uint64_t& hi){if(l<lo)lo=l;if(h>hi)hi=h;}
 // Exact O(rank) min/max of lexicographic boxes. Endpoint-only bounds are wrong
 // when any broadcast axis carries. The P* cut proves hi-lo+1 <= tileLen.
 __aicore__ inline void Window(uint32_t q,const uint64_t* a,const uint64_t* b,uint32_t first,
                              uint64_t& lo,uint64_t& hi,uint64_t& base){
  uint64_t pa[PRELU_GENERAL_RANK+1]={},pb[PRELU_GENERAL_RANK+1]={};
  for(uint32_t j=0;j<t_->rank;++j){pa[j+1]=pa[j]+a[j]*t_->strides[q][j];pb[j+1]=pb[j]+b[j]*t_->strides[q][j];}
  base=pa[t_->rank];lo=base;hi=base;Include(pb[t_->rank],pb[t_->rank],lo,hi);
  if(first==t_->rank)return;
  for(uint32_t j=first+1;j<t_->rank;++j){
   const uint64_t stride=t_->strides[q][j],suffix=t_->suffix[q][j];
   if(a[j]+1<t_->dims[j])Include(pa[j]+(a[j]+1)*stride,pa[j]+(t_->dims[j]-1)*stride+suffix-1,lo,hi);
   if(b[j]>0)Include(pb[j],pb[j]+(b[j]-1)*stride+suffix-1,lo,hi);
  }
  if(b[first]>a[first]+1)Include(pa[first]+(a[first]+1)*t_->strides[q][first],
      pa[first]+(b[first]-1)*t_->strides[q][first]+t_->suffix[q][first]-1,lo,hi);
 }
 __aicore__ inline void Expand(uint32_t q,LocalTensor<float> dst,uint64_t z,uint32_t n,uint32_t count,
                              const uint64_t* a,const uint64_t* b,uint32_t first){
  if(t_->identity[q]){
   // Initialize padded computation lanes, then direct DMA/Cast valid lanes.
   Duplicate(dst,0.0f,static_cast<int32_t>(count));Sync<HardEvent::V_MTE2>();
   if constexpr(std::is_same<T,float>::value){Copy(dst.template ReinterpretCast<T>(),Source(q),z,n);Sync<HardEvent::MTE2_V>();}
   else{Copy(raw_.Get<T>(),Source(q),z,n);Sync<HardEvent::MTE2_V>();Cast(dst,raw_.Get<T>(),RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}
   return;
  }
  uint64_t lo=0,hi=0,base=0;Window(q,a,b,first,lo,hi,base);const uint64_t slots=hi-lo+1;
  // Gather completes before compact storage is reused by the next source.
  Sync<HardEvent::V_MTE2>();
  if constexpr(std::is_same<T,float>::value){Copy(compact_.Get<T>(),Source(q),lo,slots);Sync<HardEvent::MTE2_V>();}
  else{Copy(raw_.Get<T>(),Source(q),lo,slots);Sync<HardEvent::MTE2_V>();Cast(compact_.Get<float>(),raw_.Get<T>(),RoundMode::CAST_NONE,slots);PipeBarrier<PIPE_V>();}
  auto idx=dx_.Get<int32_t>();auto tmp=pos_.Get<int32_t>();
  Muls(idx,lanes_.Get<int32_t>(),static_cast<int32_t>(t_->strides[q][t_->rank-1]),count);PipeBarrier<PIPE_V>();
  Adds(idx,idx,static_cast<int32_t>(base-lo),count);PipeBarrier<PIPE_V>();
  for(uint32_t j=0;j+1<t_->rank;++j){
   const int32_t coefficient=t_->coefficient[q][j];if(coefficient==0)continue;
   Adds(tmp,lanes_.Get<int32_t>(),static_cast<int32_t>(z%t_->periods[j]),count);PipeBarrier<PIPE_V>();
   if(t_->periods[j]!=1){Muls(tmp,tmp,t_->multiplier[j],count);PipeBarrier<PIPE_V>();ShiftRight(tmp,tmp,static_cast<int32_t>(t_->shift[j]),count);PipeBarrier<PIPE_V>();}
   Muls(tmp,tmp,coefficient,count);PipeBarrier<PIPE_V>();Add(idx,idx,tmp,count);PipeBarrier<PIPE_V>();
  }
  Muls(idx,idx,static_cast<int32_t>(4),count);PipeBarrier<PIPE_V>();
  Gather(dst,compact_.Get<float>(),idx.template ReinterpretCast<uint32_t>(),static_cast<uint32_t>(0),count);PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Compute(uint32_t n,uint32_t count){
  auto g=g_.Get<float>();auto x=x_.Get<float>();auto w=w_.Get<float>();auto dx=dx_.Get<float>();auto up=up_.Get<float>();
  auto pos=pos_.Get<float>();auto neg=compact_.Get<float>();auto mask=mask_.Get<uint8_t>();
  Duplicate(pos,0.0f,static_cast<int32_t>(count));Duplicate(dx,1.0f,static_cast<int32_t>(count));PipeBarrier<PIPE_V>();
  Compare(mask,x,pos,CMPMODE::GT,count);PipeBarrier<PIPE_V>();
  Select(neg,mask,pos,dx,SELMODE::VSEL_TENSOR_TENSOR_MODE,count);PipeBarrier<PIPE_V>();
  // Preserve builtin mask multiplication and non-fused FP32 operation ordering.
  Mul(x,neg,x,n);Mul(dx,neg,w,n);Muls(pos,neg,-1.0f,n);PipeBarrier<PIPE_V>();
  Adds(pos,pos,1.0f,n);PipeBarrier<PIPE_V>();Add(dx,dx,pos,n);PipeBarrier<PIPE_V>();
  Mul(dx,dx,g,n);Mul(up,x,g,n);PipeBarrier<PIPE_V>();
  if constexpr(!std::is_same<T,float>::value){Cast(dxRaw_.Get<T>(),dx,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();Cast(dx,dxRaw_.Get<T>(),RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}
  Muls(pos,dx,0.0f,n);PipeBarrier<PIPE_V>();Add(up,up,pos,n);PipeBarrier<PIPE_V>();
  if constexpr(!std::is_same<T,float>::value){Cast(upRaw_.Get<T>(),up,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();}
 }
 TPipe* p_;const PReluGradUpdateGeneralTilingData* t_;uint64_t start_,end_;
 GlobalTensor<T> gGm_,xGm_,wGm_,dxGm_,upGm_;
 TBuf<TPosition::VECCALC> g_,x_,w_,dx_,up_,pos_,compact_,ramp_,lanes_,mask_,raw_,dxRaw_,upRaw_;
};
}
#endif
