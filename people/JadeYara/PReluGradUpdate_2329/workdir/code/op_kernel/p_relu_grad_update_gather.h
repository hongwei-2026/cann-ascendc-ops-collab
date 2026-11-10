#ifndef JADEYARA_CONTIG_GATHER_H
#define JADEYARA_CONTIG_GATHER_H
#include "kernel_operator.h"
#include "p_relu_grad_update_gather_tiling.h"
#include <type_traits>
namespace JadeYaraContiguous {
using namespace AscendC;
template<class T> class PReluGradUpdate {
public:
 __aicore__ inline void Init(GM_ADDR g,GM_ADDR x,GM_ADDR w,GM_ADDR dx,GM_ADDR up,
  const PReluGradUpdateGatherTilingData* t,TPipe* pipe){
  p_=pipe; total_=t->totalNum; span_=t->blockSpan; tile_=t->tileLen; c_=t->channelNum;s_=t->innerSize;
  cache_=t->cacheWeights!=0;ms_=t->magicS;ks_=t->shiftS;mc_=t->magicC;kc_=t->shiftC;
  start_=static_cast<uint64_t>(GetBlockIdx())*span_; end_=start_+span_;if(end_>total_)end_=total_;
  gGm_.SetGlobalBuffer((__gm__ T*)g,total_);xGm_.SetGlobalBuffer((__gm__ T*)x,total_);wGm_.SetGlobalBuffer((__gm__ T*)w,c_);
  dxGm_.SetGlobalBuffer((__gm__ T*)dx,total_);upGm_.SetGlobalBuffer((__gm__ T*)up,total_);
  if(start_>=end_)return;
  p_->InitBuffer(gRaw_,tile_*sizeof(T)+256);p_->InitBuffer(xRaw_,tile_*sizeof(T));
  p_->InitBuffer(wRaw_,(tile_+(std::is_same<T,float>::value?128:32))*sizeof(T));
  p_->InitBuffer(dxRaw_,tile_*sizeof(T));p_->InitBuffer(upRaw_,tile_*sizeof(T));
  p_->InitBuffer(mask_,std::is_same<T,float>::value?Align(tile_,512)+256:tile_);
  p_->InitBuffer(wExpanded_,tile_*4);p_->InitBuffer(pos_,tile_*4);p_->InitBuffer(ramp_,tile_*4);
  if constexpr(!std::is_same<T,float>::value){
   p_->InitBuffer(gFloat_,tile_*4+256);p_->InitBuffer(xFloat_,tile_*4);
   p_->InitBuffer(dxFloat_,tile_*4);p_->InitBuffer(upFloat_,tile_*4);p_->InitBuffer(wCompactFloat_,(tile_+32)*4);
  }
  CreateVecIndex(ramp_.Get<int32_t>(),static_cast<int32_t>(0),static_cast<uint32_t>(tile_));PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Process(){
  if(start_>=end_)return;
  if(cache_){
   Copy(wRaw_.Get<T>(),wGm_,0,c_);Sync<HardEvent::MTE2_V>();
   if constexpr(!std::is_same<T,float>::value){Cast(wCompactFloat_.Get<float>(),wRaw_.Get<T>(),RoundMode::CAST_NONE,c_);PipeBarrier<PIPE_V>();}
  }
  for(uint64_t z=start_;z<end_;){
   uint64_t n=end_-z;if(n>tile_)n=tile_;uint64_t count=Align(n,64),phase=z%s_,ch=(z/s_)%c_;
   Sync<HardEvent::V_MTE2>();Sync<HardEvent::MTE3_V>();
   Copy(gRaw_.Get<T>(),gGm_,z,n);Copy(xRaw_.Get<T>(),xGm_,z,n);
   uint64_t hole=0,left=0,slots=0;
   if(!cache_){
    uint64_t q=(phase+count-1)/s_+1;
    if(ch+q<=c_){Copy(wRaw_.Get<T>(),wGm_,ch,q);slots=q;}
    else{left=c_-ch;uint64_t b=Align(left,32/sizeof(T));hole=b-left;Copy(wRaw_.Get<T>(),wGm_,ch,left);Copy(wRaw_.Get<T>()[b],wGm_,0,q-left);slots=b+q-left;}
   }
   Sync<HardEvent::MTE2_V>();
   if constexpr(!std::is_same<T,float>::value){
    if(!cache_){Cast(wCompactFloat_.Get<float>(),wRaw_.Get<T>(),RoundMode::CAST_NONE,slots);PipeBarrier<PIPE_V>();}
   }
   BuildWeights(phase,ch,count,left,hole);
   Compute(n,count);
   Sync<HardEvent::V_MTE3>();
   DataCopyExtParams cp={1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
   DataCopyPad(dxGm_[z],dxRaw_.Get<T>(),cp);DataCopyPad(upGm_[z],upRaw_.Get<T>(),cp);
   z+=n;
  }
  Sync<HardEvent::MTE3_V>();
 }
private:
 __aicore__ inline uint64_t Align(uint64_t n,uint64_t a){return (n+a-1)/a*a;}
 template<HardEvent E> __aicore__ inline void Sync(){event_t e=static_cast<event_t>(p_->FetchEventID(E));SetFlag<E>(e);WaitFlag<E>(e);}
 __aicore__ inline void Copy(LocalTensor<T> dst,GlobalTensor<T>&src,uint64_t offset,uint64_t n){
  T zero;if constexpr(sizeof(T)==4)*reinterpret_cast<uint32_t*>(&zero)=0;else *reinterpret_cast<uint16_t*>(&zero)=0;
  DataCopyExtParams cp={1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
  DataCopyPadExtParams<T> pad={true,0,static_cast<uint8_t>(Align(n,32/sizeof(T))-n),zero};DataCopyPad(dst,src[offset],cp,pad);
 }
 __aicore__ inline LocalTensor<float> G(){if constexpr(std::is_same<T,float>::value)return gRaw_.Get<float>();else return gFloat_.Get<float>();}
 __aicore__ inline LocalTensor<float> X(){if constexpr(std::is_same<T,float>::value)return xRaw_.Get<float>();else return xFloat_.Get<float>();}
 __aicore__ inline LocalTensor<float> DX(){if constexpr(std::is_same<T,float>::value)return dxRaw_.Get<float>();else return dxFloat_.Get<float>();}
 __aicore__ inline LocalTensor<float> UP(){if constexpr(std::is_same<T,float>::value)return upRaw_.Get<float>();else return upFloat_.Get<float>();}
 __aicore__ inline LocalTensor<float> WSource(){if constexpr(std::is_same<T,float>::value)return wRaw_.Get<float>();else return wCompactFloat_.Get<float>();}
 __aicore__ inline void Div(LocalTensor<int32_t> v,uint64_t divisor,uint64_t m,uint64_t shift,uint64_t n){
  if(divisor==1)return;
  Muls(v,v,static_cast<int32_t>(m),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
  ShiftRight(v,v,static_cast<int32_t>(shift),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void BuildWeights(uint64_t phase,uint64_t ch,uint64_t n,uint64_t left,uint64_t hole){
  LocalTensor<int32_t> idx=DX().template ReinterpretCast<int32_t>();LocalTensor<int32_t> tmp=pos_.Get<int32_t>();
  Adds(idx,ramp_.Get<int32_t>(),static_cast<int32_t>(phase),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();Div(idx,s_,ms_,ks_,n);
  if(cache_){
   Adds(idx,idx,static_cast<int32_t>(ch),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
   Adds(tmp,idx,static_cast<int32_t>(0),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();Div(tmp,c_,mc_,kc_,n);
   Muls(tmp,tmp,static_cast<int32_t>(c_),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
   Sub(idx,idx,tmp,static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
  }else if(hole!=0){
   Adds(tmp,idx,static_cast<int32_t>(1)-static_cast<int32_t>(left),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
   Maxs(tmp,tmp,static_cast<int32_t>(0),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
   Mins(tmp,tmp,static_cast<int32_t>(1),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
   Muls(tmp,tmp,static_cast<int32_t>(hole),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
   Add(idx,idx,tmp,static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
  }
  Muls(idx,idx,static_cast<int32_t>(4),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
  Gather(wExpanded_.Get<float>(),WSource(),idx.template ReinterpretCast<uint32_t>(),static_cast<uint32_t>(0),static_cast<uint32_t>(n));PipeBarrier<PIPE_V>();
 }
 __aicore__ inline void Compute(uint64_t n,uint64_t count){
  auto g=G();auto x=X();auto dx=DX();auto up=UP();auto pos=pos_.Get<float>();auto mask=mask_.Get<uint8_t>();
  if constexpr(!std::is_same<T,float>::value){Cast(g,gRaw_.Get<T>(),RoundMode::CAST_NONE,n);Cast(x,xRaw_.Get<T>(),RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}
  Duplicate(dx,0.0f,static_cast<int32_t>(count));Duplicate(pos,1.0f,static_cast<int32_t>(count));PipeBarrier<PIPE_V>();
  Compare(mask,x,dx,CMPMODE::GT,static_cast<uint32_t>(count));PipeBarrier<PIPE_V>();
  Select(up,mask,dx,pos,SELMODE::VSEL_TENSOR_TENSOR_MODE,static_cast<uint32_t>(count));PipeBarrier<PIPE_V>();
  Mul(x,up,x,n);Muls(pos,up,-1.0f,n);PipeBarrier<PIPE_V>();Adds(pos,pos,1.0f,n);
  Mul(dx,up,wExpanded_.Get<float>(),n);PipeBarrier<PIPE_V>();Add(dx,dx,pos,n);PipeBarrier<PIPE_V>();
  Mul(dx,dx,g,n);Mul(up,x,g,n);PipeBarrier<PIPE_V>();
  if constexpr(!std::is_same<T,float>::value){Cast(dxRaw_.Get<T>(),dx,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();Cast(dx,dxRaw_.Get<T>(),RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}
  Muls(pos,dx,0.0f,n);PipeBarrier<PIPE_V>();Add(up,up,pos,n);PipeBarrier<PIPE_V>();
  if constexpr(!std::is_same<T,float>::value){Cast(upRaw_.Get<T>(),up,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();}
 }
 TPipe* p_;uint64_t total_,span_,tile_,c_,s_,start_,end_,ms_,ks_,mc_,kc_;bool cache_;
 GlobalTensor<T> gGm_,xGm_,wGm_,dxGm_,upGm_;
 TBuf<TPosition::VECCALC> gRaw_,xRaw_,wRaw_,dxRaw_,upRaw_,mask_,wExpanded_,pos_,ramp_;
 TBuf<TPosition::VECCALC> gFloat_,xFloat_,dxFloat_,upFloat_,wCompactFloat_;
};
}
#endif
