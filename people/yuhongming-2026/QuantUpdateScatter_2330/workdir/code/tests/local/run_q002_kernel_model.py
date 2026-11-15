"""Run the real candidate/baseline kernel classes against a synchronous CPU API model.
The model checks arithmetic, allocation/queue ownership and DMA bounds. It does
not model device instruction timing, memory races or CANN compiler behavior.
"""
import argparse
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

CODE=Path(__file__).resolve().parents[2]
SEAT=CODE.parents[1]
STUB=r"""
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#define __aicore__
#define __gm__
using GM_ADDR=unsigned char*;
inline float bits_float(uint32_t u) { float v; std::memcpy(&v,&u,4); return v; }
inline uint32_t float_bits(float v) { uint32_t u; std::memcpy(&u,&v,4); return u; }
struct half {
    uint16_t bits=0;
    half()=default;
    half(float v) {
        // The test input generator uses exact multiples of 1/4 in the fp16 range.
        uint32_t b=float_bits(v), sign=(b>>16)&0x8000;
        int exp=static_cast<int>((b>>23)&255)-127+15;
        if(exp<=0) { bits=static_cast<uint16_t>(sign); }
        else if(exp>=31) { bits=static_cast<uint16_t>(sign|0x7c00); }
        else { bits=static_cast<uint16_t>(sign|(exp<<10)|((b>>13)&0x3ff)); }
    }
    operator float() const {
        uint32_t sign=static_cast<uint32_t>(bits&0x8000)<<16;
        uint32_t exp=(bits>>10)&31, mant=bits&1023;
        if(exp==0) return bits_float(sign);
        if(exp==31) return bits_float(sign|0x7f800000|(mant<<13));
        return bits_float(sign|((exp+112)<<23)|(mant<<13));
    }
};
struct bfloat16_t {
    uint16_t bits=0;
    bfloat16_t()=default;
    bfloat16_t(float v):bits(static_cast<uint16_t>(float_bits(v)>>16)) {}
    operator float() const { return bits_float(static_cast<uint32_t>(bits)<<16); }
};
static_assert(sizeof(half)==2 && sizeof(bfloat16_t)==2,"device dtype widths");
namespace AscendC {
enum class TPosition { VECCALC, VECIN, VECOUT };
enum class HardEvent { MTE2_MTE3, MTE3_MTE2, MTE2_S, V_MTE2, MTE2_V, V_MTE3, MTE3_V };
enum class RoundMode { CAST_NONE, CAST_RINT };
using event_t=int;
constexpr int PIPE_V=0;
struct Stats { size_t ub=0, copies=0, copyBytes=0, flags=0, queueAlloc=0; };
struct Barrier {
    std::mutex mutex;
    std::condition_variable cv;
    int total, waiting=0, generation=0;
    bool aborted=false;
    explicit Barrier(int n):total(n){}
    void cancel() { std::lock_guard<std::mutex> lock(mutex); aborted=true; cv.notify_all(); }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        if(aborted) throw std::runtime_error("barrier canceled");
        int old=generation;
        if(++waiting==total) { waiting=0; ++generation; cv.notify_all(); }
        else cv.wait(lock,[&]{return generation!=old || aborted;});
        if(aborted) throw std::runtime_error("barrier canceled");
    }
};
struct CoreContext { int index=0; Barrier* barrier=nullptr; Stats stats; };
inline thread_local CoreContext* current=nullptr;
struct Region { uintptr_t begin, end; };
inline std::vector<Region> gmRegions;
inline void ensure(bool value,const char* text) { if(!value) throw std::runtime_error(text); }
inline void checkGm(const void* p,size_t bytes) {
    uintptr_t first=reinterpret_cast<uintptr_t>(p);
    for(const auto& r:gmRegions) if(first>=r.begin && first<=r.end && bytes<=r.end-first) return;
    throw std::runtime_error("GM access outside registered tensor");
}
inline int64_t GetBlockIdx(){return current->index;}
inline void SyncAll(){current->barrier->wait();}
template<int> inline void PipeBarrier() {}
template<HardEvent> inline void SetFlag(event_t) { ++current->stats.flags; }
template<HardEvent> inline void WaitFlag(event_t) {}
struct Storage {
    std::vector<uint8_t> raw;
    size_t bytes;
    explicit Storage(size_t n):raw(n+64,0xA7),bytes(n){}
    uint8_t* data(){return raw.data()+32;}
    void verify() const {
        for(size_t i=0;i<32;++i) ensure(raw[i]==0xA7 && raw[bytes+32+i]==0xA7,"UB guard overwritten");
    }
};
template<class T> struct LocalTensor {
    std::shared_ptr<Storage> storage;
    size_t offset=0;
    int slot=-1;
    void check(size_t count) const {
        ensure(static_cast<bool>(storage),"uninitialized local buffer");
        ensure(offset<=storage->bytes && count<=((storage->bytes-offset)/sizeof(T)),"UB tensor access exceeds allocation");
        storage->verify();
    }
    T* ptr() const { return reinterpret_cast<T*>(storage->data()+offset); }
    T GetValue(int64_t i) const { ensure(i>=0,"negative local index");check(static_cast<size_t>(i)+1);return ptr()[i]; }
};
template<class T> struct GlobalTensor {
    T* ptr=nullptr;
    void SetGlobalBuffer(T* p){ptr=p;}
    GlobalTensor operator[](int64_t offset) const {
        return {reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(ptr)+offset*static_cast<int64_t>(sizeof(T)))};
    }
    T GetValue(int64_t i) const { auto sub=(*this)[i];checkGm(sub.ptr,sizeof(T));return *sub.ptr; }
};
template<TPosition> struct TBuf {
    std::shared_ptr<Storage> storage;
    template<class T> LocalTensor<T> Get(){return {storage,0,-1};}
};
template<TPosition Position,int Depth> struct TQue {
    std::vector<std::shared_ptr<Storage>> slots;
    std::vector<int> state;
    std::deque<int> ready;
    size_t next=0;
    template<class T> LocalTensor<T> AllocTensor(){
        ensure(!slots.empty(),"uninitialized queue");
        for(size_t tries=0;tries<slots.size();++tries){
            int i=static_cast<int>(next++%slots.size());
            if(state[i]==0){state[i]=1;++current->stats.queueAlloc;return {slots[i],0,i};}
        }
        throw std::runtime_error("all queue slots still owned");
    }
    template<class T> void EnQue(LocalTensor<T> t){
        ensure(t.slot>=0 && state[t.slot]==1,"EnQue without ownership");
        ensure(ready.size()<Depth,"queue depth exceeded");
        state[t.slot]=2;ready.push_back(t.slot);
    }
    template<class T> LocalTensor<T> DeQue(){
        ensure(!ready.empty(),"empty DeQue");
        int i=ready.front();ready.pop_front();state[i]=1;return {slots[i],0,i};
    }
    template<class T> void FreeTensor(LocalTensor<T> t){
        ensure(t.slot>=0 && state[t.slot]==1,"FreeTensor without ownership");
        t.storage->verify();state[t.slot]=0;
    }
};
struct TPipe {
    static inline thread_local TPipe* active=nullptr;
    std::vector<std::shared_ptr<Storage>> buffers;
    TPipe(){active=this;}
    template<TPosition P> void InitBuffer(TBuf<P>& buffer,size_t bytes){
        ensure(bytes>0 && bytes%32==0,"UB allocation not 32B aligned");
        buffer.storage=std::make_shared<Storage>(bytes);buffers.push_back(buffer.storage);current->stats.ub+=bytes;
    }
    template<TPosition P,int D> void InitBuffer(TQue<P,D>& queue,int count,size_t bytes){
        ensure(count==2,"Q002 requires two slots");
        for(int i=0;i<count;++i){
            ensure(bytes>0 && bytes%32==0,"queue allocation not aligned");
            auto p=std::make_shared<Storage>(bytes);queue.slots.push_back(p);buffers.push_back(p);current->stats.ub+=bytes;
        }
        queue.state.assign(count,0);
    }
    int FetchEventID(HardEvent){return 0;}
    void verify(){for(const auto& b:buffers)b->verify();}
};
inline TPipe* GetTPipePtr(){return TPipe::active;}
struct DataCopyExtParams { uint16_t blockCount;uint32_t blockLen;uint32_t srcStride,dstStride,rsv; };
template<class T> struct DataCopyPadExtParams { bool enable;uint8_t left,right;T value; };
template<class T> void DataCopyPad(LocalTensor<T> dst,GlobalTensor<T> src,DataCopyExtParams p,DataCopyPadExtParams<T> pad){
    ensure(p.blockCount==1 && p.srcStride==0 && p.dstStride==0,"unsupported DMA descriptor in model");
    ensure(p.blockLen%sizeof(T)==0,"DMA element length");
    size_t count=p.blockLen/sizeof(T);dst.check(count+pad.left+pad.right);checkGm(src.ptr,p.blockLen);
    ensure(pad.left==0,"unexpected left padding");
    std::memcpy(dst.ptr(),src.ptr,p.blockLen);
    for(size_t i=count;i<count+pad.right;++i)dst.ptr()[i]=pad.value;
    ++current->stats.copies;current->stats.copyBytes+=p.blockLen;
}
template<class T> void DataCopyPad(GlobalTensor<T> dst,LocalTensor<T> src,DataCopyExtParams p){
    ensure(p.blockCount==1 && p.srcStride==0 && p.dstStride==0,"unsupported store DMA descriptor");
    src.check(p.blockLen/sizeof(T));checkGm(dst.ptr,p.blockLen);std::memcpy(dst.ptr,src.ptr(),p.blockLen);
    ++current->stats.copies;current->stats.copyBytes+=p.blockLen;
}
template<class D,class S> void Cast(LocalTensor<D> dst,LocalTensor<S> src,RoundMode mode,int32_t n){
    dst.check(n);src.check(n);
    for(int32_t i=0;i<n;++i){
        float value=static_cast<float>(src.ptr()[i]);
        if(mode==RoundMode::CAST_RINT)value=std::nearbyint(value);
        dst.ptr()[i]=static_cast<D>(value);
    }
}
#define QUS_BINARY(name,op) inline void name(LocalTensor<float> d,LocalTensor<float> a,LocalTensor<float> b,int n){d.check(n);a.check(n);b.check(n);for(int i=0;i<n;++i)d.ptr()[i]=a.ptr()[i] op b.ptr()[i];}
QUS_BINARY(Mul,*)
QUS_BINARY(Div,/)
QUS_BINARY(Add,+)
#undef QUS_BINARY
inline void Maxs(LocalTensor<float>d,LocalTensor<float>a,float x,int n){d.check(n);a.check(n);for(int i=0;i<n;++i)d.ptr()[i]=std::max(a.ptr()[i],x);}
inline void Mins(LocalTensor<float>d,LocalTensor<float>a,float x,int n){d.check(n);a.check(n);for(int i=0;i<n;++i)d.ptr()[i]=std::min(a.ptr()[i],x);}
}
"""
HARNESS=r"""
#include <iostream>
#include <thread>
#include <exception>
#include "op_kernel/quant_update_scatter.h"
#undef QUANTUPDATESCATTER_H
#define NsQuantUpdateScatter BaselineQuantUpdateScatter
#include "BASELINE_HEADER"
#undef NsQuantUpdateScatter

struct Gm {
    std::vector<uint8_t> raw;
    size_t bytes;
    explicit Gm(size_t n):raw(n+64,0xD5),bytes(n){
        auto first=reinterpret_cast<uintptr_t>(data());AscendC::gmRegions.push_back({first,first+n});
    }
    uint8_t* data(){return raw.data()+32;}
    void verify()const{for(size_t i=0;i<32;++i)AscendC::ensure(raw[i]==0xD5 && raw[bytes+32+i]==0xD5,"GM guard overwritten");}
};
struct Shape {int batch,middle,rows,varRows,width,indexDelta,cores;};
struct Result {std::vector<int8_t> output;size_t maxUb=0,copies=0,flags=0,queueAlloc=0;};

template<bool Baseline,class U,class S,class I,class Z>
Result execute(Shape shape,bool inplace,bool zpPresent,bool reciprocal){
    AscendC::gmRegions.clear();
    size_t total=static_cast<size_t>(shape.batch)*shape.middle*shape.varRows*shape.width;
    size_t updateCount=static_cast<size_t>(shape.batch)*shape.middle*shape.rows*shape.width;
    int indexCount=std::max(0,shape.batch+shape.indexDelta);
    Gm var(total),out(total),updates(updateCount*sizeof(U)),scales(shape.width*sizeof(S)),zp(shape.width*sizeof(Z)),indices(indexCount*sizeof(I));
    auto* in=reinterpret_cast<int8_t*>(var.data());
    std::vector<int8_t> expected(total);
    for(size_t i=0;i<total;++i)expected[i]=in[i]=static_cast<int8_t>(static_cast<int>(i%255)-128);
    auto* index=reinterpret_cast<I*>(indices.data());
    for(int i=0;i<indexCount;++i)index[i]=static_cast<I>(i%(shape.varRows-shape.rows+1));
    auto* u=reinterpret_cast<U*>(updates.data());
    auto* scale=reinterpret_cast<S*>(scales.data());auto* zero=reinterpret_cast<Z*>(zp.data());
    for(int c=0;c<shape.width;++c){scale[c]=static_cast<S>(c%3==0?0.5f:c%3==1?1.0f:2.0f);zero[c]=static_cast<Z>(static_cast<float>(c%7)-3);}
    for(size_t i=0;i<updateCount;++i){
        float value=(static_cast<int>(i%2401)-1200)*0.25f;
        u[i]=static_cast<U>(value);
        int c=i%shape.width,k=(i/shape.width)%shape.rows,m=(i/(shape.width*shape.rows))%shape.middle,p=i/(shape.width*shape.rows*shape.middle);
        if(p>=indexCount)continue;
        float quant=reciprocal?static_cast<float>(u[i])*static_cast<float>(scale[c]):static_cast<float>(u[i])/static_cast<float>(scale[c]);
        if(zpPresent)quant+=static_cast<float>(zero[c]);
        quant=std::min(127.0f,std::max(-128.0f,quant));
        size_t dst=((p*shape.middle+m)*shape.varRows+index[p]+k)*shape.width+c;
        expected[dst]=static_cast<int8_t>(std::nearbyint(quant));
    }
    QuantUpdateScatterTilingData td{};
    td.totalNum=total;td.batchSize=shape.batch;td.midSize=shape.middle;td.updateAxis=shape.rows;
    td.varAxis=shape.varRows;td.lastDim=shape.width;td.indexElements=indexCount;
    td.coreNum=shape.cores;td.blockFactor=std::max(1,(shape.batch+shape.cores-1)/shape.cores);
    constexpr size_t ub=32768;
    int capacity=static_cast<int>((ub-8192-16384)/(Baseline?27:30)/32*32);
    capacity=std::min(8192,capacity);
    td.ubFactor=std::min((std::max(1,shape.width)+31)/32*32,capacity);
    td.hasZeroPoints=zpPresent;td.reciprocalScale=reciprocal;td.indicesIsInt64=sizeof(I)==8;
    using Op=typename std::conditional<Baseline,BaselineQuantUpdateScatter::QuantUpdateScatter<U,S,I,Z>,NsQuantUpdateScatter::QuantUpdateScatter<U,S,I,Z>>::type;
    AscendC::Barrier barrier(shape.cores);
    std::vector<AscendC::CoreContext> context(shape.cores);
    std::vector<std::thread> threads;
    std::exception_ptr error;std::mutex errorMutex;
    uint8_t* destination=inplace?var.data():out.data();
    for(int core=0;core<shape.cores;++core){
        context[core].index=core;context[core].barrier=&barrier;
        threads.emplace_back([&,core]{
            AscendC::current=&context[core];
            try{
                Op op;op.Init(var.data(),indices.data(),updates.data(),scales.data(),zpPresent?zp.data():nullptr,destination,&td);
                op.Process();AscendC::GetTPipePtr()->verify();
            }catch(...){std::lock_guard<std::mutex> lock(errorMutex);if(!error)error=std::current_exception();barrier.cancel();}
        });
    }
    for(auto& thread:threads)thread.join();
    if(error)std::rethrow_exception(error);
    for(size_t i=0;i<total;++i)if(reinterpret_cast<int8_t*>(destination)[i]!=expected[i])throw std::runtime_error("actual kernel class mismatched scalar oracle at offset "+std::to_string(i));
    var.verify();out.verify();updates.verify();scales.verify();zp.verify();indices.verify();
    Result result;result.output.assign(reinterpret_cast<int8_t*>(destination),reinterpret_cast<int8_t*>(destination)+total);
    for(const auto& c:context){result.maxUb=std::max(result.maxUb,c.stats.ub);result.copies+=c.stats.copies;result.flags+=c.stats.flags;result.queueAlloc+=c.stats.queueAlloc;}
    AscendC::ensure(result.maxUb+8192<=ub,"actual UB allocations exceed host budget");
    return result;
}

int cases=0,queued=0,direct=0;size_t largestUb=0;
template<class U,class S,class I,class Z>
void matrix(){
    const Shape shapes[]={
        {1,1,1,2,1,0,1},{1,1,1,2,7,0,4},{3,2,2,5,17,0,4},
        {3,2,2,5,31,-1,4},{3,2,2,5,32,2,4},{3,2,2,5,33,0,4},
        {1,4,3,7,65,0,2},{7,1,1,4,129,-7,3},
        {1,1,1,2048,33,0,4},{2,1,2,5,8193,0,2},
        {2048,1,1,2,1,0,1},{2049,1,1,2,1,0,1},
        {0,1,1,2,33,0,4},{3,2,0,5,33,0,4},
        {3,2,2,5,0,0,4},{65,1,1,4,7,0,8}
    };
    for(const Shape& shape:shapes)for(int inplace=0;inplace<2;++inplace)for(int zp=0;zp<2;++zp)for(int reciprocal=0;reciprocal<2;++reciprocal){
        auto baseline=execute<true,U,S,I,Z>(shape,inplace,zp,reciprocal);
        auto candidate=execute<false,U,S,I,Z>(shape,inplace,zp,reciprocal);
        AscendC::ensure(baseline.output==candidate.output,"candidate differs from saved passing baseline");
        if(candidate.queueAlloc)++queued;else ++direct;
        largestUb=std::max(largestUb,candidate.maxUb);++cases;
    }
}
int main(){
    try{
        matrix<half,float,int32_t,int32_t>();matrix<half,float,int64_t,int32_t>();
        matrix<bfloat16_t,bfloat16_t,int32_t,bfloat16_t>();matrix<bfloat16_t,bfloat16_t,int64_t,bfloat16_t>();
        auto oldCopy=execute<true,half,float,int32_t,int32_t>({1,1,1,2048,33,0,4},false,true,false);
        auto newCopy=execute<false,half,float,int32_t,int32_t>({1,1,1,2048,33,0,4},false,true,false);
        AscendC::ensure(newCopy.copies<oldCopy.copies,"larger scratch did not reduce copy calls in constructed narrow-row case");
        std::cout<<"Actual kernel-class CPU model: "<<cases<<" candidate/baseline/oracle comparisons passed.\n";
        std::cout<<"Cases exercising queues="<<queued<<", direct/no-update="<<direct<<", maximum per-core UB bytes="<<largestUb<<".\n";
        std::cout<<"Constructed copy case DMA calls: baseline="<<oldCopy.copies<<", candidate="<<newCopy.copies<<" (counts only, no timing claim).\n";
        std::cout<<"Checks: GM/UB bounds and guards, queue ownership, 4 dtype signatures, tails, empty/short/long indices, multi-tile widths, copy-only cores, inplace, optional zp, 2048/2049 cache boundary.\n";
        std::cout<<"Synchronous CPU API model only. CANN compilation, asynchronous pipeline correctness and NPU performance remain NOT_RUN.\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
"""

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cxx',default=shutil.which('g++'))
    args=parser.parse_args()
    if not args.cxx:raise SystemExit('C++17 compiler required')
    baseline=SEAT/'explorations/Q002/baseline/op_kernel/quant_update_scatter.h'
    if not baseline.is_file():raise SystemExit('Saved Q002 baseline required')
    with tempfile.TemporaryDirectory(prefix='build-q002-',dir=Path(__file__).parent) as directory:
        build=Path(directory)
        files={'kernel_operator.h':STUB,'kernel_tiling/kernel_tiling.h':'#pragma once\n','ascendc/host_api/tiling/template_argument.h':'#pragma once\n#define ASCENDC_TPL_ARGS_DECL(...)\n#define ASCENDC_TPL_SEL(...)\n'}
        for relative,content in files.items():
            p=build/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')
        saved=build/'q001_baseline'
        saved.mkdir()
        for header in baseline.parent.glob('*.h'):
            shutil.copyfile(header,saved/header.name)
        source=build/'kernel_model.cpp'
        source.write_text(HARNESS.replace('BASELINE_HEADER','q001_baseline/quant_update_scatter.h'),encoding='utf-8')
        exe=build/('kernel_model.exe' if os.name=='nt' else 'kernel_model')
        command=[args.cxx,'-std=c++17','-O2','-pthread','-Wall','-Wextra','-Wno-unused-parameter','-I',str(build),'-I',str(CODE),str(source),'-o',str(exe)]
        run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        print(run.stdout,end='')
        if run.returncode:raise SystemExit(run.returncode)
        env=os.environ.copy();env['PATH']=str(Path(args.cxx).parent)+os.pathsep+env.get('PATH','')
        run=subprocess.run([str(exe)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=90)
        print(run.stdout,end='')
        if run.returncode:raise SystemExit(run.returncode)

if __name__=='__main__':main()
