#!/usr/bin/env python3
"""Compile actual M009 Process/SetWorkspace bodies against a checked CPU shim.

Checks reduction control flow and scratch sizes; does not emulate AscendC math,
asynchronous DMA, memory visibility, SDK compilation, or NPU performance.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[5]
CODE = ROOT / 'people/hongwei-2026/MultilabelMarginLoss_2327/workdir/code'


def definition(text, marker):
    start = text.index(marker)
    begin = text.index('{', start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


HARNESS = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#define __aicore__
constexpr int64_t REDUCTION_NONE=0, REDUCTION_MEAN=1, REDUCTION_SUM=2;
constexpr int64_t RED_CHUNK=512;
void check(bool v) { if (!v) throw std::runtime_error("check failed"); }
int currentBlock=0, syncCount=0, scratchReads=0, scratchWrites=0, yWrites=0;
int GetBlockIdx() { return currentBlock; }
void SyncAll() { ++syncCount; }
constexpr int PIPE_ALL=0;
template<int> void PipeBarrier() {}
template<class T> struct LocalTensor {
    T *p; int64_t n;
    T GetValue(int64_t i) const { check(i>=0 && i<n); return p[i]; }
};
struct Buf {
    float p[RED_CHUNK]{};
    template<class T> LocalTensor<T> Get() { return {reinterpret_cast<T*>(p),RED_CHUNK}; }
};
struct Gm {
    float *p=nullptr; int64_t n=0;
    Gm operator[](int64_t i) const { check(p && i>=0 && i<n); return {p+i,n-i}; }
};
struct DataCopyExtParams { uint16_t blocks; uint32_t bytes,src,dst; uint8_t rsv; };
template<class T> struct DataCopyPadExtParams { bool pad; uint8_t left,right; T v; };
void DataCopyPad(LocalTensor<float> dst, Gm src, DataCopyExtParams args,
    DataCopyPadExtParams<float> pad) {
    check(src.p && args.blocks==1 && args.bytes%sizeof(float)==0);
    int64_t n=args.bytes/sizeof(float);
    check(n<=src.n && n+pad.left+pad.right<=dst.n);
    std::copy(src.p,src.p+n,dst.p+pad.left); ++scratchReads;
}
namespace ge { using graphStatus=int; constexpr int GRAPH_SUCCESS=0,GRAPH_FAILED=1; }
namespace gert { struct TilingContext {
    size_t bytes=999; bool nullWorkspace=false;
    size_t* GetWorkspaceSizes(int n) { check(n==1); return nullWorkspace?nullptr:&bytes; }
}; }
#define OP_CHECK_NULL_WITH_CONTEXT(ctx,ptr) if (!(ptr)) return ge::GRAPH_FAILED
// HOST_FUNCTIONS
std::vector<float> rowLosses, output, scratch;
std::vector<int> processed;
template<class T> class MultilabelMarginLoss {
public:
    int64_t rowCount_=0,classNum_=1,targetLen_=1,reduction_=0,usedCores_=1;
    int64_t rowStart_=0,batchSize_=0;
    Buf redBuf_; Gm rowLossGm_;
    int64_t AlignUp(int64_t n,int64_t a) { return (n+a-1)/a*a; }
    float ProcessOneRow(int64_t i) { check(++processed.at(i)==1); return rowLosses.at(i); }
    void StoreY(int64_t i,float x) { output.at(i)=x; ++yWrites; }
    void StoreRowLoss(int64_t i,float x) {
        check(rowLossGm_.p && i>=0 && i<rowLossGm_.n);
        rowLossGm_.p[i]=x; ++scratchWrites;
    }
    void Process();
};
// PROCESS_FUNCTION
int cases=0;
void run(int64_t n,int64_t reduction,int64_t cores) {
    ++cases;
    rowLosses.resize(n); processed.assign(n,0);
    // Values expose wrong mean denominator and changed row order/precision.
    for (int64_t i=0;i<n;++i) rowLosses[i]=(i%7==0)?10000.25f:(i+1)*0.03125f;
    output.assign(reduction==0?n:1,-999.f);
    scratch.assign(n,-999.f);
    int64_t rows=(n+cores-1)/cores;
    syncCount=scratchReads=scratchWrites=yWrites=0;
    // Replay every other block before core0; this is not a concurrency model.
    for (int64_t step=1;step<=cores;++step) {
        currentBlock=static_cast<int>(step%cores);
        MultilabelMarginLoss<float> op;
        op.batchSize_=n; op.reduction_=reduction; op.usedCores_=cores;
        op.rowStart_=currentBlock*rows;
        op.rowCount_=std::max<int64_t>(0,std::min(rows,n-op.rowStart_));
        bool needsScratch=(reduction!=0 && cores>1);
        op.rowLossGm_={needsScratch?scratch.data():nullptr,needsScratch?n:0};
        op.Process();
    }
    for (int v:processed) check(v==1);
    float expected=0;
    for (float v:rowLosses) expected+=v;
    if (reduction==1) expected/=static_cast<float>(n);
    if (reduction==0) {
        check(output==rowLosses && yWrites==n);
    } else { check(output[0]==expected && yWrites==1); }
    if (reduction==0 || cores==1) {
        check(syncCount==0 && scratchReads==0 && scratchWrites==0);
    } else {
        check(syncCount==cores && scratchWrites==n);
        check(scratchReads==(n+RED_CHUNK-1)/RED_CHUNK);
    }
}
int main() {
    for (int reduction=0;reduction<3;++reduction) {
        for (int n:{1,4,9,100,513,1025}) {
            run(n,reduction,1);
            if(n>1) run(n,reduction,std::min(n,64));
        }
        // Explicit idle cores must still enter hardware SyncAll.
        run(100,reduction,64);
    }
    gert::TilingContext ctx;
    for (int64_t n:{1,4,8,9,100,513,1025}) {
        for (int reduction=0;reduction<3;++reduction) {
            for (int cores:{1,4}) {
                check(SetWorkspace(&ctx,n,reduction,cores)==ge::GRAPH_SUCCESS);
                size_t expected=reduction==0||cores==1?0:((n*4+31)/32)*32;
                check(ctx.bytes==expected);
            }
        }
    }
    ctx.nullWorkspace=true;
    check(SetWorkspace(&ctx,4,1,1)==ge::GRAPH_FAILED);
    std::cout << "Actual Process flow: " << cases << " cases PASS; workspace: 42 cases PASS\n";
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=shutil.which('g++'))
    args = parser.parse_args()
    if not args.cxx:
        raise SystemExit('A C++17 compiler is required for this CPU check.')
    host = (CODE / 'op_host/multilabel_margin_loss_tiling.cpp').read_text(encoding='utf-8')
    kernel = (CODE / 'op_kernel/multilabel_margin_loss.h').read_text(encoding='utf-8')
    functions = definition(host, 'int64_t CeilDivI64(') + '\n' + definition(
        host, 'ge::graphStatus SetWorkspace(')
    process = definition(kernel,
        'template <typename T>\n__aicore__ inline void MultilabelMarginLoss<T>::Process()')
    source = HARNESS.replace('// HOST_FUNCTIONS', functions).replace('// PROCESS_FUNCTION', process)
    with tempfile.TemporaryDirectory(prefix='m009-flow-') as tmp:
        path = Path(tmp)
        cpp = path / 'flow.cpp'
        exe = path / 'flow.exe'
        cpp.write_text(source, encoding='utf-8')
        subprocess.run([args.cxx, '-std=c++17', '-O2', '-Wall', '-Wextra', str(cpp), '-o', str(exe)], check=True)
        subprocess.run([str(exe)], check=True)
    print('CPU control-flow replay only; CANN/NPU validation NOT RUN.')


if __name__ == '__main__':
    main()
