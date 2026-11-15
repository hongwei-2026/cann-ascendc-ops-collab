"""Validate actual host resource expressions and Q003/Q002 kernel classes.
Synchronous CPU API stubs only: no CANN build, device timing or asynchronous validation.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

CODE=Path(__file__).resolve().parents[2]
SEAT=CODE.parents[1]
spec=importlib.util.spec_from_file_location('q002model',Path(__file__).with_name('run_q002_kernel_model.py'))
q002=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q002)

RESOURCE_PREAMBLE=r'''
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
namespace ge { enum DataType {DT_FLOAT16,DT_BF16}; }
struct TestDesc {ge::DataType type;ge::DataType GetDataType()const{return type;}};
template<class T>T CeilDiv(T a,T b){return a/b+(a%b!=0);}
#define OP_CHECK_IF(condition,log,ret) do{if(condition)throw std::runtime_error("UB plan rejected");}while(false)
struct Plan {int64_t used,factor,tile,cache,perElem;};
inline Plan actualResourcePlan(int64_t batchSize,int64_t totalNum,int64_t lastDim,
    int64_t coreNum,uint64_t ubSize,int64_t indexElements,int64_t indicesIsInt64,
    bool isFp16,bool hasZeroPoints){
    const TestDesc desc{isFp16?ge::DT_FLOAT16:ge::DT_BF16};
    const TestDesc* updatesDesc=&desc;
'''

MODEL_EXTRA=r'''
size_t currentTestUb=32768;
void allocationMatrix(){
    size_t plans=0;
    const int64_t batches[]={0,1,3,64,2048,2049,4097,131073};
    const int64_t lengths[]={0,1,33,8192,8193,65537};
    const int64_t devices[]={32768,65536,196608,262144};
    const int64_t cores[]={1,3,64};
    for(auto batch:batches)for(auto width:lengths)for(auto ub:devices)for(auto core:cores)
    for(int wide=0;wide<2;++wide)for(int bf=0;bf<2;++bf)for(int hasZp=0;hasZp<2;++hasZp)
    for(int mode=0;mode<3;++mode){
        const int64_t indices=mode==0?0:mode==1?std::max(int64_t{0},batch-1):batch+2;
        auto p=actualResourcePlan(batch,batch*width*2,width,core,ub,indices,wide,!bf,hasZp);
        AscendC::ensure(p.tile>0 && p.tile%32==0 && p.tile<=8192,"invalid planned tile");
        const int64_t expectedElem=18+(bf?2:0)+(hasZp?(bf?6:8):0);
        AscendC::ensure(p.perElem==expectedElem,"dtype resource coefficient wrong");
        AscendC::ensure(p.tile*p.perElem+p.cache+8192<=ub,"host UB plan exceeds device capacity");
        for(int64_t block=0;block<p.used;++block){
            const int64_t first=block*p.factor;
            const int64_t active=std::max(int64_t{0},std::min(p.factor,batch-first));
            int64_t cached=std::max(int64_t{0},std::min(active,indices-first));
            if(cached>2048)cached=0;
            const int64_t bytes=CeilDiv(cached*(wide?int64_t{8}:int64_t{4}),int64_t{32})*32;
            AscendC::ensure(bytes<=p.cache,"tail cache exceeds reserved host bound");
        }
        ++plans;
    }
    std::cout<<"Actual host resource block: "<<plans<<" allocation plans and launched-core cache bounds passed.\n";
}
'''


def build_plan_source(host):
    constants=host[host.index('constexpr int64_t kReserveBytes'):host.index('int64_t ShapeSize')]
    block=host[host.index('    const int64_t batchCores'):host.index('    QuantUpdateScatterTilingData* tiling')]
    return RESOURCE_PREAMBLE+constants+block+'    return {usedCores,blockFactor,ubFactor,cacheBytes,bytesPerElem};\n}\n#undef OP_CHECK_IF\n'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cxx',default=shutil.which('g++'))
    args=parser.parse_args()
    if not args.cxx:raise SystemExit('C++17 compiler required')
    baseline=SEAT/'explorations/Q003/baseline/op_kernel/quant_update_scatter.h'
    if not baseline.is_file():raise SystemExit('Q002 passing baseline missing')
    host=(CODE/'op_host/quant_update_scatter_tiling.cpp').read_text(encoding='utf-8')
    harness=q002.HARNESS
    harness=harness.replace('#include "op_kernel/quant_update_scatter.h"','#include "q003_plan.h"\n#include "op_kernel/quant_update_scatter.h"')
    old='''    constexpr size_t ub=32768;
    int capacity=static_cast<int>((ub-8192-16384)/(Baseline?27:30)/32*32);
    capacity=std::min(8192,capacity);
    td.ubFactor=std::min((std::max(1,shape.width)+31)/32*32,capacity);'''
    new='''    constexpr size_t ub=32768;
    if constexpr (Baseline) {
        int capacity=static_cast<int>((ub-8192-16384)/30/32*32);
        capacity=std::min(8192,capacity);
        td.ubFactor=std::min((std::max(1,shape.width)+31)/32*32,capacity);
    } else {
        const auto plan=actualResourcePlan(shape.batch,total,shape.width,shape.cores,ub,indexCount,sizeof(I)==8,
            std::is_same<U,half>::value,zpPresent);
        td.coreNum=plan.used;td.blockFactor=plan.factor;td.ubFactor=plan.tile;
    }
    const int launchCores=static_cast<int>(td.coreNum);'''
    if old not in harness:raise SystemExit('Q002 fixture contract changed')
    harness=harness.replace(old,new)
    harness=harness.replace('AscendC::Barrier barrier(shape.cores);','AscendC::Barrier barrier(launchCores);').replace('context(shape.cores);','context(launchCores);').replace('core<shape.cores;','core<launchCores;')
    harness=harness.replace('int cases=0,queued=0,direct=0;',MODEL_EXTRA+'\nint cases=0,queued=0,direct=0;')
    harness=harness.replace('constexpr size_t ub=32768;', 'const size_t ub=currentTestUb;')
    harness=harness.replace('struct Shape {int batch,middle,rows,varRows,width,indexDelta,cores;};', 'extern size_t currentTestUb;\nstruct Shape {int batch,middle,rows,varRows,width,indexDelta,cores;};')
    harness=harness.replace('{3,2,2,5,0,0,4},{65,1,1,4,7,0,8}', '{3,2,2,5,0,0,4},{65,1,1,4,7,0,8},\n        {1,1,2,5,7905,0,1},{1,1,1,2,8192,0,1},{3,1,2,5,16385,0,3}')

    harness=harness.replace('''        matrix<half,float,int32_t,int32_t>();matrix<half,float,int64_t,int32_t>();
        matrix<bfloat16_t,bfloat16_t,int32_t,bfloat16_t>();matrix<bfloat16_t,bfloat16_t,int64_t,bfloat16_t>();''','''        allocationMatrix();
        for(size_t budget: {size_t{32768},size_t{65536},size_t{196608},size_t{262144}}) {
            currentTestUb=budget;
            matrix<half,float,int32_t,int32_t>();matrix<half,float,int64_t,int32_t>();
            matrix<bfloat16_t,bfloat16_t,int32_t,bfloat16_t>();matrix<bfloat16_t,bfloat16_t,int64_t,bfloat16_t>();
        }
        currentTestUb=32768;''')
    harness=harness.replace('''        auto oldCopy=execute<true,half,float,int32_t,int32_t>({1,1,1,2048,33,0,4},false,true,false);
        auto newCopy=execute<false,half,float,int32_t,int32_t>({1,1,1,2048,33,0,4},false,true,false);
        AscendC::ensure(newCopy.copies<oldCopy.copies,"larger scratch did not reduce copy calls in constructed narrow-row case");''','''        const Shape wide{1,1,2,5,513,0,1};
        auto oldCopy=execute<true,half,float,int64_t,int32_t>(wide,true,true,false);
        auto newCopy=execute<false,half,float,int64_t,int32_t>(wide,true,true,false);
        AscendC::ensure(newCopy.copies<oldCopy.copies,"new resource plan did not reduce constructed wide-row tile loads");''')
    harness=harness.replace('Constructed copy case DMA calls:','Constructed wide-row DMA calls:')
    harness=harness.replace('candidate/baseline/oracle comparisons passed.','Q003/Q002/scalar-oracle comparisons passed.')
    with tempfile.TemporaryDirectory(prefix='build-q003-',dir=Path(__file__).parent) as directory:
        build=Path(directory)
        files={'kernel_operator.h':q002.STUB,'kernel_tiling/kernel_tiling.h':'#pragma once\n','ascendc/host_api/tiling/template_argument.h':'#pragma once\n#define ASCENDC_TPL_ARGS_DECL(...)\n#define ASCENDC_TPL_SEL(...)\n','q003_plan.h':build_plan_source(host)}
        for relative,content in files.items():
            p=build/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content,encoding='utf-8')
        saved=build/'q002_baseline';saved.mkdir()
        for header in baseline.parent.glob('*.h'):shutil.copyfile(header,saved/header.name)
        source=build/'model.cpp';source.write_text(harness.replace('BASELINE_HEADER','q002_baseline/quant_update_scatter.h'),encoding='utf-8')
        exe=build/('model.exe' if os.name=='nt' else 'model')
        command=[args.cxx,'-std=c++17','-O2','-pthread','-Wall','-Wextra','-Wno-unused-parameter','-I',str(build),'-I',str(CODE),str(source),'-o',str(exe)]
        run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        print(run.stdout,end='')
        if run.returncode:raise SystemExit(run.returncode)
        env=os.environ.copy();env['PATH']=str(Path(args.cxx).parent)+os.pathsep+env.get('PATH','')
        run=subprocess.run([str(exe)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=90)
        print(run.stdout,end='')
        if run.returncode:raise SystemExit(run.returncode)

if __name__=='__main__':main()