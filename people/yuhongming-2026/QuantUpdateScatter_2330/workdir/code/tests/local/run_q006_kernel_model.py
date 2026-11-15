"""Q006 actual-class Q002/oracle comparisons and logical dependency checks.
No CANN compilation, hardware timing or NPU scheduling claim is made.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

CODE = Path(__file__).resolve().parents[2]
SEAT = CODE.parents[1]
spec = importlib.util.spec_from_file_location('q002', Path(__file__).with_name('run_q002_kernel_model.py'))
q002 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q002)

DEPENDENCIES = r'''
// Per-engine vector clocks model happens-before, not elapsed time.
// Values execute synchronously; an unordered UB read/write is still rejected.
struct Clock {
    size_t value[3]{0,0,0};
    void join(const Clock& other){for(int i=0;i<3;++i)value[i]=std::max(value[i],other.value[i]);}
    bool contains(const Clock& other)const{for(int i=0;i<3;++i)if(value[i]<other.value[i])return false;return true;}
};
struct Dependency {
    Clock engines[3],events[7]; // 0=MTE2, 1=V, 2=MTE3; scalar events only order cached indices.
    Clock stamp(int engine){++engines[engine].value[engine];return engines[engine];}
    static int from(HardEvent event){switch(event){
        case HardEvent::MTE2_MTE3:case HardEvent::MTE2_V:case HardEvent::MTE2_S:return 0;
        case HardEvent::V_MTE2:case HardEvent::V_MTE3:return 1;
        default:return 2;
    }}
    static int to(HardEvent event){switch(event){
        case HardEvent::MTE2_MTE3:case HardEvent::V_MTE3:return 2;
        case HardEvent::MTE3_MTE2:case HardEvent::V_MTE2:return 0;
        default:return 1;
    }}
    void set(HardEvent event){events[static_cast<int>(event)]=engines[from(event)];}
    void wait(HardEvent event){engines[to(event)].join(events[static_cast<int>(event)]);}
};
inline thread_local Dependency dep;
inline thread_local bool dependencyChecks=true;
'''


def model_stub():
    stub = q002.STUB
    anchor = 'struct Stats {'
    pos = stub.index(anchor)
    stub = stub[:pos] + DEPENDENCIES + stub[pos:]
    stub = stub.replace('size_t ub=0, copies=0, copyBytes=0, flags=0, queueAlloc=0;',
                        'size_t ub=0, copies=0, copyBytes=0, flags=0, queueAlloc=0, maxInputReady=0, ubChecks=0;')
    stub = stub.replace('template<HardEvent> inline void SetFlag(event_t) { ++current->stats.flags; }',
                        'template<HardEvent Event> inline void SetFlag(event_t) { ++current->stats.flags;dep.set(Event); }')
    stub = stub.replace('template<HardEvent> inline void WaitFlag(event_t) {}',
                        'template<HardEvent Event> inline void WaitFlag(event_t) {dep.wait(Event);}')
    stub = stub.replace('    size_t bytes;\n    explicit Storage',
                        '    size_t bytes;\n    Clock lastWrite,readers[3],available,published;\n    explicit Storage')
    marker = 'template<class T> struct LocalTensor {'
    accesses = r'''
inline void readStorage(const std::shared_ptr<Storage>& storage,int engine){
    if(dependencyChecks)ensure(dep.engines[engine].contains(storage->lastWrite),"unordered UB read (missing producer dependency)");
    storage->readers[engine]=dep.stamp(engine);++current->stats.ubChecks;
}
inline void writeStorage(const std::shared_ptr<Storage>& storage,int engine){
    if(dependencyChecks){
        ensure(dep.engines[engine].contains(storage->lastWrite),"unordered UB write after write");
        for(int i=0;i<3;++i)ensure(dep.engines[engine].contains(storage->readers[i]),"UB reused before consumer dependency");
    }
    storage->lastWrite=dep.stamp(engine);++current->stats.ubChecks;
}
'''
    stub = stub.replace(marker, accesses + marker)
    stub = stub.replace('state[i]=1;++current->stats.queueAlloc;',
                        'state[i]=1;dep.engines[Position==TPosition::VECIN?0:1].join(slots[i]->available);++current->stats.queueAlloc;')
    stub = stub.replace('        state[t.slot]=2;ready.push_back(t.slot);', '''        t.storage->published=dep.engines[Position==TPosition::VECIN?0:1];
        state[t.slot]=2;ready.push_back(t.slot);
        if constexpr(Position==TPosition::VECIN)current->stats.maxInputReady=std::max(current->stats.maxInputReady,ready.size());''')
    stub = stub.replace('int i=ready.front();ready.pop_front();state[i]=1;',
                        'int i=ready.front();ready.pop_front();state[i]=1;dep.engines[Position==TPosition::VECIN?1:2].join(slots[i]->published);')
    stub = stub.replace('        t.storage->verify();state[t.slot]=0;',
                        '        t.storage->verify();t.storage->available=dep.engines[Position==TPosition::VECIN?1:2];state[t.slot]=0;')
    stub = stub.replace('    std::vector<std::shared_ptr<Storage>> buffers;',
                        '    std::vector<std::shared_ptr<Storage>> buffers;\n    std::vector<std::function<void()>> queueChecks;')
    stub = stub.replace('        queue.state.assign(count,0);', '''        queue.state.assign(count,0);
        queueChecks.push_back([&queue]{
            ensure(queue.ready.empty(),"prefetch queue not drained at kernel end");
            for(int state:queue.state)ensure(state==0,"queue tensor still owned at kernel end");
        });''')
    stub = stub.replace('void verify(){for(const auto& b:buffers)b->verify();}',
                        'void verify(){for(const auto& b:buffers)b->verify();for(const auto& check:queueChecks)check();}')
    stub = stub.replace('#include <deque>', '#include <deque>\n#include <functional>')
    stub = stub.replace('    std::memcpy(dst.ptr(),src.ptr,p.blockLen);',
                        '    writeStorage(dst.storage,0);\n    std::memcpy(dst.ptr(),src.ptr,p.blockLen);')
    stub = stub.replace('    src.check(p.blockLen/sizeof(T));checkGm(dst.ptr,p.blockLen);std::memcpy',
                        '    src.check(p.blockLen/sizeof(T));readStorage(src.storage,2);checkGm(dst.ptr,p.blockLen);std::memcpy')
    stub = stub.replace('    dst.check(n);src.check(n);',
                        '    dst.check(n);src.check(n);readStorage(src.storage,1);writeStorage(dst.storage,1);')
    stub = stub.replace('d.check(n);a.check(n);b.check(n);for(',
                        'd.check(n);a.check(n);b.check(n);readStorage(a.storage,1);readStorage(b.storage,1);writeStorage(d.storage,1);for(')
    stub = stub.replace('d.check(n);a.check(n);for(',
                        'd.check(n);a.check(n);readStorage(a.storage,1);writeStorage(d.storage,1);for(')
    return stub


NEGATIVE_CHECKS = r'''
void missingDependencyProbe(){
    AscendC::CoreContext context;AscendC::current=&context;AscendC::dep=AscendC::Dependency{};
    auto storage=std::make_shared<AscendC::Storage>(32);
    AscendC::writeStorage(storage,0);
    bool rejected=false;
    try{AscendC::readStorage(storage,1);}catch(const std::runtime_error&){rejected=true;}
    AscendC::ensure(rejected,"negative probe: missing MTE2->V accepted");
    AscendC::dep.engines[1].join(AscendC::dep.engines[0]);
    AscendC::readStorage(storage,1);
    rejected=false;
    try{AscendC::writeStorage(storage,0);}catch(const std::runtime_error&){rejected=true;}
    AscendC::ensure(rejected,"negative probe: input slot reused before V reader completed");
    AscendC::dep.engines[0].join(AscendC::dep.engines[1]);
    AscendC::writeStorage(storage,0);
}
'''


def harness():
    source = q002.HARNESS
    source = source.replace('size_t maxUb=0,copies=0,flags=0,queueAlloc=0;',
                            'size_t maxUb=0,copies=0,flags=0,queueAlloc=0,maxInputReady=0,ubChecks=0;')
    source = source.replace('constexpr size_t ub=32768;', 'const size_t ub=currentTestUb;')
    source = source.replace('(Baseline?27:30)', '30')
    source = source.replace('struct Shape {', 'size_t currentTestUb=32768;\nstruct Shape {')
    source = source.replace('    AscendC::gmRegions.clear();', '    AscendC::gmRegions.clear();AscendC::dep=AscendC::Dependency{};')
    source = source.replace('for(int c=0;c<shape.width;++c){scale[c]=static_cast<S>(c%3==0?0.5f:c%3==1?1.0f:2.0f);zero[c]=static_cast<Z>(static_cast<float>(c%7)-3);}',
                            'const float scaleValues[]={0.5f,1.0f,2.0f,2.0f/3.0f,1.0f/2.5002f,0.3f,1.7f};for(int c=0;c<shape.width;++c){scale[c]=static_cast<S>(scaleValues[c%7]);zero[c]=static_cast<Z>(static_cast<float>(c%7)-3);}')

    source = source.replace('result.queueAlloc+=c.stats.queueAlloc;',
                            'result.queueAlloc+=c.stats.queueAlloc;result.maxInputReady=std::max(result.maxInputReady,c.stats.maxInputReady);result.ubChecks+=c.stats.ubChecks;')
    source = source.replace('int cases=0,queued=0,direct=0;', NEGATIVE_CHECKS + '\nint cases=0,queued=0,direct=0;')
    source = source.replace('{3,2,2,5,0,0,4},{65,1,1,4,7,0,8}', '''{3,2,2,5,0,0,4},{65,1,1,4,7,0,8},
        {1,3,1,5,33,0,1},{1,1,2,5,33,0,1},{1,2,3,3,65,0,1},
        {7,3,1,4,129,-2,3},{2,1,1,3,8192,0,1},{1,2,2,4,16385,0,1},
        {1,16,128,256,33,0,8},{1,2,16,32,8193,0,8}''')
    source = source.replace('        if(candidate.queueAlloc)++queued;else ++direct;', '''        AscendC::ensure(candidate.maxUb==baseline.maxUb,"Q006 changed Q002 UB allocations");
        AscendC::ensure(candidate.copies==baseline.copies,"Q006 changed Q002 DMA count");
        AscendC::ensure(baseline.maxInputReady<=1,"Q002 trace should enqueue one input at a time");
        if(candidate.maxInputReady==2)++prefetched;
        if(candidate.queueAlloc)++queued;else ++direct;''')
    source = source.replace('int cases=0,queued=0,direct=0;', 'int cases=0,queued=0,direct=0,prefetched=0;')
    begin = source.index('        auto oldCopy=')
    end = source.index('    }catch(', begin)
    source = source[:begin] + r'''
        AscendC::ensure(prefetched>0,"candidate did not issue two input rows before DeQue");
        std::cout<<"Q006 actual kernel-class CPU model: "<<cases<<" Q006/Q002/oracle comparisons passed.\n";
        std::cout<<"Queued="<<queued<<", direct/no-update="<<direct<<", cases reaching two pending input rows="<<prefetched<<".\n";
        std::cout<<"Checks: four signatures, clip/RINT, tails, short/empty/long indices, 2048/2049 cache, middle gaps, full-axis updates, copy-only cores, inplace, zp/reciprocal, multiple UB capacities.\n";
        std::cout<<"Q002 DMA counts/UB sizes unchanged. Logical engine dependency model and negative probes passed; no hardware timing model.\n";
        std::cout<<"CANN build, NPU asynchronous behavior, device precision and performance NOT_RUN.\n";
''' + source[end:]
    source = source.replace('''        matrix<half,float,int32_t,int32_t>();matrix<half,float,int64_t,int32_t>();
        matrix<bfloat16_t,bfloat16_t,int32_t,bfloat16_t>();matrix<bfloat16_t,bfloat16_t,int64_t,bfloat16_t>();''', '''        missingDependencyProbe();
        for(size_t ub: {size_t{32768},size_t{65536},size_t{196608},size_t{262144}}){
            currentTestUb=ub;
            matrix<half,float,int32_t,int32_t>();matrix<half,float,int64_t,int32_t>();
            matrix<bfloat16_t,bfloat16_t,int32_t,bfloat16_t>();matrix<bfloat16_t,bfloat16_t,int64_t,bfloat16_t>();
        }''')
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxx', default=shutil.which('g++'))
    args = parser.parse_args()
    if not args.cxx:
        parser.error('C++17 compiler required')
    baseline = SEAT / 'explorations/Q006/baseline/op_kernel'
    candidate = (CODE / 'op_kernel/quant_update_scatter.h').read_text(encoding='utf-8')
    assert 'TQue<TPosition::VECIN, 2>' in candidate and 'TQue<TPosition::VECOUT, 2>' in candidate
    old = (baseline / 'quant_update_scatter.h').read_text(encoding='utf-8')
    for method in ('Init', 'CopyVarToOut', 'LoadQuantParams', 'QuantizeTile', 'LoadIndices', 'ReadIndex'):
        def body(text):
            start = text.index('::' + method + '(')
            start = text.index('{', start)
            return text[start:text.index('\n}\n', start) + 3]
        assert body(candidate) == body(old), 'Q006 unexpectedly changed ' + method
    with tempfile.TemporaryDirectory(prefix='q006-model-', dir=Path(__file__).parent) as temporary:
        build = Path(temporary)
        files = {'kernel_operator.h': model_stub(), 'kernel_tiling/kernel_tiling.h': '#pragma once\n',
                 'ascendc/host_api/tiling/template_argument.h': '#pragma once\n#define ASCENDC_TPL_ARGS_DECL(...)\n#define ASCENDC_TPL_SEL(...)\n'}
        for relative, content in files.items():
            path = build / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        saved = build / 'q002_baseline'
        saved.mkdir()
        for header in baseline.glob('*.h'):
            shutil.copyfile(header, saved / header.name)
        source = build / 'q006_model.cpp'
        source.write_text(harness().replace('BASELINE_HEADER', 'q002_baseline/quant_update_scatter.h'), encoding='utf-8')
        executable = build / ('q006_model.exe' if os.name == 'nt' else 'q006_model')
        command = [args.cxx, '-std=c++17', '-O2', '-ffp-contract=off', '-pthread', '-Wall', '-Wextra', '-Wno-unused-parameter',
                   '-I', str(build), '-I', str(CODE), str(source), '-o', str(executable)]
        env = os.environ.copy()
        env['PATH'] = str(Path(args.cxx).parent) + os.pathsep + env.get('PATH', '')
        for command in (command, [str(executable)]):
            result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding='utf-8', errors='replace', timeout=180)
            print(result.stdout, end='', flush=True)
            if result.returncode:
                return result.returncode
    print('Source contracts: Init/UB, copy, quant arithmetic, parameter loads and index cache identical to Q002.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
