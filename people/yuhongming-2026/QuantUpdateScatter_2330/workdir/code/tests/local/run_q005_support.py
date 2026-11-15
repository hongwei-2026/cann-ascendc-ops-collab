"""Compile/run the actual Q005 runner against synchronous CPU ACL stubs.
This checks diagnostic plumbing only, NOT CANN/NPU precision or scheduling.
"""
from pathlib import Path
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / 'tools'))
import profile_q005 as q005

ACL = r'''
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
struct aclTensor { std::vector<int64_t> shape; int type; void* data; };
struct aclScalar {};
struct aclTensorList {};
using aclDataType=int;
using aclrtStream=void*;
enum { ACL_INT8, ACL_INT32, ACL_INT64, ACL_FLOAT16, ACL_FLOAT, ACL_BF16 };
constexpr int ACL_FORMAT_ND=0, ACL_MEM_MALLOC_HUGE_FIRST=0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE=0, ACL_MEMCPY_DEVICE_TO_HOST=1;
inline const char* aclGetRecentErrMsg(){return "CPU diagnostic stub";}
inline int aclInit(const char*){return 0;}
inline int aclrtSetDevice(int){return 0;}
inline int aclrtCreateStream(aclrtStream* s){*s=reinterpret_cast<void*>(1);return 0;}
inline int aclrtSynchronizeStream(aclrtStream){return 0;}
inline int aclrtDestroyStream(aclrtStream){return 0;}
inline int aclrtResetDevice(int){return 0;}
inline int aclFinalize(){return 0;}
inline int aclrtMalloc(void** p,size_t n,int){*p=std::malloc(n);return *p?0:1;}
inline int aclrtMemcpy(void* d,size_t cap,const void* s,size_t n,int){if(n>cap)return 1;std::memcpy(d,s,n);return 0;}
inline int aclrtFree(void* p){std::free(p);return 0;}
inline aclTensor* aclCreateTensor(const int64_t* dims,size_t rank,int type,const int64_t*,int64_t,int,const int64_t*,size_t,void* data){return new aclTensor{{dims,dims+rank},type,data};}
inline int aclDestroyTensor(aclTensor* t){delete t;return 0;}
inline aclScalar* aclCreateScalar(const void*,int){return new aclScalar;}
inline int aclDestroyScalar(aclScalar* p){delete p;return 0;}
inline aclTensorList* aclCreateTensorList(aclTensor* const*,size_t){return new aclTensorList;}
inline int aclDestroyTensorList(aclTensorList* p){delete p;return 0;}
'''
API = r'''
#pragma once
#include <acl/acl.h>
struct aclOpExecutor { aclTensor *v,*i,*u,*s,*z,*o; bool reciprocal; };
inline float value(const aclTensor* t,size_t n){
    if(t->type==ACL_FLOAT)return static_cast<float*>(t->data)[n];
    if(t->type==ACL_INT32)return static_cast<int32_t*>(t->data)[n];
    uint16_t x=static_cast<uint16_t*>(t->data)[n];
    if(t->type==ACL_BF16){uint32_t bits=uint32_t(x)<<16;float f;std::memcpy(&f,&bits,4);return f;}
    const int exp=(x>>10)&31,mant=x&1023;
    float f=exp?std::ldexp(float(1024+mant),exp-25):std::ldexp(float(mant),-24);
    return (x&0x8000)?-f:f;
}
inline int aclnnQuantUpdateScatterGetWorkspaceSize(aclTensor* v,aclTensor* i,aclTensor* u,aclTensor* s,aclTensor* z,char*,int64_t axis,int64_t qaxis,bool rec,aclTensor* o,uint64_t* ws,aclOpExecutor** e){
    if(axis!=-2||qaxis!=-1||v->shape.size()!=4)return 161002;
    if(!((u->type==ACL_FLOAT16&&s->type==ACL_FLOAT&&(!z||z->type==ACL_INT32))||
         (u->type==ACL_BF16&&s->type==ACL_BF16&&(!z||z->type==ACL_BF16))))return 161002;
    // Alternate required size to exercise workspace growth/reuse.
    static int calls=0;*ws=(++calls%2)?64:128;
    *e=new aclOpExecutor{v,i,u,s,z,o,rec};return 0;
}
inline int aclnnQuantUpdateScatter(void* ws,uint64_t n,aclOpExecutor* e,aclrtStream){
    if(!ws||n<64)return 1;
    auto v=e->v;auto u=e->u;auto o=e->o;
    size_t b=v->shape[0],m=v->shape[1],vr=v->shape[2],w=v->shape[3],ur=u->shape[2];
    if(v->data!=o->data)std::memcpy(o->data,v->data,b*m*vr*w);
    for(size_t p=0;p<b;++p){
        size_t idx=e->i->type==ACL_INT32?static_cast<int32_t*>(e->i->data)[p]:static_cast<int64_t*>(e->i->data)[p];
        for(size_t middle=0;middle<m;++middle)for(size_t k=0;k<ur;++k)for(size_t c=0;c<w;++c){
            float x=value(u,((p*m+middle)*ur+k)*w+c),scale=value(e->s,c);
            x=e->reciprocal?x*scale:x/scale;if(e->z)x+=value(e->z,c);
            x=std::min(127.0f,std::max(-128.0f,x));
            static_cast<int8_t*>(o->data)[((p*m+middle)*vr+idx+k)*w+c]=static_cast<int8_t>(std::nearbyint(x));
        }
    }
    delete e;return 0;
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxx', default=shutil.which('g++'))
    args = parser.parse_args()
    if not args.cxx:
        parser.error('C++17 compiler required')
    q005.self_test()
    with tempfile.TemporaryDirectory(prefix='q005-runner-', dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        files = {'acl/acl.h': ACL, 'aclnn/aclnn_base.h': '#pragma once\n',
                 'aclnn_quant_update_scatter.h': API}
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
        executable = root / ('profile_case.exe' if os.name == 'nt' else 'profile_case')
        command = [args.cxx, '-std=c++17', '-O2', '-Wall', '-Wextra',
                   '-I', str(root), '-I', str(CODE / 'examples'),
                   str(CODE / 'examples/profile_quant_update_scatter.cpp'), '-o', str(executable)]
        env = os.environ.copy()
        env['PATH'] = str(Path(args.cxx).parent) + os.pathsep + env.get('PATH', '')
        subprocess.run(command, env=env, check=True)
        specs = q005.generate(root / 'cases')
        for case in specs:
            folder = root / 'cases' / case['id']
            command = q005.runner_arguments(executable, folder, case, 0, 1, 2)
            subprocess.run(command, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            result = q005.verify(root / 'cases', case)
            if not result['passed']:
                raise RuntimeError(str(result))
        invalid = q005.runner_arguments(executable, root / 'cases' / specs[0]['id'], specs[0], 0, 1, 2)
        invalid[9] = '48'
        result = subprocess.run(invalid, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert result.returncode == 2 and b'INDEX_BITS' in result.stdout
        print(f'Actual Q005 runner CPU checks: {len(specs)} cases x 3 launches exact-match; '
              'all four signatures, optional zp, reciprocal, inplace, workspace growth, invalid bits rejected.')
        print('Synchronous stub API only. Device compilation, precision, metrics and performance NOT_RUN.')


if __name__ == '__main__':
    main()
