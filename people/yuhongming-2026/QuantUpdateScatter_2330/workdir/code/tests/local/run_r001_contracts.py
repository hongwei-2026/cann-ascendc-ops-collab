"""Compile actual registration/entry/error-helper source against minimal API stubs.
These are source-contract checks, not a CANN compilation or NPU emulation.
"""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import tempfile

CODE = Path(__file__).resolve().parents[2]
STUBS = {
"register/op_def_registry.h": r"""
#pragma once
#include <initializer_list>
#include <map>
#include <string>
#include <vector>
namespace ge {
enum DataType { DT_INT8, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_FLOAT };
enum Format { FORMAT_ND };
}
namespace ops {
constexpr int REQUIRED=1, OPTIONAL=0;
struct Port {
    std::vector<ge::DataType> types;
    std::vector<ge::Format> formats, unknownFormats;
    Port& ParamType(int) { return *this; }
    Port& DataType(std::initializer_list<ge::DataType> v) { types=v; return *this; }
    Port& Format(std::initializer_list<ge::Format> v) { formats=v; return *this; }
    Port& UnknownShapeFormat(std::initializer_list<ge::Format> v) { unknownFormats=v; return *this; }
    Port& AutoContiguous() { return *this; }
};
struct Attribute {
    Attribute& String() { return *this; }
    Attribute& AttrType(int) { return *this; }
    Attribute& Int(int) { return *this; }
    Attribute& Bool(int) { return *this; }
};
struct Core { Core& AddConfig(const char*) { return *this; } };
class OpDef {
public:
    std::map<std::string,Port> inputs, outputs;
    explicit OpDef(const char*) {}
    Port& Input(const char* n) { return inputs[n]; }
    Port& Output(const char* n) { return outputs[n]; }
    Attribute& Attr(const char*) { return attribute; }
    Core& AICore() { return core; }
private:
    Attribute attribute;
    Core core;
};
}
#define OP_ADD(T) static_assert(true, "registration stub")
""",
"acl/acl.h": r"""
#pragma once
#include <cstddef>
#include <cstdint>
struct aclTensor {};
struct aclScalar {};
struct aclTensorList {};
using aclDataType=int;
using aclrtStream=void*;
constexpr int ACL_FORMAT_ND=0, ACL_MEM_MALLOC_HUGE_FIRST=0;
constexpr int ACL_MEMCPY_HOST_TO_DEVICE=0, ACL_MEMCPY_DEVICE_TO_HOST=1;
extern const char* test_error_detail;
extern int test_error_reads;
inline const char* aclGetRecentErrMsg() { ++test_error_reads; return test_error_detail; }
inline int aclrtMalloc(void** p,std::size_t,int) { *p=nullptr; return 0; }
inline int aclrtMemcpy(void*,std::size_t,const void*,std::size_t,int) { return 0; }
inline int aclrtFree(void*) { return 0; }
inline aclTensor* aclCreateTensor(const int64_t*,std::size_t,aclDataType,const int64_t*,int64_t,int,const int64_t*,std::size_t,void*) { return nullptr; }
inline aclScalar* aclCreateScalar(const void*,aclDataType) { return nullptr; }
inline int aclDestroyScalar(aclScalar*) { return 0; }
inline int aclDestroyTensor(aclTensor*) { return 0; }
inline aclTensorList* aclCreateTensorList(aclTensor* const*,std::size_t) { return nullptr; }
inline int aclDestroyTensorList(aclTensorList*) { return 0; }
""",
"aclnn/aclnn_base.h": "#pragma once\n",
}
HARNESS = r"""
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>
#include "op_host/quant_update_scatter_def.cpp"
#include "op_kernel/quant_update_scatter_tiling_data.h"

// The real entry is included below. Only device primitives and its class body
// are replaced, so we can observe the C++ template type selected by each branch.
#define QUANTUPDATESCATTER_H
#define __global__
#define __aicore__
#define REGISTER_TILING_DEFAULT(T)
#define GET_TILING_DATA_WITH_STRUCT(T, name, ptr) const T& name=*reinterpret_cast<const T*>(ptr)
using GM_ADDR=unsigned char*;
struct half {};
struct bfloat16_t {};
int recorded_mode=-1, recorded_bits=0, process_calls=0;
namespace NsQuantUpdateScatter {
template<class U,class S,class I,class Z> struct QuantUpdateScatter {
    void Init(GM_ADDR,GM_ADDR,GM_ADDR,GM_ADDR,GM_ADDR,GM_ADDR,const QuantUpdateScatterTilingData*) {
        recorded_mode=std::is_same<U,half>::value ? 0 : 1;
        recorded_bits=sizeof(I)*8;
        if constexpr(std::is_same<U,half>::value) {
            static_assert(std::is_same<S,float>::value && std::is_same<Z,int32_t>::value);
        } else {
            static_assert(std::is_same<S,bfloat16_t>::value && std::is_same<Z,bfloat16_t>::value);
        }
    }
    void Process() { ++process_calls; }
};
}
#include "op_kernel/quant_update_scatter.cpp"
const char* test_error_detail=nullptr;
int test_error_reads=0;
#include "examples/sample_support.h"

int main() {
    ops::QuantUpdateScatter definition("QuantUpdateScatter");
    assert(definition.inputs.size()==5 && definition.outputs.size()==1);
    const ge::DataType expected[4][5]={
        {ge::DT_INT8,ge::DT_INT32,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_INT32},
        {ge::DT_INT8,ge::DT_INT64,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16},
        {ge::DT_INT8,ge::DT_INT32,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16},
        {ge::DT_INT8,ge::DT_INT64,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_INT32}
    };
    const char* names[]={"var","indices","updates","quant_scales","quant_zero_points"};
    for(int row=0;row<4;++row) {
        for(int input=0;input<5;++input) {
            const auto& p=definition.inputs.at(names[input]);
            assert(p.types.size()==4 && p.formats.size()==4 && p.unknownFormats.size()==4);
            assert(p.types[row]==expected[row][input]);
        }
        assert(definition.outputs.at("var_out").types[row]==ge::DT_INT8);
    }
    const auto& out=definition.outputs.at("var_out");
    assert(out.types.size()==4 && out.formats.size()==4 && out.unknownFormats.size()==4);
    for(int mode=0;mode<2;++mode) {
        for(int wide=0;wide<2;++wide) {
            QuantUpdateScatterTilingData td{};
            td.indicesIsInt64=wide;
            process_calls=0;
            auto* address=reinterpret_cast<GM_ADDR>(&td);
            if(mode==0) quant_update_scatter<0>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,address);
            else quant_update_scatter<1>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,address);
            assert(recorded_mode==mode && recorded_bits==(wide?64:32) && process_calls==1);
        }
    }
    check(0,"success");
    assert(test_error_reads==0);
    test_error_detail="signature rejected: fp16 updates with int64 indices";
    try { check(161002,"aclnnQuantUpdateScatterGetWorkspaceSize"); assert(false); }
    catch(const std::runtime_error& e) {
        const std::string message=e.what();
        assert(message.find("aclnnQuantUpdateScatterGetWorkspaceSize")!=std::string::npos);
        assert(message.find("161002")!=std::string::npos);
        assert(message.find(test_error_detail)!=std::string::npos);
    }
    test_error_detail=nullptr;
    try { QUS_ACL_CHECK(161002); assert(false); }
    catch(const std::runtime_error& e) { assert(std::string(e.what()).find("161002")!=std::string::npos); }
    std::cout<<"Actual-source contract checks OK: 4 signatures, 4 index dispatches, ACL call/detail diagnostics.\n";
    std::cout<<"Minimal API stubs only; NOT a CANN build or NPU precision test.\n";
}
"""


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cxx',default=shutil.which('g++') or shutil.which('clang++'))
    args=parser.parse_args()
    if not args.cxx:
        raise SystemExit('A C++17 compiler is required for source-contract checks.')
    with tempfile.TemporaryDirectory(prefix='build-r001-',dir=Path(__file__).parent) as directory:
        build=Path(directory)
        for relative,content in STUBS.items():
            path=build/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding='utf-8')
        source=build/'contracts.cpp'
        source.write_text(HARNESS,encoding='utf-8')
        executable=build/('contracts.exe' if os.name=='nt' else 'contracts')
        command=[args.cxx,'-std=c++17','-Wall','-Wextra','-Wno-unused-parameter','-pedantic','-I',str(build),'-I',str(CODE),str(source),'-o',str(executable)]
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        print(result.stdout,end='')
        if result.returncode:
            raise SystemExit(result.returncode)
        env=os.environ.copy()
        env['PATH']=str(Path(args.cxx).parent)+os.pathsep+env.get('PATH','')
        result=subprocess.run([str(executable)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        print(result.stdout,end='')
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
