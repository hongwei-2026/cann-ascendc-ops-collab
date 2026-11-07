/*!
 * \file pad_v3_grad_def.cpp
 * \brief PadV3Grad 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class PadV3Grad : public OpDef {
public:
    explicit PadV3Grad(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("paddings")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("mode")
            .AttrType(OPTIONAL)
            .String("reflect");
        this->Attr("paddings_contiguous")
            .AttrType(OPTIONAL)
            .Bool(1);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(PadV3Grad);
} // namespace ops
