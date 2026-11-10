#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "p_relu_grad_update_shape_utils.h"
namespace ops {
static ge::graphStatus InferShapePReluGradUpdate(gert::InferShapeContext* context){
 if(!context)return ge::GRAPH_FAILED;
 const auto* g=context->GetInputShape(0);const auto* x=context->GetInputShape(1);const auto* w=context->GetInputShape(2);
 auto* dx=context->GetOutputShape(0);auto* up=context->GetOutputShape(1);
 if(!g||!x||!w||!dx||!up)return ge::GRAPH_FAILED;
 const auto dims=[](const gert::Shape& s){JadeYaraShape::Dims d;for(size_t j=0;j<s.GetDimNum();++j)d.push_back(s.GetDim(j));return d;};
 const auto gd=dims(*g),xd=dims(*x),wd=dims(*w);
 if(JadeYaraShape::UnknownRank(gd)||JadeYaraShape::UnknownRank(xd)||JadeYaraShape::UnknownRank(wd)){
  // Sentinel propagation only; unknown-rank graph compilation remains unverified.
  dx->SetDimNum(1);dx->SetDim(0,-2);*up=*dx;return ge::GRAPH_SUCCESS;
 }
 const auto* desc=context->GetInputDesc(1);const bool blocked=desc&&ge::GetPrimaryFormat(desc->GetOriginFormat())==ge::FORMAT_NC1HWC0;
 JadeYaraShape::Dims out,mapped;
 if(!JadeYaraShape::Normalize(gd,xd,wd,blocked,false,out,mapped))return ge::GRAPH_FAILED;
 dx->SetDimNum(out.size());for(size_t j=0;j<out.size();++j)dx->SetDim(j,out[j]);*up=*dx;
 return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypePReluGradUpdate(gert::InferDataTypeContext* context){
 if(!context)return ge::GRAPH_FAILED;
 const ge::DataType dtype=context->GetInputDataType(0);
 if(dtype==ge::DT_UNDEFINED)return ge::GRAPH_FAILED;
 if(context->SetOutputDataType(0,dtype)!=ge::GRAPH_SUCCESS)return ge::GRAPH_FAILED;
 return context->SetOutputDataType(1,dtype);
}
IMPL_OP_INFERSHAPE(PReluGradUpdate)
 .InferShape(InferShapePReluGradUpdate)
 .InferDataType(InferDataTypePReluGradUpdate);
}
