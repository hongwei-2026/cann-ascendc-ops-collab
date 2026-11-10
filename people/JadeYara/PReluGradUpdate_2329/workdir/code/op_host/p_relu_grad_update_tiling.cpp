/*!
 * \file p_relu_grad_update_tiling.cpp
 * \brief Runtime tiling for PReluGradUpdate.
 */

#include <algorithm>
#include "p_relu_grad_update_shape_utils.h"
#include "../op_kernel/p_relu_grad_update_general_tiling.h"
#include <cstdint>
#include "../op_kernel/p_relu_grad_update_gather_tiling.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/p_relu_grad_update_tiling_data.h"
#include "../op_kernel/p_relu_grad_update_tiling_key.h"

namespace optiling {
namespace {

// The kernel has no GM scratch data. P026's temporary 16MB workspace was based
// on the old 507035 diagnosis and adds millisecond-scale fixed overhead.
constexpr uint32_t WS_SYS_SIZE = 0;
constexpr uint64_t UB_USAGE_NUMERATOR = 8U;
constexpr uint64_t UB_USAGE_DENOMINATOR = 10U;
// P100 每元素 UB（BUFFER_NUM=1）：
//   float: 4×4 + neg/fg(2×4) + mask≈1 = 25
//   fp16/bf16→fp32: 4×2 + float grads/features/dx/update/neg/fg(6×4) + mask≈1 = 33
constexpr int64_t FLOAT_BYTES_PER_ELEMENT = 25;
constexpr int64_t LOW_PRECISION_BYTES_PER_ELEMENT = 33;
constexpr int64_t CMP_ALIGN_FLOAT = 64;   // 中间计算在 fp32 上，Compare 按 64 齐

int64_t CeilDivI64(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t AlignUpI64(int64_t value, int64_t alignment)
{
    return CeilDivI64(value, alignment) * alignment;
}

int64_t AlignDownI64(int64_t value, int64_t alignment)
{
    return (value / alignment) * alignment;
}

bool SameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}


bool GatherMagic(uint64_t d,uint64_t x,uint64_t& m,uint64_t& shift)
{
 if(d==1){m=1;shift=0;return true;}
 uint64_t scale=1;shift=0;while(scale<x*d){scale<<=1;++shift;}
 m=(scale+d-1)/d;return x*m<=0x7fffffffULL;
}
bool TryGather(gert::TilingContext* context,int64_t total,int64_t channel,int64_t inner,int64_t weightLen,
 ge::DataType dtype,int64_t blockFactor,int64_t cores,uint64_t available)
{
 if(total<=0||weightLen<=1||channel<=1||inner<=0||inner>256)return false;
 const int64_t width=dtype==ge::DT_FLOAT?4:2;
 if(inner==1 && channel%(32/width)==0)return false;
 const int64_t cap=dtype==ge::DT_FLOAT?4096:3072;
 const int64_t tile=AlignUpI64(std::min(blockFactor,cap),128);
 const int64_t segment=inner==1?channel:inner;
 if(blockFactor<8*segment||tile<8*segment)return false;
 const uint64_t budget=dtype==ge::DT_FLOAT?32*static_cast<uint64_t>(tile)+AlignUpI64(tile,512)+1024:43*static_cast<uint64_t>(tile)+704;
 if(budget>available)return false;
 const uint64_t qcap=(inner+tile-2)/inner+1;const bool cache=static_cast<uint64_t>(channel)<=qcap;
 uint64_t ms=0,ks=0,mc=0,kc=0;
 if(!GatherMagic(inner,inner+tile-2,ms,ks))return false;
 if(cache&&!GatherMagic(channel,channel+qcap-2,mc,kc))return false;
 auto* t=context->GetTilingData<PReluGradUpdateGatherTilingData>();if(t==nullptr)return false;
 t->totalNum=total;t->blockSpan=blockFactor;t->tileLen=tile;t->channelNum=channel;t->innerSize=inner;
 t->qCap=qcap;t->cacheWeights=cache?1:0;t->magicS=ms;t->shiftS=ks;t->magicC=mc;t->shiftC=kc;
 context->SetBlockDim(static_cast<uint32_t>(cores));
 if(dtype==ge::DT_FLOAT)context->SetTilingKey(GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_GATHER));
 else if(dtype==ge::DT_FLOAT16)context->SetTilingKey(GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_GATHER));
 else context->SetTilingKey(GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_GATHER));
 return true;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context, "platform info is null");
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    // Atlas A2 上 Select 的 selMode=1/2 需要 8K Unified Buffer 作内部暂存，
    // 必须在 GetCoreMemSize 之前声明，否则返回的 ubSize 会把这 8K 算进可用空间。
    platform.ReserveLocalMemory(platform_ascendc::ReservedSize::RESERVED_SIZE_8K);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum <= 0 || ubSize == 0) {
        OP_LOGE(context, "invalid platform data: coreNum=%ld, ubSize=%lu", coreNum, ubSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        OP_LOGE(context, "workspace pointer is null");
        return ge::GRAPH_FAILED;
    }
    workspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus PReluGradUpdateLegacyTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* gradsStorage = context->GetInputShape(0);
    const gert::StorageShape* featuresStorage = context->GetInputShape(1);
    const gert::StorageShape* weightsStorage = context->GetInputShape(2);
    const auto* inputDesc = context->GetInputDesc(0);
    if (gradsStorage == nullptr || featuresStorage == nullptr || weightsStorage == nullptr || inputDesc == nullptr) {
        OP_LOGE(context, "input shape or dtype is null");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& grads = gradsStorage->GetStorageShape();
    const gert::Shape& features = featuresStorage->GetStorageShape();
    const gert::Shape& weights = weightsStorage->GetStorageShape();
    if (grads.GetDimNum() < 2 || !SameShape(grads, features) || weights.GetDimNum() != 1) {
        OP_LOGE(context, "invalid ranks or mismatched grads/features shapes");
        return ge::GRAPH_FAILED;
    }

    const int64_t totalNum = grads.GetShapeSize();
    const int64_t channelNum = grads.GetDim(1);
    const int64_t weightLen = weights.GetDim(0);
    if (totalNum < 0 || channelNum <= 0 || (weightLen != 1 && weightLen != channelNum)) {
        OP_LOGE(context, "invalid shape values: total=%ld, channel=%ld, weightLen=%ld",
            totalNum, channelNum, weightLen);
        return ge::GRAPH_FAILED;
    }

    int64_t innerSize = 1;
    for (size_t i = 2; i < grads.GetDimNum(); ++i) {
        const int64_t dim = grads.GetDim(i);
        if (dim <= 0) {
            OP_LOGE(context, "all runtime dimensions must be positive");
            return ge::GRAPH_FAILED;
        }
        innerSize *= dim;
    }
    const int64_t sampleSize = channelNum * innerSize;

    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = 0;
    int64_t bytesPerElement = 0;
    int64_t cmpAlignElems = CMP_ALIGN_FLOAT;
    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
        bytesPerElement = FLOAT_BYTES_PER_ELEMENT;
        cmpAlignElems = CMP_ALIGN_FLOAT;
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT);
#if PRELU_ENABLE_FP16
    } else if (dtype == ge::DT_FLOAT16) {
        typeSize = 2;
        bytesPerElement = LOW_PRECISION_BYTES_PER_ELEMENT;
        cmpAlignElems = CMP_ALIGN_FLOAT;
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16);
#endif
#if PRELU_ENABLE_BF16
    } else if (dtype == ge::DT_BF16) {
        typeSize = 2;
        bytesPerElement = LOW_PRECISION_BYTES_PER_ELEMENT;
        cmpAlignElems = CMP_ALIGN_FLOAT;
        tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16);
#endif
    } else {
        OP_LOGE(context, "unsupported dtype=%d", static_cast<int32_t>(dtype));
        return ge::GRAPH_FAILED;
    }

    const bool vectorWeights = weightLen > 1 && innerSize == 1;
    if (vectorWeights) {
        bytesPerElement += typeSize;  // contiguous channel weights queue
        if (dtype == ge::DT_FLOAT) {
            tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_S1);
        } else if (dtype == ge::DT_FLOAT16) {
            tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_S1);
        } else {
            tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_S1);
        }
    }

    if (!vectorWeights) {
        bytesPerElement -= sizeof(float);  // J008 featuresFloat reuse
    }

    uint64_t ubSize = 0;
    int64_t hardwareCoreNum = 0;
    if (GetPlatformInfo(context, ubSize, hardwareCoreNum) != ge::GRAPH_SUCCESS ||
        SetWorkspace(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const int64_t blockElements = 32 / typeSize;
    const uint64_t usableUb = ubSize * UB_USAGE_NUMERATOR / UB_USAGE_DENOMINATOR;
    int64_t maxTile = static_cast<int64_t>(usableUb / static_cast<uint64_t>(bytesPerElement));
    // kernel 会把 count 向上对齐到 cmpAlignElems；maxTile 取同倍数保证对齐后不超 UB。
    maxTile = AlignDownI64(maxTile, cmpAlignElems);
    if (maxTile < cmpAlignElems) {
        OP_LOGE(context, "UB is too small for one 256B-aligned tile");
        return ge::GRAPH_FAILED;
    }

    int64_t usedCores = 1;
    int64_t blockFactor = totalNum;
    int64_t tileLen = totalNum == 0 ? blockElements : std::min(totalNum, maxTile);
    int64_t rowsPerTile = 0;
    int64_t splitBatchGroups = 0;

    if (totalNum > 0) {
        const int64_t targetCoreNum = weightLen == 1
            ? std::min(hardwareCoreNum, std::max<int64_t>(4, CeilDivI64(totalNum, 1024)))
            : hardwareCoreNum;
        blockFactor = AlignUpI64(CeilDivI64(totalNum, targetCoreNum), blockElements);
        usedCores = std::min(targetCoreNum, CeilDivI64(totalNum, blockFactor));
        tileLen = std::min(blockFactor, maxTile);
        // A single weight is scalar even when C=1. Only distinct channel
        // weights use one-channel Muls tiles, or a contiguous C segment when S=1.
        if (weightLen > 1 && innerSize > 0) {
            if (!(vectorWeights && channelNum % blockElements == 0)) {
                tileLen = std::min(tileLen, vectorWeights ? channelNum : innerSize);
            }
        }
    }

    // Reuse the same weight phase across tiles without increasing loop count.
    if (vectorWeights && channelNum % blockElements == 0 && blockFactor > tileLen
        && tileLen > channelNum) {
        const int64_t residentTile = AlignDownI64(tileLen, channelNum);
        if (CeilDivI64(blockFactor, residentTile) == CeilDivI64(blockFactor, tileLen)) {
            tileLen = residentTile;
            rowsPerTile = residentTile / channelNum;
        }
    }

    // Separate binaries preserve the measured fallback instruction paths.
    if (vectorWeights && rowsPerTile > 0) {
        if (dtype == ge::DT_FLOAT) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_RESIDENT);
        else if (dtype == ge::DT_FLOAT16) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_RESIDENT);
        else tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_RESIDENT);
    } else if (vectorWeights && channelNum % blockElements == 0 && tileLen > channelNum) {
        if (dtype == ge::DT_FLOAT) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_MERGED);
        else if (dtype == ge::DT_FLOAT16) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_MERGED);
        else tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_MERGED);
    }

    // A core owns one channel tile and a contiguous batch range.
    // Reuse weights where a complete channel vector cannot fit in one tile.
    if (vectorWeights && channelNum % blockElements == 0 && channelNum > maxTile) {
        const int64_t cTiles = CeilDivI64(channelNum, maxTile);
        if (cTiles <= std::min<int64_t>(3, hardwareCoreNum)) {
            const int64_t batchGroups = hardwareCoreNum / cTiles;
            const int64_t batchSize = totalNum / channelNum;
            if (batchSize >= 2 * hardwareCoreNum) {
                tileLen = AlignUpI64(CeilDivI64(channelNum, cTiles), CMP_ALIGN_FLOAT);
                splitBatchGroups = batchGroups;
                usedCores = cTiles * batchGroups;
                if (dtype == ge::DT_FLOAT) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_SPLITC);
                else if (dtype == ge::DT_FLOAT16) tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_SPLITC);
                else tilingKey = GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_SPLITC);
            }
        }
    }

    if (TryGather(context,totalNum,channelNum,innerSize,weightLen,dtype,blockFactor,usedCores,usableUb)) return ge::GRAPH_SUCCESS;

    PReluGradUpdateTilingData* tiling = context->GetTilingData<PReluGradUpdateTilingData>();
    if (tiling == nullptr) {
        OP_LOGE(context, "tiling data pointer is null");
        return ge::GRAPH_FAILED;
    }
    tiling->totalNum = totalNum;
    tiling->coreNum = usedCores;
    tiling->blockFactor = blockFactor;
    tiling->tileLen = tileLen;
    tiling->channelNum = channelNum;
    tiling->innerSize = innerSize;
    tiling->sampleSize = sampleSize;
    tiling->weightsPerChannel = weightLen > 1 ? 1 : 0;
    tiling->rowsPerTile = rowsPerTile;
    tiling->splitBatchGroups = splitBatchGroups;

    context->SetBlockDim(static_cast<uint32_t>(usedCores));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}


// General input normalization is deliberately separate from the unchanged J069
// fastpath selector. Existing valid positive ND shapes receive its exact metadata.
static JadeYaraShape::Dims StorageDims(const gert::StorageShape* s) {
    JadeYaraShape::Dims d;
    if (s) for(size_t j=0;j<s->GetStorageShape().GetDimNum();++j) d.push_back(s->GetStorageShape().GetDim(j));
    return d;
}
static ge::graphStatus PReluGradUpdateTilingFunc(gert::TilingContext* context) {
    if(!context)return ge::GRAPH_FAILED;
    for(size_t i=0;i<3;++i)if(!context->GetInputShape(i)||!context->GetInputDesc(i))return ge::GRAPH_FAILED;
    for(size_t i=0;i<2;++i)if(!context->GetOutputShape(i)||!context->GetOutputDesc(i))return ge::GRAPH_FAILED;
    const auto dtype=context->GetInputDesc(0)->GetDataType();
    const auto format=ge::GetPrimaryFormat(context->GetInputDesc(1)->GetStorageFormat());
    const auto wf=ge::GetPrimaryFormat(context->GetInputDesc(2)->GetStorageFormat());
    if(ge::GetPrimaryFormat(context->GetInputDesc(0)->GetStorageFormat())!=format)return ge::GRAPH_FAILED;
    if(format!=ge::FORMAT_ND && format!=ge::FORMAT_NCHW && format!=ge::FORMAT_NC1HWC0)return ge::GRAPH_FAILED;
    // GE may propagate NCHW to plain ND weights without changing their storage
    // shape (observed rank1 [7] in public ACL). Both labels denote the same
    // compact unblocked storage; NC1HWC0 still requires its blocked layout.
    const bool plainWeight=(wf==ge::FORMAT_ND)||(format==ge::FORMAT_NCHW && wf==ge::FORMAT_NCHW);
    if(format==ge::FORMAT_NC1HWC0 ? wf!=ge::FORMAT_NC1HWC0 : !plainWeight)return ge::GRAPH_FAILED;
    for(size_t i=1;i<3;++i)if(context->GetInputDesc(i)->GetDataType()!=dtype)return ge::GRAPH_FAILED;
    for(size_t i=0;i<2;++i)if(context->GetOutputDesc(i)->GetDataType()!=dtype || ge::GetPrimaryFormat(context->GetOutputDesc(i)->GetStorageFormat())!=format)return ge::GRAPH_FAILED;
    uint64_t width=0,key=0;
    if(dtype==ge::DT_FLOAT){width=4;key=GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_GENERAL);}
#if PRELU_ENABLE_FP16
    else if(dtype==ge::DT_FLOAT16){width=2;key=GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16_GENERAL);}
#endif
#if PRELU_ENABLE_BF16
    else if(dtype==ge::DT_BF16){width=2;key=GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16_GENERAL);}
#endif
    else return ge::GRAPH_FAILED;
    const auto g=StorageDims(context->GetInputShape(0)),x=StorageDims(context->GetInputShape(1)),w=StorageDims(context->GetInputShape(2));
    JadeYaraShape::Dims out,mapped;
    if(!JadeYaraShape::Normalize(g,x,w,format==ge::FORMAT_NC1HWC0,true,out,mapped)){
        OP_LOGE(context,"unsupported or incompatible physical broadcast descriptors");return ge::GRAPH_FAILED;
    }
    if(StorageDims(context->GetOutputShape(0))!=out || StorageDims(context->GetOutputShape(1))!=out)return ge::GRAPH_FAILED;
    uint64_t total=0,sourceN[3]={};
    if(!JadeYaraShape::Product(out,total)||!JadeYaraShape::Product(g,sourceN[0])||!JadeYaraShape::Product(x,sourceN[1])||!JadeYaraShape::Product(mapped,sourceN[2]))return ge::GRAPH_FAILED;
    const uint64_t maxElements=static_cast<uint64_t>(std::numeric_limits<int64_t>::max())/width;
    if(total>maxElements || sourceN[0]>maxElements || sourceN[1]>maxElements || sourceN[2]>maxElements)return ge::GRAPH_FAILED;
    if(format==ge::FORMAT_ND && total>0 && g==x && g.size()>=2 && w.size()==1 && (w[0]==1||w[0]==g[1])) {
        return PReluGradUpdateLegacyTilingFunc(context);
    }
    uint64_t ubSize=0;int64_t cores=0;
    if(GetPlatformInfo(context,ubSize,cores)!=ge::GRAPH_SUCCESS||SetWorkspace(context)!=ge::GRAPH_SUCCESS)return ge::GRAPH_FAILED;
    // Nine FP32 arrays: g,x,w,dx,up,pos,compact/ramp/lanes; mask is one byte/lane.
    // Low precision adds one reusable raw input and two raw outputs. Two 256B
    // phase pads separate all critical binary arithmetic source pairs modulo512.
    constexpr uint64_t tile=PRELU_GENERAL_TILE;
    const uint64_t ubBytes=37*tile+512+(width==2?6*tile:0);
    if(ubBytes>ubSize*UB_USAGE_NUMERATOR/UB_USAGE_DENOMINATOR)return ge::GRAPH_FAILED;
    auto* t=context->GetTilingData<PReluGradUpdateGeneralTilingData>();if(!t)return ge::GRAPH_FAILED;
    *t={};t->totalNum=total;t->tileLen=tile;t->rank=out.size();
    for(size_t q=0;q<3;++q)t->sourceElements[q]=sourceN[q];
    for(size_t j=0;j<out.size();++j)t->dims[j]=static_cast<uint64_t>(out[j]);
    context->SetTilingKey(key);
    // No division, modulo, suffix calculation or GM copy for an empty tensor.
    if(total==0){context->SetBlockDim(1);return ge::GRAPH_SUCCESS;}
    const uint64_t alignment=32/width,coreLimit=static_cast<uint64_t>(cores);
    const uint64_t perCore=total/coreLimit+(total%coreLimit!=0);
    t->blockSpan=(perCore/alignment+(perCore%alignment!=0))*alignment;
    const uint64_t used=total/t->blockSpan+(total%t->blockSpan!=0);
    context->SetBlockDim(static_cast<uint32_t>(used));
    t->splitPeriod=total;
    uint64_t period=1;
    for(size_t r=out.size();r>0;--r){const size_t j=r-1;t->periods[j]=period;
        if(period>=tile && period<t->splitPeriod)t->splitPeriod=period;
        period*=static_cast<uint64_t>(out[j]);
        if(t->periods[j]<tile){uint64_t m=0,k=0;
            if(!GatherMagic(t->periods[j],t->periods[j]+tile-2,m,k))return ge::GRAPH_FAILED;
            t->multiplier[j]=static_cast<int32_t>(m);t->shift[j]=static_cast<int32_t>(k);
        }
    }
    JadeYaraShape::Dims sources[3]={g,x,mapped};
    for(size_t q=0;q<3;++q){
        sources[q].insert(sources[q].begin(),out.size()-sources[q].size(),1);
        t->identity[q]=sources[q]==out?1:0;uint64_t step=1;
        for(size_t r=out.size();r>0;--r){const size_t j=r-1;t->suffix[q][j]=step;
            t->strides[q][j]=sources[q][j]==1?0:step;step*=static_cast<uint64_t>(sources[q][j]);}
        for(size_t j=0;j+1<out.size();++j)if(t->periods[j]<tile){
            const int64_t coefficient=static_cast<int64_t>(t->strides[q][j])-static_cast<int64_t>(t->dims[j+1]*t->strides[q][j+1]);
            t->coefficient[q][j]=static_cast<int32_t>(coefficient);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForPReluGradUpdate([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct PReluGradUpdateCompileInfo {};

IMPL_OP_OPTILING(PReluGradUpdate)
    .Tiling(PReluGradUpdateTilingFunc)
    .TilingParse<PReluGradUpdateCompileInfo>(TilingParseForPReluGradUpdate);

} // namespace optiling
