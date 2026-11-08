#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "layer_norm_v3_tiling_data.h"

namespace LayerNormV3UT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "LayerNormV3";

struct LayerNormV3TestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> gammaShape;
    ge::DataType gammaDtype;
    ge::Format gammaFormat;
    std::initializer_list<int64_t> betaShape;
    ge::DataType betaDtype;
    ge::Format betaFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::initializer_list<int64_t> meanShape;
    ge::DataType meanDtype;
    ge::Format meanFormat;
    std::initializer_list<int64_t> rstdShape;
    ge::DataType rstdDtype;
    ge::Format rstdFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// 样例 x[4,6,8]、begin_norm_axis=1：row=4,col=48,gamma=48。
// core>=4 时 usedCores=4, rowsPerCore=1, rowRemain=0。
// col=48 < 64 → tileLength=48, tileAlign=64。epsilon 1e-5 的 float 位型是 925353388。
// workspace 固定 16MB。key：fp16=0，fp32=1，bf16=2，fp16+fp32 gamma=3，bf16+fp32 gamma=4。
static const std::string kSampleTiling = "4 48 48 1 0 4 48 64 925353388 ";
static LayerNormV3TestParam testCases[] = {
    {"layer_norm_v3_f32", {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, kSampleTiling, {16777216}, 64, 262144, 4096},
    {"layer_norm_v3_f16", {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 1, 1}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, kSampleTiling, {16777216}, 64, 262144, 4096},
    {"layer_norm_v3_bf16", {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, {6, 8}, ge::DT_BF16, ge::FORMAT_ND, {6, 8}, ge::DT_BF16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, {4, 1, 1}, ge::DT_BF16, ge::FORMAT_ND, {4, 1, 1}, ge::DT_BF16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 2UL, kSampleTiling, {16777216}, 64, 262144, 4096},
    {"layer_norm_v3_f16_f32g", {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 3UL, kSampleTiling, {16777216}, 64, 262144, 4096},
    {"layer_norm_v3_bf16_f32g", {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, {6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, {4, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 1, 1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 4UL, kSampleTiling, {16777216}, 64, 262144, 4096},
};

class LayerNormV3TilingTest : public testing::TestWithParam<LayerNormV3TestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "LayerNormV3TilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "LayerNormV3TilingTest TearDown." << std::endl;
    }
};

struct LayerNormV3CompileInfo {} compileInfo;

static void TestOneParamCase(const LayerNormV3TestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape gammaShape = {param.gammaShape, param.gammaShape};
    gert::StorageShape betaShape = {param.betaShape, param.betaShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape meanShape = {param.meanShape, param.meanShape};
    gert::StorageShape rstdShape = {param.rstdShape, param.rstdShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
        {gammaShape, param.gammaDtype, param.gammaFormat},
        {betaShape, param.betaDtype, param.betaFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat},
        {meanShape, param.meanDtype, param.meanFormat},
        {rstdShape, param.rstdDtype, param.rstdFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("begin_norm_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("begin_params_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(1)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(1e-05f)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(LayerNormV3TilingTest, tiling_test)
{
    const LayerNormV3TestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    LayerNormV3TilingTests,
    LayerNormV3TilingTest,
    testing::ValuesIn(testCases));

}
