#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "p_relu_grad_update_tiling_data.h"

namespace PReluGradUpdateUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "PReluGradUpdate";

struct PReluGradUpdateTestParam {
    std::string caseName;
    std::initializer_list<int64_t> gradsShape;
    ge::DataType gradsDtype;
    ge::Format gradsFormat;
    std::initializer_list<int64_t> featuresShape;
    ge::DataType featuresDtype;
    ge::Format featuresFormat;
    std::initializer_list<int64_t> weightsShape;
    ge::DataType weightsDtype;
    ge::Format weightsFormat;
    std::initializer_list<int64_t> dxShape;
    ge::DataType dxDtype;
    ge::Format dxFormat;
    std::initializer_list<int64_t> updateShape;
    ge::DataType updateDtype;
    ge::Format updateFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static PReluGradUpdateTestParam testCases[] = {
    {"float_channel_weights", {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND,
        "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "192 1 192 192 6 8 48 1 4 ", {0}, 64, 262144, 4096},
    {"float_scalar_weight", {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND,
        {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT, ge::FORMAT_ND,
        "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "192 1 192 192 6 8 48 0 4 ", {0}, 64, 262144, 4096},
    {"float16_key", {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND,
        {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {6}, ge::DT_FLOAT16, ge::FORMAT_ND,
        {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_FLOAT16, ge::FORMAT_ND,
        "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "192 1 192 192 6 8 48 1 4 ", {0}, 64, 262144, 4096},
    {"bf16_key", {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND,
        {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, {6}, ge::DT_BF16, ge::FORMAT_ND,
        {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND, {4, 6, 8}, ge::DT_BF16, ge::FORMAT_ND,
        "Ascend910B", ge::GRAPH_SUCCESS, 2UL, "192 1 192 192 6 8 48 1 4 ", {0}, 64, 262144, 4096},
};

class PReluGradUpdateTilingTest : public testing::TestWithParam<PReluGradUpdateTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "PReluGradUpdateTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "PReluGradUpdateTilingTest TearDown." << std::endl;
    }
};

struct PReluGradUpdateCompileInfo {} compileInfo;

static void TestOneParamCase(const PReluGradUpdateTestParam &param)
{
    gert::StorageShape gradsShape = {param.gradsShape, param.gradsShape};
    gert::StorageShape featuresShape = {param.featuresShape, param.featuresShape};
    gert::StorageShape weightsShape = {param.weightsShape, param.weightsShape};
    gert::StorageShape dxShape = {param.dxShape, param.dxShape};
    gert::StorageShape updateShape = {param.updateShape, param.updateShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{gradsShape, param.gradsDtype, param.gradsFormat},
        {featuresShape, param.featuresDtype, param.featuresFormat},
        {weightsShape, param.weightsDtype, param.weightsFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{dxShape, param.dxDtype, param.dxFormat},
        {updateShape, param.updateDtype, param.updateFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

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

TEST_P(PReluGradUpdateTilingTest, tiling_test)
{
    const PReluGradUpdateTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    PReluGradUpdateTilingTests,
    PReluGradUpdateTilingTest,
    testing::ValuesIn(testCases));

}
