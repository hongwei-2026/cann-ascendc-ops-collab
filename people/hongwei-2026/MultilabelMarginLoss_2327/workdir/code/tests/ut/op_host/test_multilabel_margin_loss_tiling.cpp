#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "multilabel_margin_loss_tiling_data.h"

namespace MultilabelMarginLossUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "MultilabelMarginLoss";

struct MultilabelMarginLossTestParam {
    std::string caseName;
    std::string reduction;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> targetShape;
    ge::DataType targetDtype;
    ge::Format targetFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
    std::initializer_list<int64_t> is_targetShape;
    ge::DataType is_targetDtype;
    ge::Format is_targetFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

static MultilabelMarginLossTestParam testCases[] = {
    // tilingData: batchSize classNum targetLen reduction usedCores rowsPerCore tailRows classTile targetTile
    {"tiny_mean", "mean", {4, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6}, ge::DT_INT32, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "4 6 6 1 1 4 4 6 6 ", {0UL}, 64, 262144, 4096},
    {"none_no_workspace", "none", {4, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6}, ge::DT_INT32, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 6}, ge::DT_INT32, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "4 6 6 0 4 1 1 6 6 ", {0UL}, 64, 262144, 4096},
    {"tiny_sum_bf16", "sum", {4, 6}, ge::DT_BF16, ge::FORMAT_ND, {4, 6}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::DT_BF16, ge::FORMAT_ND, {4, 6}, ge::DT_INT32, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 2UL, "4 6 6 2 1 4 4 6 6 ", {0UL}, 64, 262144, 4096},
    {"multicore_mean", "mean", {100, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {100, 6}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {100, 6}, ge::DT_INT32, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "100 6 6 1 50 2 2 6 6 ", {416UL}, 64, 262144, 4096},
    {"single_core_fp16", "mean", {1, 257}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 11}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 257}, ge::DT_INT32, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "1 257 11 1 1 1 1 257 11 ", {0UL}, 64, 262144, 4096},
};

class MultilabelMarginLossTilingTest : public testing::TestWithParam<MultilabelMarginLossTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "MultilabelMarginLossTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "MultilabelMarginLossTilingTest TearDown." << std::endl;
    }
};

struct MultilabelMarginLossCompileInfo {} compileInfo;

static void TestOneParamCase(const MultilabelMarginLossTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape targetShape = {param.targetShape, param.targetShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    gert::StorageShape is_targetShape = {param.is_targetShape, param.is_targetShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
        {targetShape, param.targetDtype, param.targetFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat},
        {is_targetShape, param.is_targetDtype, param.is_targetFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("reduction", Ops::Math::AnyValue::CreateFrom<std::string>(param.reduction)));
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

TEST_P(MultilabelMarginLossTilingTest, tiling_test)
{
    const MultilabelMarginLossTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    MultilabelMarginLossTilingTests,
    MultilabelMarginLossTilingTest,
    testing::ValuesIn(testCases));

}
