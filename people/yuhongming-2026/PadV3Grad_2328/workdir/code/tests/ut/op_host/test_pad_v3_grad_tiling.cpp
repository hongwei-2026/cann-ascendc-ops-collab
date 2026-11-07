#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "pad_v3_grad_tiling_data.h"

namespace PadV3GradUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "PadV3Grad";

static int32_t kSamplePads[] = {0, 0, 0, 0, 2, 2, 2, 2};
static int64_t kSamplePadsI64[] = {0, 0, 0, 0, 2, 2, 2, 2};
static int32_t kAsymContig[] = {0, 0, 0, 0, 1, 2, 3, 0};
static int32_t kAsymSplit[] = {0, 0, 1, 3, 0, 0, 2, 0};
static int32_t kReplicatePads[] = {1, 1};
static int32_t kNegPads[] = {0, 0, -1, 0};

struct PadV3GradTestParam {
    const char* caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    std::initializer_list<int64_t> padShape;
    ge::DataType padDtype;
    void* padData;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    const char* mode;
    bool contiguous;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    const char* expectTilingData;
    size_t expectWorkspace;
    uint64_t maxAIVNum;
};

static PadV3GradTestParam testCases[] = {
    {"reflect_f32", {1, 1, 8, 8}, ge::DT_FLOAT, {4, 2}, ge::DT_INT32, kSamplePads, {1, 1, 4, 4}, ge::DT_FLOAT, "reflect", true, ge::GRAPH_SUCCESS, 1UL, "4 2 64 16 1 0 0 4 1 1 8 8 0 0 0 0 1 1 4 4 0 0 0 0 0 0 2 2 0 0 0 0 0 0 2 2 0 0 0 0 ", 512, 64},
    {"reflect_f16", {1, 1, 8, 8}, ge::DT_FLOAT16, {4, 2}, ge::DT_INT32, kSamplePads, {1, 1, 4, 4}, ge::DT_FLOAT16, "reflect", true, ge::GRAPH_SUCCESS, 0UL, "4 2 64 16 1 0 0 4 1 1 8 8 0 0 0 0 1 1 4 4 0 0 0 0 0 0 2 2 0 0 0 0 0 0 2 2 0 0 0 0 ", 512, 64},
    {"reflect_bf16_i64", {1, 1, 8, 8}, ge::DT_BF16, {4, 2}, ge::DT_INT64, kSamplePadsI64, {1, 1, 4, 4}, ge::DT_BF16, "reflect", true, ge::GRAPH_SUCCESS, 2UL, "4 2 64 16 1 0 0 4 1 1 8 8 0 0 0 0 1 1 4 4 0 0 0 0 0 0 2 2 0 0 0 0 0 0 2 2 0 0 0 0 ", 512, 64},
    {"constant_rows", {1, 1, 8, 8}, ge::DT_FLOAT, {4, 2}, ge::DT_INT32, kSamplePads, {1, 1, 4, 4}, ge::DT_FLOAT, "constant", true, ge::GRAPH_SUCCESS, 1UL, "4 0 64 16 4 1 0 4 1 1 8 8 0 0 0 0 1 1 4 4 0 0 0 0 0 0 2 2 0 0 0 0 0 0 2 2 0 0 0 0 ", 0, 64},
    {"asym_contig", {1, 1, 7, 10}, ge::DT_FLOAT, {4, 2}, ge::DT_INT32, kAsymContig, {1, 1, 4, 7}, ge::DT_FLOAT, "reflect", true, ge::GRAPH_SUCCESS, 1UL, "4 2 70 28 1 0 0 7 1 1 7 10 0 0 0 0 1 1 4 7 0 0 0 0 0 0 1 3 0 0 0 0 0 0 2 0 0 0 0 0 ", 576, 64},
    {"asym_split", {1, 1, 7, 10}, ge::DT_FLOAT, {8}, ge::DT_INT32, kAsymSplit, {1, 1, 4, 7}, ge::DT_FLOAT, "reflect", false, ge::GRAPH_SUCCESS, 1UL, "4 2 70 28 1 0 0 7 1 1 7 10 0 0 0 0 1 1 4 7 0 0 0 0 0 0 1 3 0 0 0 0 0 0 2 0 0 0 0 0 ", 576, 64},
    {"replicate", {6}, ge::DT_FLOAT, {1, 2}, ge::DT_INT32, kReplicatePads, {4}, ge::DT_FLOAT, "replicate", true, ge::GRAPH_SUCCESS, 1UL, "1 1 6 4 1 0 0 4 6 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 ", 64, 64},
    {"neg_pad", {4, 4}, ge::DT_FLOAT, {2, 2}, ge::DT_INT32, kNegPads, {4, 4}, ge::DT_FLOAT, "reflect", true, ge::GRAPH_FAILED, 0UL, "", 0, 64},
};

class PadV3GradTilingTest : public testing::TestWithParam<PadV3GradTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "PadV3GradTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "PadV3GradTilingTest TearDown." << std::endl;
    }
};

struct PadV3GradCompileInfo {} compileInfo;

static void TestOneParamCase(const PadV3GradTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape paddingsShape = {param.padShape, param.padShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, ge::FORMAT_ND},
        {paddingsShape, param.padDtype, ge::FORMAT_ND, true, param.padData}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>(param.mode)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(param.contiguous)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        262144,
        4096);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, {param.expectWorkspace});
}

TEST_P(PadV3GradTilingTest, tiling_test)
{
    const PadV3GradTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    PadV3GradTilingTests,
    PadV3GradTilingTest,
    testing::ValuesIn(testCases));

}
