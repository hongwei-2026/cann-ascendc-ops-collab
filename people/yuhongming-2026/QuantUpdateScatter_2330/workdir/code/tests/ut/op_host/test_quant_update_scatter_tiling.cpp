#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "quant_update_scatter_tiling_data.h"

namespace QuantUpdateScatterUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "QuantUpdateScatter";

struct QuantUpdateScatterTestParam {
    std::string caseName;
    std::initializer_list<int64_t> varShape;
    ge::DataType varDtype;
    ge::Format varFormat;
    std::initializer_list<int64_t> indicesShape;
    ge::DataType indicesDtype;
    ge::Format indicesFormat;
    std::initializer_list<int64_t> updatesShape;
    ge::DataType updatesDtype;
    ge::Format updatesFormat;
    std::initializer_list<int64_t> quant_scalesShape;
    ge::DataType quant_scalesDtype;
    ge::Format quant_scalesFormat;
    std::initializer_list<int64_t> quant_zero_pointsShape;
    ge::DataType quant_zero_pointsDtype;
    ge::Format quant_zero_pointsFormat;
    std::initializer_list<int64_t> var_outShape;
    ge::DataType var_outDtype;
    ge::Format var_outFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// Q001 expectations include copy-only cores and 32B-aligned UB tile lengths.
static QuantUpdateScatterTestParam testCases[] = {
    // totalNum blockFactor ubFactor coreNum batchSize midSize updateAxis varAxis lastDim
    // indexElements hasZeroPoints reciprocalScale indicesIsInt64
    {"quant_update_scatter_0", {1, 2, 32}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::DT_INT32, ge::FORMAT_ND, {1, 1, 32}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 1, 32}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 1, 32}, ge::DT_INT32, ge::FORMAT_ND, {1, 2, 32}, ge::DT_INT8, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "64 1 32 1 1 1 1 2 32 1 1 0 0 ", {0}, 64, 262144, 4096},
    {"q001_copy_cores_batch1", {1, 2048, 64}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::DT_INT32, ge::FORMAT_ND, {1, 1, 64}, ge::DT_FLOAT16, ge::FORMAT_ND, {64}, ge::DT_FLOAT, ge::FORMAT_ND, {64}, ge::DT_INT32, ge::FORMAT_ND, {1, 2048, 64}, ge::DT_INT8, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "131072 1 64 4 1 1 1 2048 64 1 1 0 0 ", {0}, 64, 262144, 4096},
    {"q001_bf16_tail33", {3, 2, 5, 33}, ge::DT_INT8, ge::FORMAT_ND, {3}, ge::DT_INT64, ge::FORMAT_ND, {3, 2, 2, 33}, ge::DT_BF16, ge::FORMAT_ND, {33}, ge::DT_BF16, ge::FORMAT_ND, {33}, ge::DT_BF16, ge::FORMAT_ND, {3, 2, 5, 33}, ge::DT_INT8, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "990 1 64 3 3 2 2 5 33 3 1 0 1 ", {0}, 64, 262144, 4096},

    {"r001_fp16_indices64", {3, 2, 5, 33}, ge::DT_INT8, ge::FORMAT_ND, {3}, ge::DT_INT64, ge::FORMAT_ND, {3, 2, 2, 33}, ge::DT_FLOAT16, ge::FORMAT_ND, {33}, ge::DT_FLOAT, ge::FORMAT_ND, {33}, ge::DT_INT32, ge::FORMAT_ND, {3, 2, 5, 33}, ge::DT_INT8, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "990 1 64 3 3 2 2 5 33 3 1 0 1 ", {0}, 64, 262144, 4096},
    {"r001_bf16_indices32", {3, 2, 5, 33}, ge::DT_INT8, ge::FORMAT_ND, {3}, ge::DT_INT32, ge::FORMAT_ND, {3, 2, 2, 33}, ge::DT_BF16, ge::FORMAT_ND, {33}, ge::DT_BF16, ge::FORMAT_ND, {33}, ge::DT_BF16, ge::FORMAT_ND, {3, 2, 5, 33}, ge::DT_INT8, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "990 1 64 3 3 2 2 5 33 3 1 0 0 ", {0}, 64, 262144, 4096},

    {"q002_wide_ub_two_slots", {1, 4, 8193}, ge::DT_INT8, ge::FORMAT_ND, {1}, ge::DT_INT64, ge::FORMAT_ND, {1, 2, 8193}, ge::DT_FLOAT16, ge::FORMAT_ND, {8193}, ge::DT_FLOAT, ge::FORMAT_ND, {8193}, ge::DT_INT32, ge::FORMAT_ND, {1, 4, 8193}, ge::DT_INT8, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "32772 1 7904 2 1 1 2 4 8193 1 1 0 1 ", {0}, 64, 262144, 4096},

};

class QuantUpdateScatterTilingTest : public testing::TestWithParam<QuantUpdateScatterTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "QuantUpdateScatterTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "QuantUpdateScatterTilingTest TearDown." << std::endl;
    }
};

struct QuantUpdateScatterCompileInfo {} compileInfo;

static void TestOneParamCase(const QuantUpdateScatterTestParam &param)
{
    gert::StorageShape varShape = {param.varShape, param.varShape};
    gert::StorageShape indicesShape = {param.indicesShape, param.indicesShape};
    gert::StorageShape updatesShape = {param.updatesShape, param.updatesShape};
    gert::StorageShape quant_scalesShape = {param.quant_scalesShape, param.quant_scalesShape};
    gert::StorageShape quant_zero_pointsShape = {param.quant_zero_pointsShape, param.quant_zero_pointsShape};
    gert::StorageShape var_outShape = {param.var_outShape, param.var_outShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{varShape, param.varDtype, param.varFormat},
        {indicesShape, param.indicesDtype, param.indicesFormat},
        {updatesShape, param.updatesDtype, param.updatesFormat},
        {quant_scalesShape, param.quant_scalesDtype, param.quant_scalesFormat},
        {quant_zero_pointsShape, param.quant_zero_pointsDtype, param.quant_zero_pointsFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{var_outShape, param.var_outDtype, param.var_outFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("reduce", Ops::Math::AnyValue::CreateFrom<std::string>("update")));
    attrs_.push_back(gert::TilingContextPara::OpAttr("axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-2)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("quant_axis", Ops::Math::AnyValue::CreateFrom<int64_t>(-1)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("reciprocal_scale", Ops::Math::AnyValue::CreateFrom<bool>(false)));
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

TEST_P(QuantUpdateScatterTilingTest, tiling_test)
{
    const QuantUpdateScatterTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    QuantUpdateScatterTilingTests,
    QuantUpdateScatterTilingTest,
    testing::ValuesIn(testCases));

}
