#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "sync_batch_norm_gather_stats_tiling_data.h"

namespace SyncBatchNormGatherStatsUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "SyncBatchNormGatherStats";

struct SyncBatchNormGatherStatsTestParam {
    std::string caseName;
    std::initializer_list<int64_t> total_sumShape;
    ge::DataType total_sumDtype;
    ge::Format total_sumFormat;
    std::initializer_list<int64_t> total_square_sumShape;
    ge::DataType total_square_sumDtype;
    ge::Format total_square_sumFormat;
    std::initializer_list<int64_t> sample_countShape;
    ge::DataType sample_countDtype;
    ge::Format sample_countFormat;
    std::initializer_list<int64_t> meanShape;
    ge::DataType meanDtype;
    ge::Format meanFormat;
    std::initializer_list<int64_t> varianceShape;
    ge::DataType varianceDtype;
    ge::Format varianceFormat;
    std::initializer_list<int64_t> batch_meanShape;
    ge::DataType batch_meanDtype;
    ge::Format batch_meanFormat;
    std::initializer_list<int64_t> batch_invstdShape;
    ge::DataType batch_invstdDtype;
    ge::Format batch_invstdFormat;
    std::initializer_list<int64_t> new_meanShape;
    ge::DataType new_meanDtype;
    ge::Format new_meanFormat;
    std::initializer_list<int64_t> new_varianceShape;
    ge::DataType new_varianceDtype;
    ge::Format new_varianceFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/sync_batch_norm_gather_stats_tiling_key.h 和 op_host/sync_batch_norm_gather_stats_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/sync_batch_norm_gather_stats_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/sync_batch_norm_gather_stats_tiling.cpp 中 GetWorkspaceSize 的逻辑
static SyncBatchNormGatherStatsTestParam testCases[] = {
    // totalNum blockFactor ubFactor worldSize countNum countIsInt64 momentumBits epsBits
    // C=6 时合并到单核，ubFactor 对齐到 8
    {"sync_batch_norm_gather_stats_0", {1, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 6}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_INT32, ge::FORMAT_ND, {6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::DT_FLOAT, ge::FORMAT_ND, {6}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "6 6 8 1 1 0 1036831949 925353388 ", {0}, 64, 262144, 4096},
};

class SyncBatchNormGatherStatsTilingTest : public testing::TestWithParam<SyncBatchNormGatherStatsTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "SyncBatchNormGatherStatsTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "SyncBatchNormGatherStatsTilingTest TearDown." << std::endl;
    }
};

struct SyncBatchNormGatherStatsCompileInfo {} compileInfo;

static void TestOneParamCase(const SyncBatchNormGatherStatsTestParam &param)
{
    gert::StorageShape total_sumShape = {param.total_sumShape, param.total_sumShape};
    gert::StorageShape total_square_sumShape = {param.total_square_sumShape, param.total_square_sumShape};
    gert::StorageShape sample_countShape = {param.sample_countShape, param.sample_countShape};
    gert::StorageShape meanShape = {param.meanShape, param.meanShape};
    gert::StorageShape varianceShape = {param.varianceShape, param.varianceShape};
    gert::StorageShape batch_meanShape = {param.batch_meanShape, param.batch_meanShape};
    gert::StorageShape batch_invstdShape = {param.batch_invstdShape, param.batch_invstdShape};
    gert::StorageShape new_meanShape = {param.new_meanShape, param.new_meanShape};
    gert::StorageShape new_varianceShape = {param.new_varianceShape, param.new_varianceShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{total_sumShape, param.total_sumDtype, param.total_sumFormat},
        {total_square_sumShape, param.total_square_sumDtype, param.total_square_sumFormat},
        {sample_countShape, param.sample_countDtype, param.sample_countFormat},
        {meanShape, param.meanDtype, param.meanFormat},
        {varianceShape, param.varianceDtype, param.varianceFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{batch_meanShape, param.batch_meanDtype, param.batch_meanFormat},
        {batch_invstdShape, param.batch_invstdDtype, param.batch_invstdFormat},
        {new_meanShape, param.new_meanDtype, param.new_meanFormat},
        {new_varianceShape, param.new_varianceDtype, param.new_varianceFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("momentum", Ops::Math::AnyValue::CreateFrom<float>(0.1f)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("eps", Ops::Math::AnyValue::CreateFrom<float>(1e-05f)));
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

TEST_P(SyncBatchNormGatherStatsTilingTest, tiling_test)
{
    const SyncBatchNormGatherStatsTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    SyncBatchNormGatherStatsTilingTests,
    SyncBatchNormGatherStatsTilingTest,
    testing::ValuesIn(testCases));

}
