#include <cstring>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "p_relu_grad_update_tiling_data.h"
#include "p_relu_grad_update_general_tiling.h"
#include "p_relu_grad_update_tiling_key.h"

namespace {
using Dims = std::initializer_list<int64_t>;
struct PReluGradUpdateCompileInfo {} info;
bool Tile(Dims g, Dims x, Dims w, Dims out, ge::DataType dt,
          ge::Format fmt, TilingInfo& result) {
    gert::StorageShape gs{g,g}, xs{x,x}, ws{w,w}, os{out,out};
    const auto wf = fmt == ge::FORMAT_NC1HWC0 ? fmt : ge::FORMAT_ND;
    std::vector<gert::TilingContextPara::TensorDescription> inputs{
        {gs,dt,fmt},{xs,dt,fmt},{ws,dt,wf}};
    std::vector<gert::TilingContextPara::TensorDescription> outputs{{os,dt,fmt},{os,dt,fmt}};
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    gert::TilingContextPara context("PReluGradUpdate", inputs, outputs, attrs,
                                   &info, 64, 262144, 4096);
    return ExecuteTiling(context, result);
}
class LegacyTiling : public testing::TestWithParam<ge::DataType> {};
uint64_t Key(ge::DataType dt) {
    if (dt==ge::DT_FLOAT) return GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT);
    if (dt==ge::DT_FLOAT16) return GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT16);
    return GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_BF16);
}
TEST_P(LegacyTiling, ChannelSampleCoversEveryElement) {
    TilingInfo result; auto dt=GetParam();
    ASSERT_TRUE(Tile({4,6,8},{4,6,8},{6},{4,6,8},dt,ge::FORMAT_ND,result));
    ASSERT_EQ(result.tilingDataSize,sizeof(PReluGradUpdateTilingData));
    PReluGradUpdateTilingData t{}; std::memcpy(&t,result.tilingData.get(),sizeof(t));
    EXPECT_EQ(result.tilingKey,Key(dt)); EXPECT_EQ(t.totalNum,192);
    EXPECT_EQ(t.channelNum,6); EXPECT_EQ(t.innerSize,8); EXPECT_EQ(t.sampleSize,48);
    EXPECT_EQ(t.weightsPerChannel,1); EXPECT_EQ(t.tileLen,8);
    EXPECT_EQ(t.blockFactor,dt==ge::DT_FLOAT?8:16);
    EXPECT_EQ(result.blockNum,dt==ge::DT_FLOAT?24u:12u);
    EXPECT_EQ(t.coreNum,result.blockNum); EXPECT_EQ(t.rowsPerTile,0);
    EXPECT_EQ(t.splitBatchGroups,0); EXPECT_EQ(result.workspaceSizes,std::vector<int64_t>{0});
    EXPECT_EQ(t.blockFactor*t.coreNum,t.totalNum);
}
TEST_P(LegacyTiling, ScalarWeightUsesFourCores) {
    TilingInfo result;
    ASSERT_TRUE(Tile({128,1,1},{128,1,1},{1},{128,1,1},GetParam(),ge::FORMAT_ND,result));
    ASSERT_EQ(result.tilingDataSize,sizeof(PReluGradUpdateTilingData));
    PReluGradUpdateTilingData t{}; std::memcpy(&t,result.tilingData.get(),sizeof(t));
    EXPECT_EQ(result.tilingKey,Key(GetParam())); EXPECT_EQ(result.blockNum,4u);
    EXPECT_EQ(t.blockFactor,32); EXPECT_EQ(t.tileLen,32); EXPECT_EQ(t.weightsPerChannel,0);
}
INSTANTIATE_TEST_SUITE_P(Dtypes,LegacyTiling,
                         testing::Values(ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_BF16));
TEST(GeneralTiling, BroadcastUsesItsOwnMetadata) {
    TilingInfo result;
    ASSERT_TRUE(Tile({1,7,3},{2,7,1},{7},{2,7,3},ge::DT_FLOAT,ge::FORMAT_ND,result));
    ASSERT_EQ(result.tilingDataSize,sizeof(PReluGradUpdateGeneralTilingData));
    PReluGradUpdateGeneralTilingData t{};std::memcpy(&t,result.tilingData.get(),sizeof(t));
    EXPECT_EQ(result.tilingKey,GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_GENERAL));
    EXPECT_EQ(t.totalNum,42u); EXPECT_EQ(t.rank,3u); EXPECT_EQ(t.sourceElements[0],21u);
    EXPECT_EQ(t.sourceElements[1],14u); EXPECT_EQ(t.sourceElements[2],7u);
    EXPECT_EQ(t.tileLen,2048u); EXPECT_GE(result.blockNum*t.blockSpan,t.totalNum);
    EXPECT_LT((result.blockNum-1)*t.blockSpan,t.totalNum);
}
TEST(GeneralTiling, EmptyOutputIsOneBlockNoOp) {
    TilingInfo result;
    ASSERT_TRUE(Tile({0,3,4},{0,3,4},{3},{0,3,4},ge::DT_FLOAT,ge::FORMAT_ND,result));
    ASSERT_EQ(result.tilingDataSize,sizeof(PReluGradUpdateGeneralTilingData));
    PReluGradUpdateGeneralTilingData t{};std::memcpy(&t,result.tilingData.get(),sizeof(t));
    EXPECT_EQ(t.totalNum,0u); EXPECT_EQ(result.blockNum,1u); EXPECT_EQ(t.blockSpan,0u);
    EXPECT_EQ(result.tilingKey,GET_TPL_TILING_KEY(PRELUGRADUPDATE_TPL_FLOAT_GENERAL));
}
TEST(GeneralTiling, RejectsWeightExpansionAndWrongOutput) {
    TilingInfo result;
    EXPECT_FALSE(Tile({1,1,2},{1,1,2},{3},{1,3,2},ge::DT_FLOAT,ge::FORMAT_ND,result));
    EXPECT_FALSE(Tile({1,7,3},{2,7,1},{7},{1,7,3},ge::DT_FLOAT,ge::FORMAT_ND,result));
}
}
