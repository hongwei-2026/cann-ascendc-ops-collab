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

struct PadV3GradTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> paddingsShape;
    ge::DataType paddingsDtype;
    ge::Format paddingsFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
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
//   expectTilingKey:  参考 op_kernel/pad_v3_grad_tiling_key.h 和 op_host/pad_v3_grad_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/pad_v3_grad_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/pad_v3_grad_tiling.cpp 中 GetWorkspaceSize 的逻辑
static PadV3GradTestParam testCases[] = {
    {"pad_v3_grad_0", {1, 1, 8, 8}, ge::DT_FLOAT, ge::FORMAT_ND, {4, 2}, ge::DT_INT32, ge::FORMAT_ND, {1, 1, 4, 4}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
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
    gert::StorageShape paddingsShape = {param.paddingsShape, param.paddingsShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
        {paddingsShape, param.paddingsDtype, param.paddingsFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("mode", Ops::Math::AnyValue::CreateFrom<std::string>("reflect")));
    attrs_.push_back(gert::TilingContextPara::OpAttr("paddings_contiguous", Ops::Math::AnyValue::CreateFrom<bool>(true)));
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
