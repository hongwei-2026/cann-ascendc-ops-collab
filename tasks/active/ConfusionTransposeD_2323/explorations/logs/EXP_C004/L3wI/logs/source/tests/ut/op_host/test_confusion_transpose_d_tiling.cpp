#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "confusion_transpose_d_tiling_data.h"

namespace ConfusionTransposeDUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "ConfusionTransposeD";

struct ConfusionTransposeDTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
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

// float32 sample: copySpan=5, 32B/row grain=40 elements, usedCores=3.
static ConfusionTransposeDTestParam testCases[] = {
    {"confusion_transpose_d_0",
     {2, 3, 4, 5},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {2, 4, 3, 5},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     1UL,
     "120 40 120 1 4 4 2 3 4 5 1 1 1 1 2 4 3 5 1 1 1 1 0 2 1 3 4 5 6 7 5 ",
     {16UL * 1024UL * 1024UL},
     64,
     262144,
     4096},
};

class ConfusionTransposeDTilingTest : public testing::TestWithParam<ConfusionTransposeDTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ConfusionTransposeDTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ConfusionTransposeDTilingTest TearDown." << std::endl;
    }
};

struct ConfusionTransposeDCompileInfo {} compileInfo;

static void TestOneParamCase(const ConfusionTransposeDTestParam& param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("perm", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({0, 2, 1, 3})));
    attrs_.push_back(gert::TilingContextPara::OpAttr("shape", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>({2, 4, 3, 5})));
    attrs_.push_back(gert::TilingContextPara::OpAttr("transpose_first", Ops::Math::AnyValue::CreateFrom<bool>(true)));
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

TEST_P(ConfusionTransposeDTilingTest, tiling_test)
{
    const ConfusionTransposeDTestParam& param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    ConfusionTransposeDTilingTests,
    ConfusionTransposeDTilingTest,
    testing::ValuesIn(testCases));

}
