#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "ctc_loss_v2_tiling_data.h"

namespace CtcLossV2UT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "CtcLossV2";

struct CtcLossV2TestParam {
    std::string caseName;
    std::initializer_list<int64_t> log_probsShape;
    ge::DataType log_probsDtype;
    ge::Format log_probsFormat;
    std::initializer_list<int64_t> targetsShape;
    ge::DataType targetsDtype;
    ge::Format targetsFormat;
    std::initializer_list<int64_t> input_lengthsShape;
    ge::DataType input_lengthsDtype;
    ge::Format input_lengthsFormat;
    std::initializer_list<int64_t> target_lengthsShape;
    ge::DataType target_lengthsDtype;
    ge::Format target_lengthsFormat;
    std::initializer_list<int64_t> neg_log_likelihoodShape;
    ge::DataType neg_log_likelihoodDtype;
    ge::Format neg_log_likelihoodFormat;
    std::initializer_list<int64_t> log_alphaShape;
    ge::DataType log_alphaDtype;
    ge::Format log_alphaFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// T=10,N=1,C=5,S=8,blank=0,zi=0,int32,usedCores=1
static CtcLossV2TestParam testCases[] = {
    {"ctc_loss_v2_0",
     {10, 1, 5},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1, 8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1, 10, 17},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     1UL,
     "10 1 5 8 0 0 0 1 ",
     {16UL * 1024UL * 1024UL},
     64,
     262144,
     4096},
};

class CtcLossV2TilingTest : public testing::TestWithParam<CtcLossV2TestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CtcLossV2TilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CtcLossV2TilingTest TearDown." << std::endl;
    }
};

struct CtcLossV2CompileInfo {} compileInfo;

static void TestOneParamCase(const CtcLossV2TestParam& param)
{
    gert::StorageShape log_probsShape = {param.log_probsShape, param.log_probsShape};
    gert::StorageShape targetsShape = {param.targetsShape, param.targetsShape};
    gert::StorageShape input_lengthsShape = {param.input_lengthsShape, param.input_lengthsShape};
    gert::StorageShape target_lengthsShape = {param.target_lengthsShape, param.target_lengthsShape};
    gert::StorageShape neg_log_likelihoodShape = {param.neg_log_likelihoodShape, param.neg_log_likelihoodShape};
    gert::StorageShape log_alphaShape = {param.log_alphaShape, param.log_alphaShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{log_probsShape, param.log_probsDtype, param.log_probsFormat},
         {targetsShape, param.targetsDtype, param.targetsFormat},
         {input_lengthsShape, param.input_lengthsDtype, param.input_lengthsFormat},
         {target_lengthsShape, param.target_lengthsDtype, param.target_lengthsFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{neg_log_likelihoodShape, param.neg_log_likelihoodDtype, param.neg_log_likelihoodFormat},
         {log_alphaShape, param.log_alphaDtype, param.log_alphaFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("blank", Ops::Math::AnyValue::CreateFrom<int64_t>(0)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("reduction", Ops::Math::AnyValue::CreateFrom<std::string>("mean")));
    attrs_.push_back(gert::TilingContextPara::OpAttr("zero_infinity", Ops::Math::AnyValue::CreateFrom<bool>(false)));
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

TEST_P(CtcLossV2TilingTest, tiling_test)
{
    const CtcLossV2TestParam& param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    CtcLossV2TilingTests,
    CtcLossV2TilingTest,
    testing::ValuesIn(testCases));

}
