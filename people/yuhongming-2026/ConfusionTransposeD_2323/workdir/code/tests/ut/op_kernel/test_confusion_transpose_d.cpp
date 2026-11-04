/*!
 * \file test_confusion_transpose_d.cpp
 * \brief ConfusionTransposeD 算子 kernel UT 测试
 */

#include "confusion_transpose_d_tiling.h"
#include "../../../op_kernel/confusion_transpose_d.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

class ConfusionTransposeDKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ConfusionTransposeDKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ConfusionTransposeDKernelTest TearDown" << endl;
    }
};

TEST_F(ConfusionTransposeDKernelTest, test_kernel_run)
{
    constexpr size_t size = 120;
    constexpr size_t tilingDataSize = sizeof(ConfusionTransposeDTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 120 * 4;
    constexpr size_t yByteSize = 120 * 4;
    std::vector<uint8_t> yHost(yByteSize);

    ASSERT_NE(std::getenv("SAMPLE_DIR"), nullptr);
    const std::string root = std::getenv("SAMPLE_DIR");
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    {
        std::ifstream f(root + "/input/x.bin", std::ios::binary);
        ASSERT_TRUE(f.good());
        f.read(reinterpret_cast<char*>(x), 480);
        ASSERT_EQ(f.gcount(), 480);
    }

    ConfusionTransposeDTilingData* tilingData = reinterpret_cast<ConfusionTransposeDTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    tilingData->transposeFirst = 1;
    tilingData->rank = 4;
    tilingData->rankIn = 4;
    tilingData->dimIn0 = 2; tilingData->dimIn1 = 3; tilingData->dimIn2 = 4; tilingData->dimIn3 = 5;
    tilingData->dimIn4 = 1; tilingData->dimIn5 = 1; tilingData->dimIn6 = 1; tilingData->dimIn7 = 1;
    tilingData->dimShape0 = 2; tilingData->dimShape1 = 4; tilingData->dimShape2 = 3; tilingData->dimShape3 = 5;
    tilingData->dimShape4 = 1; tilingData->dimShape5 = 1; tilingData->dimShape6 = 1; tilingData->dimShape7 = 1;
    tilingData->perm0 = 0; tilingData->perm1 = 2; tilingData->perm2 = 1; tilingData->perm3 = 3;
    tilingData->perm4 = 4; tilingData->perm5 = 5; tilingData->perm6 = 6; tilingData->perm7 = 7;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((confusion_transpose_d<1>), numBlocks, x, y, workspace, tiling);

    memcpy(yHost.data(), y, yByteSize);
    {
        std::ofstream _ofs((root + "/output/y.bin"), std::ios::binary);
        _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize);
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
