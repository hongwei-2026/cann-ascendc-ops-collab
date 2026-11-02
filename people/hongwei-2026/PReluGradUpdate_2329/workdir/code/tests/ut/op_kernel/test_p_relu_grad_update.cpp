/*!
 * \file test_p_relu_grad_update.cpp
 * \brief PReluGradUpdate 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "p_relu_grad_update_tiling.h"
#include "../../../op_kernel/p_relu_grad_update.cpp"

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

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class PReluGradUpdateKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "PReluGradUpdateKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "PReluGradUpdateKernelTest TearDown" << endl;
    }
};

TEST_F(PReluGradUpdateKernelTest, test_kernel_run)
{
    constexpr size_t size = 192;
    constexpr size_t tilingDataSize = sizeof(PReluGradUpdateTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t gradsByteSize = 192 * 4;
    constexpr size_t featuresByteSize = 192 * 4;
    constexpr size_t weightsByteSize = 6 * 4;
    constexpr size_t dxByteSize = 192 * 4;
    constexpr size_t updateByteSize = 192 * 4;
    std::vector<uint8_t> gradsHost(gradsByteSize);
    std::vector<uint8_t> featuresHost(featuresByteSize);
    std::vector<uint8_t> weightsHost(weightsByteSize);
    std::vector<uint8_t> dxHost(dxByteSize);
    std::vector<uint8_t> updateHost(updateByteSize);
    
    
    ASSERT_NE(std::getenv("SAMPLE_DIR"), nullptr);
    const std::string root = std::getenv("SAMPLE_DIR");
    uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(featuresByteSize);
    uint8_t* weights = (uint8_t*)AscendC::GmAlloc(weightsByteSize);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(dxByteSize);
    uint8_t* update = (uint8_t*)AscendC::GmAlloc(updateByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    { std::ifstream f(root + "/input/grads.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(grads), 768); ASSERT_EQ(f.gcount(), 768); }
    { std::ifstream f(root + "/input/features.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(features), 768); ASSERT_EQ(f.gcount(), 768); }
    { std::ifstream f(root + "/input/weights.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(weights), 24); ASSERT_EQ(f.gcount(), 24); }
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/p_relu_grad_update_tiling_data.h 中的字段定义
    //         参考 op_host/p_relu_grad_update_tiling.cpp 中的 tiling 计算逻辑
    PReluGradUpdateTilingData* tilingData = reinterpret_cast<PReluGradUpdateTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->coreNum = 1;
    tilingData->blockFactor = size;
    tilingData->tileLen = size;
    tilingData->channelNum = 6;
    tilingData->innerSize = 8;
    tilingData->sampleSize = 48;
    tilingData->weightsPerChannel = 1;
    tilingData->rowsPerTile = 4;

    // fp32 -> schMode 0
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF((p_relu_grad_update<0>), numBlocks, grads, features, weights, dx, update, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(dxHost.data(), dx, dxByteSize);
    { std::ofstream _ofs((root + "/output/dx.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(dxHost.data()), dxByteSize); }
    memcpy(updateHost.data(), update, updateByteSize);
    { std::ofstream _ofs((root + "/output/update.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(updateHost.data()), updateByteSize); }
    
    AscendC::GmFree(grads);
    AscendC::GmFree(features);
    AscendC::GmFree(weights);
    AscendC::GmFree(dx);
    AscendC::GmFree(update);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
