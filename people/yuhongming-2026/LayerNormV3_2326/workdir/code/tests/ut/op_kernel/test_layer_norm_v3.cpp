/*!
 * \file test_layer_norm_v3.cpp
 * \brief LayerNormV3 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "layer_norm_v3_tiling.h"
#include "../../../op_kernel/layer_norm_v3.cpp"

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

class LayerNormV3KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "LayerNormV3KernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "LayerNormV3KernelTest TearDown" << endl;
    }
};

TEST_F(LayerNormV3KernelTest, test_kernel_run)
{
    constexpr size_t tilingDataSize = sizeof(LayerNormV3TilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t xByteSize = 192 * 4;
    constexpr size_t gammaByteSize = 48 * 4;
    constexpr size_t betaByteSize = 48 * 4;
    constexpr size_t yByteSize = 192 * 4;
    constexpr size_t meanByteSize = 4 * 4;
    constexpr size_t rstdByteSize = 4 * 4;
    std::vector<uint8_t> xHost(xByteSize);
    std::vector<uint8_t> gammaHost(gammaByteSize);
    std::vector<uint8_t> betaHost(betaByteSize);
    std::vector<uint8_t> yHost(yByteSize);
    std::vector<uint8_t> meanHost(meanByteSize);
    std::vector<uint8_t> rstdHost(rstdByteSize);
    
    
    ASSERT_NE(std::getenv("SAMPLE_DIR"), nullptr);
    const std::string root = std::getenv("SAMPLE_DIR");
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    { std::ifstream f(root + "/input/x.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(x), 768); ASSERT_EQ(f.gcount(), 768); }
    { std::ifstream f(root + "/input/gamma.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(gamma), 192); ASSERT_EQ(f.gcount(), 192); }
    { std::ifstream f(root + "/input/beta.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(beta), 192); ASSERT_EQ(f.gcount(), 192); }
    
    // 单核跑完全部 4 行。字段与 op_host/layer_norm_v3_tiling.cpp 一致。
    // epsilon 1e-5f 的位型：925353388。
    LayerNormV3TilingData* tilingData = reinterpret_cast<LayerNormV3TilingData*>(tiling);
    tilingData->rowNum = 4;
    tilingData->colNum = 48;
    tilingData->gammaNum = 48;
    tilingData->rowsPerCore = 4;
    tilingData->rowRemain = 0;
    tilingData->usedCores = 1;
    tilingData->tileLength = 48;
    tilingData->tileAlign = 64;
    tilingData->epsilonBits = 925353388;
    
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((layer_norm_v3<1>), numBlocks, x, gamma, beta, y, mean, rstd, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(yHost.data(), y, yByteSize);
    { std::ofstream _ofs((root + "/output/y.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(yHost.data()), yByteSize); }
    memcpy(meanHost.data(), mean, meanByteSize);
    { std::ofstream _ofs((root + "/output/mean.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(meanHost.data()), meanByteSize); }
    memcpy(rstdHost.data(), rstd, rstdByteSize);
    { std::ofstream _ofs((root + "/output/rstd.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(rstdHost.data()), rstdByteSize); }
    
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(y);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
