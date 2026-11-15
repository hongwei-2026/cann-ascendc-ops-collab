/*!
 * \file test_quant_update_scatter.cpp
 * \brief QuantUpdateScatter 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "quant_update_scatter_tiling.h"
#include "../../../op_kernel/quant_update_scatter.cpp"

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

class QuantUpdateScatterKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "QuantUpdateScatterKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "QuantUpdateScatterKernelTest TearDown" << endl;
    }
};

TEST_F(QuantUpdateScatterKernelTest, test_kernel_run)
{
    constexpr size_t size = 64;
    constexpr size_t tilingDataSize = sizeof(QuantUpdateScatterTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t varByteSize = 64 * 1;
    constexpr size_t indicesByteSize = 1 * 4;
    constexpr size_t updatesByteSize = 32 * 2;
    constexpr size_t quant_scalesByteSize = 32 * 4;
    constexpr size_t quant_zero_pointsByteSize = 32 * 4;
    constexpr size_t var_outByteSize = 64 * 1;
    std::vector<uint8_t> varHost(varByteSize);
    std::vector<uint8_t> indicesHost(indicesByteSize);
    std::vector<uint8_t> updatesHost(updatesByteSize);
    std::vector<uint8_t> quant_scalesHost(quant_scalesByteSize);
    std::vector<uint8_t> quant_zero_pointsHost(quant_zero_pointsByteSize);
    std::vector<uint8_t> var_outHost(var_outByteSize);
    
    
    ASSERT_NE(std::getenv("SAMPLE_DIR"), nullptr);
    const std::string root = std::getenv("SAMPLE_DIR");
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(varByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* updates = (uint8_t*)AscendC::GmAlloc(updatesByteSize);
    uint8_t* quant_scales = (uint8_t*)AscendC::GmAlloc(quant_scalesByteSize);
    uint8_t* quant_zero_points = (uint8_t*)AscendC::GmAlloc(quant_zero_pointsByteSize);
    uint8_t* var_out = (uint8_t*)AscendC::GmAlloc(var_outByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    { std::ifstream f(root + "/input/var.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(var), 64); ASSERT_EQ(f.gcount(), 64); }
    { std::ifstream f(root + "/input/indices.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(indices), 4); ASSERT_EQ(f.gcount(), 4); }
    { std::ifstream f(root + "/input/updates.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(updates), 64); ASSERT_EQ(f.gcount(), 64); }
    { std::ifstream f(root + "/input/quant_scales.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(quant_scales), 128); ASSERT_EQ(f.gcount(), 128); }
    { std::ifstream f(root + "/input/quant_zero_points.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(quant_zero_points), 128); ASSERT_EQ(f.gcount(), 128); }
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/quant_update_scatter_tiling_data.h 中的字段定义
    //         参考 op_host/quant_update_scatter_tiling.cpp 中的 tiling 计算逻辑
    QuantUpdateScatterTilingData* tilingData = reinterpret_cast<QuantUpdateScatterTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/quant_update_scatter_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((quant_update_scatter<1>), numBlocks, var, indices, updates, quant_scales, quant_zero_points, var_out, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(var_outHost.data(), var_out, var_outByteSize);
    { std::ofstream _ofs((root + "/output/var_out.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(var_outHost.data()), var_outByteSize); }
    
    AscendC::GmFree(var);
    AscendC::GmFree(indices);
    AscendC::GmFree(updates);
    AscendC::GmFree(quant_scales);
    AscendC::GmFree(quant_zero_points);
    AscendC::GmFree(var_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
