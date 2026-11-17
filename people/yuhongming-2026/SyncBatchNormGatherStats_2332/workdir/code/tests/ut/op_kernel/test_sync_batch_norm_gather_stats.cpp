/*!
 * \file test_sync_batch_norm_gather_stats.cpp
 * \brief SyncBatchNormGatherStats 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "sync_batch_norm_gather_stats_tiling.h"
#include "../../../op_kernel/sync_batch_norm_gather_stats.cpp"

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

class SyncBatchNormGatherStatsKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "SyncBatchNormGatherStatsKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "SyncBatchNormGatherStatsKernelTest TearDown" << endl;
    }
};

TEST_F(SyncBatchNormGatherStatsKernelTest, test_kernel_run)
{
    constexpr size_t size = 6;
    constexpr size_t tilingDataSize = sizeof(SyncBatchNormGatherStatsTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t total_sumByteSize = 6 * 4;
    constexpr size_t total_square_sumByteSize = 6 * 4;
    constexpr size_t sample_countByteSize = 1 * 4;
    constexpr size_t meanByteSize = 6 * 4;
    constexpr size_t varianceByteSize = 6 * 4;
    constexpr size_t batch_meanByteSize = 6 * 4;
    constexpr size_t batch_invstdByteSize = 6 * 4;
    constexpr size_t new_meanByteSize = 6 * 4;
    constexpr size_t new_varianceByteSize = 6 * 4;
    std::vector<uint8_t> total_sumHost(total_sumByteSize);
    std::vector<uint8_t> total_square_sumHost(total_square_sumByteSize);
    std::vector<uint8_t> sample_countHost(sample_countByteSize);
    std::vector<uint8_t> meanHost(meanByteSize);
    std::vector<uint8_t> varianceHost(varianceByteSize);
    std::vector<uint8_t> batch_meanHost(batch_meanByteSize);
    std::vector<uint8_t> batch_invstdHost(batch_invstdByteSize);
    std::vector<uint8_t> new_meanHost(new_meanByteSize);
    std::vector<uint8_t> new_varianceHost(new_varianceByteSize);
    
    
    ASSERT_NE(std::getenv("SAMPLE_DIR"), nullptr);
    const std::string root = std::getenv("SAMPLE_DIR");
    uint8_t* total_sum = (uint8_t*)AscendC::GmAlloc(total_sumByteSize);
    uint8_t* total_square_sum = (uint8_t*)AscendC::GmAlloc(total_square_sumByteSize);
    uint8_t* sample_count = (uint8_t*)AscendC::GmAlloc(sample_countByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc(varianceByteSize);
    uint8_t* batch_mean = (uint8_t*)AscendC::GmAlloc(batch_meanByteSize);
    uint8_t* batch_invstd = (uint8_t*)AscendC::GmAlloc(batch_invstdByteSize);
    uint8_t* new_mean = (uint8_t*)AscendC::GmAlloc(new_meanByteSize);
    uint8_t* new_variance = (uint8_t*)AscendC::GmAlloc(new_varianceByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    { std::ifstream f(root + "/input/total_sum.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(total_sum), 24); ASSERT_EQ(f.gcount(), 24); }
    { std::ifstream f(root + "/input/total_square_sum.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(total_square_sum), 24); ASSERT_EQ(f.gcount(), 24); }
    { std::ifstream f(root + "/input/sample_count.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(sample_count), 4); ASSERT_EQ(f.gcount(), 4); }
    { std::ifstream f(root + "/input/mean.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(mean), 24); ASSERT_EQ(f.gcount(), 24); }
    { std::ifstream f(root + "/input/variance.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(variance), 24); ASSERT_EQ(f.gcount(), 24); }
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/sync_batch_norm_gather_stats_tiling_data.h 中的字段定义
    //         参考 op_host/sync_batch_norm_gather_stats_tiling.cpp 中的 tiling 计算逻辑
    SyncBatchNormGatherStatsTilingData* tilingData = reinterpret_cast<SyncBatchNormGatherStatsTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(SyncBatchNormGatherStatsTilingData));
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    tilingData->worldSize = 1;
    tilingData->countNum = 1;
    tilingData->countIsInt64 = 0;
    tilingData->momentumBits = 1036831949; // 0.1f
    tilingData->epsBits = 925353388;       // 1e-5f
    
    // TODO: tilingKey 应与 op_host/sync_batch_norm_gather_stats_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((sync_batch_norm_gather_stats<1>), numBlocks, total_sum, total_square_sum, sample_count, mean, variance, batch_mean, batch_invstd, new_mean, new_variance, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(batch_meanHost.data(), batch_mean, batch_meanByteSize);
    { std::ofstream _ofs((root + "/output/batch_mean.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(batch_meanHost.data()), batch_meanByteSize); }
    memcpy(batch_invstdHost.data(), batch_invstd, batch_invstdByteSize);
    { std::ofstream _ofs((root + "/output/batch_invstd.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(batch_invstdHost.data()), batch_invstdByteSize); }
    memcpy(new_meanHost.data(), new_mean, new_meanByteSize);
    { std::ofstream _ofs((root + "/output/new_mean.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(new_meanHost.data()), new_meanByteSize); }
    memcpy(new_varianceHost.data(), new_variance, new_varianceByteSize);
    { std::ofstream _ofs((root + "/output/new_variance.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(new_varianceHost.data()), new_varianceByteSize); }
    
    AscendC::GmFree(total_sum);
    AscendC::GmFree(total_square_sum);
    AscendC::GmFree(sample_count);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(batch_mean);
    AscendC::GmFree(batch_invstd);
    AscendC::GmFree(new_mean);
    AscendC::GmFree(new_variance);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
