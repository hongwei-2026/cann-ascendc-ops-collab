/*!
 * \file test_ctc_loss_v2.cpp
 * \brief CtcLossV2 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "ctc_loss_v2_tiling.h"
#include "../../../op_kernel/ctc_loss_v2.cpp"

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

class CtcLossV2KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "CtcLossV2KernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "CtcLossV2KernelTest TearDown" << endl;
    }
};

TEST_F(CtcLossV2KernelTest, test_kernel_run)
{
    constexpr size_t size = 50;
    constexpr size_t tilingDataSize = sizeof(CtcLossV2TilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t log_probsByteSize = 50 * 4;
    constexpr size_t targetsByteSize = 8 * 4;
    constexpr size_t input_lengthsByteSize = 1 * 4;
    constexpr size_t target_lengthsByteSize = 1 * 4;
    constexpr size_t neg_log_likelihoodByteSize = 1 * 4;
    constexpr size_t log_alphaByteSize = 170 * 4;
    std::vector<uint8_t> log_probsHost(log_probsByteSize);
    std::vector<uint8_t> targetsHost(targetsByteSize);
    std::vector<uint8_t> input_lengthsHost(input_lengthsByteSize);
    std::vector<uint8_t> target_lengthsHost(target_lengthsByteSize);
    std::vector<uint8_t> neg_log_likelihoodHost(neg_log_likelihoodByteSize);
    std::vector<uint8_t> log_alphaHost(log_alphaByteSize);
    
    
    ASSERT_NE(std::getenv("SAMPLE_DIR"), nullptr);
    const std::string root = std::getenv("SAMPLE_DIR");
    uint8_t* log_probs = (uint8_t*)AscendC::GmAlloc(log_probsByteSize);
    uint8_t* targets = (uint8_t*)AscendC::GmAlloc(targetsByteSize);
    uint8_t* input_lengths = (uint8_t*)AscendC::GmAlloc(input_lengthsByteSize);
    uint8_t* target_lengths = (uint8_t*)AscendC::GmAlloc(target_lengthsByteSize);
    uint8_t* neg_log_likelihood = (uint8_t*)AscendC::GmAlloc(neg_log_likelihoodByteSize);
    uint8_t* log_alpha = (uint8_t*)AscendC::GmAlloc(log_alphaByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    { std::ifstream f(root + "/input/log_probs.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(log_probs), 200); ASSERT_EQ(f.gcount(), 200); }
    { std::ifstream f(root + "/input/targets.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(targets), 32); ASSERT_EQ(f.gcount(), 32); }
    { std::ifstream f(root + "/input/input_lengths.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(input_lengths), 4); ASSERT_EQ(f.gcount(), 4); }
    { std::ifstream f(root + "/input/target_lengths.bin", std::ios::binary); ASSERT_TRUE(f.good()); f.read(reinterpret_cast<char*>(target_lengths), 4); ASSERT_EQ(f.gcount(), 4); }
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/ctc_loss_v2_tiling_data.h 中的字段定义
    //         参考 op_host/ctc_loss_v2_tiling.cpp 中的 tiling 计算逻辑
    CtcLossV2TilingData* tilingData = reinterpret_cast<CtcLossV2TilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/ctc_loss_v2_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((ctc_loss_v2<1>), numBlocks, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(neg_log_likelihoodHost.data(), neg_log_likelihood, neg_log_likelihoodByteSize);
    { std::ofstream _ofs((root + "/output/neg_log_likelihood.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(neg_log_likelihoodHost.data()), neg_log_likelihoodByteSize); }
    memcpy(log_alphaHost.data(), log_alpha, log_alphaByteSize);
    { std::ofstream _ofs((root + "/output/log_alpha.bin"), std::ios::binary); _ofs.write(reinterpret_cast<const char*>(log_alphaHost.data()), log_alphaByteSize); }
    
    AscendC::GmFree(log_probs);
    AscendC::GmFree(targets);
    AscendC::GmFree(input_lengths);
    AscendC::GmFree(target_lengths);
    AscendC::GmFree(neg_log_likelihood);
    AscendC::GmFree(log_alpha);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
