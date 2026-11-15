/*!
 * \file test_quant_update_scatter.cpp
 * \brief QuantUpdateScatter 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "quant_update_scatter_tiling.h"
#include "../../../op_kernel/quant_update_scatter.cpp"

#include <array>
#include <algorithm>
#include <cmath>
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
    
    // Input fixture is MODE_0: int32 indices, fp16 updates, fp32 scales, int32 zp.
    QuantUpdateScatterTilingData* tilingData = reinterpret_cast<QuantUpdateScatterTilingData*>(tiling);
    std::memset(tilingData, 0, sizeof(*tilingData));
    tilingData->totalNum = size;
    tilingData->blockFactor = 1;
    tilingData->ubFactor = 32;
    tilingData->coreNum = numBlocks;
    tilingData->batchSize = 1;
    tilingData->midSize = 1;
    tilingData->updateAxis = 1;
    tilingData->varAxis = 2;
    tilingData->lastDim = 32;
    tilingData->indexElements = 1;
    tilingData->hasZeroPoints = 1;
    tilingData->reciprocalScale = 0;
    tilingData->indicesIsInt64 = 0;
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((quant_update_scatter<0>), numBlocks, var, indices, updates, quant_scales, quant_zero_points, var_out, workspace, tiling);
    
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


namespace {
struct ScopedQ001Gm {
    uint8_t* ptr;
    explicit ScopedQ001Gm(size_t bytes) : ptr(static_cast<uint8_t*>(AscendC::GmAlloc(bytes))) {}
    ~ScopedQ001Gm() { if (ptr != nullptr) { AscendC::GmFree(ptr); } }
    ScopedQ001Gm(const ScopedQ001Gm&) = delete;
    ScopedQ001Gm& operator=(const ScopedQ001Gm&) = delete;
};

template <uint32_t Mode, bool IndicesInt64 = (Mode == 1)>
void RunGeneratedQ001Case(bool inplace, bool hasZp, bool reciprocal)
{
    constexpr uint32_t cores = 4; // block 3 only participates in copy and SyncAll
    constexpr int64_t batch = 3, middle = 2, rows = 2, varRows = 5, width = 33;
    constexpr size_t total = batch * middle * varRows * width;
    constexpr size_t updateCount = batch * middle * rows * width;
    constexpr size_t indexBytes = IndicesInt64 ? sizeof(int64_t) : sizeof(int32_t);
    constexpr size_t scaleBytes = Mode == 0 ? sizeof(float) : sizeof(uint16_t);
    constexpr size_t zpBytes = Mode == 0 ? sizeof(int32_t) : sizeof(uint16_t);
    ScopedQ001Gm var(total), out(total), indices(batch * indexBytes), updates(updateCount * 2);
    ScopedQ001Gm scales(width * scaleBytes), zp(width * zpBytes), workspace(32);
    ScopedQ001Gm tiling(sizeof(QuantUpdateScatterTilingData));
    ASSERT_NE(var.ptr, nullptr); ASSERT_NE(out.ptr, nullptr); ASSERT_NE(indices.ptr, nullptr);
    ASSERT_NE(updates.ptr, nullptr); ASSERT_NE(scales.ptr, nullptr); ASSERT_NE(zp.ptr, nullptr);
    ASSERT_NE(workspace.ptr, nullptr); ASSERT_NE(tiling.ptr, nullptr);
    std::vector<int8_t> expected(total);
    for (size_t i = 0; i < total; ++i) {
        expected[i] = static_cast<int8_t>(static_cast<int>(i % 255) - 128);
    }
    std::memcpy(var.ptr, expected.data(), total);
    std::memset(out.ptr, 0x5a, total);
    const int64_t indexValues[batch] = {0, 3, 1};
    for (int64_t p = 0; p < batch; ++p) {
        if constexpr (!IndicesInt64) {
            reinterpret_cast<int32_t*>(indices.ptr)[p] = static_cast<int32_t>(indexValues[p]);
        } else {
            reinterpret_cast<int64_t*>(indices.ptr)[p] = indexValues[p];
        }
    }
    for (int64_t column = 0; column < width; ++column) {
        const float scale = column % 3 == 0 ? 0.5f : (column % 3 == 1 ? 1.0f : 2.0f);
        const int32_t zero = static_cast<int32_t>(column % 5) - 2;
        if constexpr (Mode == 0) {
            reinterpret_cast<float*>(scales.ptr)[column] = scale;
            reinterpret_cast<int32_t*>(zp.ptr)[column] = zero;
        } else {
            reinterpret_cast<uint16_t*>(scales.ptr)[column] = FloatToBFloat16(scale);
            reinterpret_cast<uint16_t*>(zp.ptr)[column] = FloatToBFloat16(static_cast<float>(zero));
        }
    }
    for (size_t i = 0; i < updateCount; ++i) {
        // Exactly representable in both fp16 and bf16, with ties and saturation.
        const float value = static_cast<float>(static_cast<int>(i % 255) - 127);
        reinterpret_cast<uint16_t*>(updates.ptr)[i] = Mode == 0 ? FloatToHalf(value) : FloatToBFloat16(value);
        const int64_t column = i % width;
        const int64_t row = (i / width) % rows;
        const int64_t mid = (i / (width * rows)) % middle;
        const int64_t p = i / (width * rows * middle);
        const float scale = column % 3 == 0 ? 0.5f : (column % 3 == 1 ? 1.0f : 2.0f);
        float result = reciprocal ? value * scale : value / scale;
        if (hasZp) { result += static_cast<float>(column % 5) - 2.0f; }
        result = std::max(-128.0f, std::min(127.0f, result));
        const size_t dest = ((p * middle + mid) * varRows + indexValues[p] + row) * width + column;
        expected[dest] = static_cast<int8_t>(std::nearbyint(result));
    }
    auto* td = reinterpret_cast<QuantUpdateScatterTilingData*>(tiling.ptr);
    *td = QuantUpdateScatterTilingData{};
    td->totalNum = total; td->blockFactor = 1; td->ubFactor = 32; td->coreNum = cores;
    td->batchSize = batch; td->midSize = middle; td->updateAxis = rows;
    td->varAxis = varRows; td->lastDim = width; td->indexElements = batch;
    td->hasZeroPoints = hasZp; td->reciprocalScale = reciprocal; td->indicesIsInt64 = IndicesInt64;
    uint8_t* result = inplace ? var.ptr : out.ptr;
    uint8_t* zeroPoints = hasZp ? zp.ptr : nullptr;
    ICPU_SET_TILING_KEY(Mode);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF((quant_update_scatter<Mode>), cores, var.ptr, indices.ptr, updates.ptr, scales.ptr,
        zeroPoints, result, workspace.ptr, tiling.ptr);
    for (size_t i = 0; i < total; ++i) {
        ASSERT_EQ(reinterpret_cast<int8_t*>(result)[i], expected[i]) << "offset=" << i;
    }
}
} // namespace

TEST_F(QuantUpdateScatterKernelTest, q001_mode0_multicore_tail33)
{
    RunGeneratedQ001Case<0>(false, true, false);
}
TEST_F(QuantUpdateScatterKernelTest, q001_mode1_multicore_tail33)
{
    RunGeneratedQ001Case<1>(false, true, false);
}
TEST_F(QuantUpdateScatterKernelTest, q001_mode0_inplace_without_zp_reciprocal)
{
    RunGeneratedQ001Case<0>(true, false, true);
}
TEST_F(QuantUpdateScatterKernelTest, q001_mode1_inplace_without_zp_reciprocal)
{
    RunGeneratedQ001Case<1>(true, false, true);
}

TEST_F(QuantUpdateScatterKernelTest, r001_mode0_indices64)
{
    RunGeneratedQ001Case<0, true>(false, true, false);
}
TEST_F(QuantUpdateScatterKernelTest, r001_mode1_indices32)
{
    RunGeneratedQ001Case<1, false>(false, true, false);
}
TEST_F(QuantUpdateScatterKernelTest, r001_mode0_indices64_inplace_no_zp)
{
    RunGeneratedQ001Case<0, true>(true, false, true);
}
TEST_F(QuantUpdateScatterKernelTest, r001_mode1_indices32_inplace_no_zp)
{
    RunGeneratedQ001Case<1, false>(true, false, true);
}
