/*!
 * \file quant_update_scatter_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _QUANTUPDATESCATTER_TILING_DATA_H_
#define _QUANTUPDATESCATTER_TILING_DATA_H_

struct QuantUpdateScatterTilingData {
    int64_t totalNum = 0;       // var 元素总数（用于整段拷贝）
    int64_t blockFactor = 1;    // 每个核处理的 batch（updates.dim0）数
    int64_t ubFactor = 0;       // 每次 UB 处理的 lastDim 元素数
    int64_t coreNum = 1;        // 实际使用核数
    int64_t batchSize = 0;      // updates.shape[0]（与 indices 对齐的批次数）
    int64_t midSize = 1;        // product(shape[1:-2])，无中间维时为 1
    int64_t updateAxis = 0;     // updates 在 axis 维长度
    int64_t varAxis = 0;        // var 在 axis 维长度
    int64_t lastDim = 0;        // quant_axis / 最后一维长度
    int64_t indexElements = 0;  // indices 元素个数
    int64_t hasZeroPoints = 0;  // 1: 有 quant_zero_points
    int64_t reciprocalScale = 0;// 1: updates * scales；0: updates / scales
    int64_t indicesIsInt64 = 0; // 1: indices 为 int64
};
#endif
