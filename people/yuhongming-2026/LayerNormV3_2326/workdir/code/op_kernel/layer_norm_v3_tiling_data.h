/*!
 * \file layer_norm_v3_tiling_data.h
 * \brief LayerNormV3 tiling。字段全是 int64，UT 按 int64 顺序打印。
 */

#ifndef _LAYERNORMV3_TILING_DATA_H_
#define _LAYERNORMV3_TILING_DATA_H_

struct LayerNormV3TilingData {
    int64_t rowNum = 0;       // begin_norm_axis 之前的元素个数；全张量归一化时为 1
    int64_t colNum = 0;       // 每个归一化行的元素个数
    int64_t gammaNum = 0;     // gamma/beta 元素个数（与 x 的尾维对齐）
    int64_t rowsPerCore = 0;  // 多数核上的行数（floor）
    int64_t rowRemain = 0;    // 前 rowRemain 个核各多 1 行
    int64_t usedCores = 1;    // 实际启动的 vector 核数
    int64_t tileLength = 0;   // 单次沿列搬入的最大有效元素数
    int64_t tileAlign = 64;   // UB 缓冲长度，64 对齐；尾部补 0 后再归约
    int64_t epsilonBits = 0;  // float32 epsilon 的低 32 位
};
#endif
