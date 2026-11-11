/*!
 * \file confusion_transpose_d_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _CONFUSIONTRANSPOSED_TILING_DATA_H_
#define _CONFUSIONTRANSPOSED_TILING_DATA_H_

constexpr int64_t CTD_MAX_DIM = 8;

struct ConfusionTransposeDTilingData {
    int64_t totalNum = 0;       // 总元素数量
    int64_t blockFactor = 1;    // 每个核处理的输出元素数量
    int64_t ubFactor = 0;       // 输出方向 UB tile 元素数
    int64_t transposeFirst = 1; // 1: permute→reshape; 0: reshape→permute
    int64_t rank = 0;           // perm / shape 长度
    int64_t rankIn = 0;         // 输入 rank
    // 输入 shape
    int64_t dimIn0 = 1;
    int64_t dimIn1 = 1;
    int64_t dimIn2 = 1;
    int64_t dimIn3 = 1;
    int64_t dimIn4 = 1;
    int64_t dimIn5 = 1;
    int64_t dimIn6 = 1;
    int64_t dimIn7 = 1;
    // attr shape（reshape 目标）
    int64_t dimShape0 = 1;
    int64_t dimShape1 = 1;
    int64_t dimShape2 = 1;
    int64_t dimShape3 = 1;
    int64_t dimShape4 = 1;
    int64_t dimShape5 = 1;
    int64_t dimShape6 = 1;
    int64_t dimShape7 = 1;
    // perm
    int64_t perm0 = 0;
    int64_t perm1 = 1;
    int64_t perm2 = 2;
    int64_t perm3 = 3;
    int64_t perm4 = 4;
    int64_t perm5 = 5;
    int64_t perm6 = 6;
    int64_t perm7 = 7;
    int64_t copySpan = 1;     // 实际转置前 shape 的连续后缀
};
#endif
