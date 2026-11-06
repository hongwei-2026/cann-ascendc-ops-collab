/*!
 * \file layer_norm_v3_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _LAYERNORMV3_TILING_DATA_H_
#define _LAYERNORMV3_TILING_DATA_H_

struct LayerNormV3TilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
};
#endif
