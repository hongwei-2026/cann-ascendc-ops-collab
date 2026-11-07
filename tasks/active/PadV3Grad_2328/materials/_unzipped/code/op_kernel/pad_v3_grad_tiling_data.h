/*!
 * \file pad_v3_grad_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _PADV3GRAD_TILING_DATA_H_
#define _PADV3GRAD_TILING_DATA_H_

struct PadV3GradTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个核处理的元素数量
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
};
#endif
