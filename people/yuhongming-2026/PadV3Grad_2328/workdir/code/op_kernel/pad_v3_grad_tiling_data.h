/*!
 * \file pad_v3_grad_tiling_data.h
 * \brief 全部 int64，UT 按这个顺序把 tiling 打成空格分隔的字符串。
 */

#ifndef _PADV3GRAD_TILING_DATA_H_
#define _PADV3GRAD_TILING_DATA_H_

struct PadV3GradTilingData {
    int64_t rank = 0;
    int64_t mode = 0; // 0 constant, 1 edge/replicate, 2 reflect, 3 symmetric, 4 circular
    int64_t totalX = 0;
    int64_t totalY = 0;
    int64_t usedCores = 1;
    int64_t rowsPerCore = 0;
    int64_t rowRemain = 0;
    int64_t yRowLength = 0;
    int64_t xDim[8] = {};
    int64_t yDim[8] = {};
    int64_t front[8] = {};
    int64_t back[8] = {};
};
#endif
