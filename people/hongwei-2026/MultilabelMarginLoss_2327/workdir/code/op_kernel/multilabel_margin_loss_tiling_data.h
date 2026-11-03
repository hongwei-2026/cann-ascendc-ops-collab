/*!
 * \file multilabel_margin_loss_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _MULTILABELMARGINLOSS_TILING_DATA_H_
#define _MULTILABELMARGINLOSS_TILING_DATA_H_

struct MultilabelMarginLossTilingData {
    int64_t batchSize = 0;     // N
    int64_t classNum = 0;      // C
    int64_t targetLen = 0;     // K
    int64_t reduction = 1;     // 0=none, 1=mean, 2=sum
    int64_t usedCores = 1;
    int64_t rowsPerCore = 1;   // 前缀分核：核 i 处理 [i*rowsPerCore, min((i+1)*rowsPerCore, N))
    int64_t tailRows = 0;      // 记录尾核行数（kernel 用 min 截断）
    int64_t classTile = 0;     // C 方向滑窗；整行可装下时 == classNum（design_final）
    int64_t targetTile = 0;    // K 方向分块；整行可装下时 == targetLen
};
#endif
