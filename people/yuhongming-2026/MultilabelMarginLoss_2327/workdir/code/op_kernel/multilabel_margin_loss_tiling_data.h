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
    int64_t rowsPerCore = 1;   // 前 usedCores-1 个核的行数
    int64_t tailRows = 0;      // 最后一个核的行数（可与 rowsPerCore 相同）
};
#endif
