/*!
 * \file ctc_loss_v2_tiling_data.h
 * \brief tiling data struct — 对齐 ops_legacy TIK CTCLossV2 样例路径
 */

#ifndef _CTCLOSSV2_TILING_DATA_H_
#define _CTCLOSSV2_TILING_DATA_H_

struct CtcLossV2TilingData {
    int64_t timeSize = 0;       // T
    int64_t batchSize = 0;      // N
    int64_t classNum = 0;       // C
    int64_t maxTargetLen = 0;   // S（targets 末维 / 用于 U=2S+1）
    int64_t blank = 0;
    int64_t zeroInfinity = 0;   // 0/1
    int64_t targetsIsInt64 = 0; // 0=int32, 1=int64
    int64_t usedCores = 1;
};
#endif
