/*!
 * \file p_relu_grad_update_tiling_data.h
 * \brief tiling data struct — P0 per SUMMARY (row-slice / pack-rows)
 */

#ifndef _PRELUGRADUPDATE_TILING_DATA_H_
#define _PRELUGRADUPDATE_TILING_DATA_H_

#include <cstdint>

constexpr int64_t PRELU_TILE_MODE_ROW_SLICE = 0;
constexpr int64_t PRELU_TILE_MODE_PACK_ROWS = 1;

struct PReluGradUpdateTilingData {
    int64_t n = 0;
    int64_t c = 0;
    int64_t inner = 0;
    int64_t rows = 0;
    int64_t totalNum = 0;
    int64_t ubFactor = 0;       // elements per ROW_SLICE chunk (aligned)
    int64_t rowsPerTile = 1;    // whole rows per PACK_ROWS chunk
    int64_t chunksPerRow = 1;   // ROW_SLICE: ceil(inner/ubFactor); else 1
    int64_t blockFactor = 1;    // chunks per core
    int64_t coreNum = 1;
    int64_t tileMode = 0;       // 0 ROW_SLICE, 1 PACK_ROWS
    int64_t dtypeKey = 1;       // 0 fp16, 1 fp32, 2 bf16
};
#endif
