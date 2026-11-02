/*!
 * \file p_relu_grad_update_tiling_data.h
 * \brief tiling data — align design_final §3.3.5
 */

#ifndef _PRELUGRADUPDATE_TILING_DATA_H_
#define _PRELUGRADUPDATE_TILING_DATA_H_

#include <cstdint>

struct PReluGradUpdateTilingData {
    int64_t totalNum = 0;           // M
    int64_t coreNum = 1;            // used AIV cores
    int64_t blockFactor = 1;        // elements per core
    int64_t tileLen = 0;            // elements per UB tile
    int64_t channelNum = 0;         // C
    int64_t innerSize = 0;          // inner = prod(dims[2:])
    int64_t sampleSize = 0;         // S = C * inner
    int64_t weightsPerChannel = 1;  // 1=[C], 0=[1]
    int64_t rowsPerTile = 0;        // mode R; 0 => mode C
};
#endif
