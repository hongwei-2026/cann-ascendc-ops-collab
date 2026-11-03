/*!
 * \file p_relu_grad_update_tiling_data.h
 * \brief Host/kernel shared tiling data for PReluGradUpdate.
 */

#ifndef PRELUGRADUPDATE_TILING_DATA_H
#define PRELUGRADUPDATE_TILING_DATA_H

#include <cstdint>

struct PReluGradUpdateTilingData {
    int64_t totalNum = 0;
    int64_t coreNum = 1;
    int64_t blockFactor = 0;
    int64_t tileLen = 0;
    int64_t channelNum = 0;
    int64_t innerSize = 0;
    int64_t sampleSize = 0;
    int64_t weightsPerChannel = 0;
    int64_t rowsPerTile = 0;
};

#endif
