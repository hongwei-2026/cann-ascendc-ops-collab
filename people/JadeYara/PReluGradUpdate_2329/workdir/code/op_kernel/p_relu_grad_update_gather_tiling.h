#ifndef JADEYARA_GATHER_TILING_H
#define JADEYARA_GATHER_TILING_H
#include <cstdint>
struct PReluGradUpdateGatherTilingData {
 uint64_t totalNum, blockSpan, tileLen, channelNum, innerSize;
 uint64_t qCap, cacheWeights, magicS, shiftS, magicC, shiftC;
};
#endif