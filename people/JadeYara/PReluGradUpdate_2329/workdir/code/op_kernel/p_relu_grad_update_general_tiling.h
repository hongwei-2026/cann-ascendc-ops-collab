#ifndef JADEYARA_GENERAL_TILING_H
#define JADEYARA_GENERAL_TILING_H
#include <cstdint>
// CANN 9.3 tbe/common/utils/para_check.py:43,651-653: RANK_LIMIT=8.
constexpr uint32_t PRELU_GENERAL_RANK = 8;
constexpr uint64_t PRELU_GENERAL_TILE = 2048;
struct PReluGradUpdateGeneralTilingData {
 uint64_t totalNum, blockSpan, tileLen, splitPeriod, rank;
 uint64_t sourceElements[3], identity[3];
 uint64_t dims[PRELU_GENERAL_RANK], periods[PRELU_GENERAL_RANK];
 uint64_t strides[3][PRELU_GENERAL_RANK], suffix[3][PRELU_GENERAL_RANK];
 int32_t coefficient[3][PRELU_GENERAL_RANK];
 int32_t multiplier[PRELU_GENERAL_RANK], shift[PRELU_GENERAL_RANK];
};
static_assert(sizeof(PReluGradUpdateGeneralTilingData)==760,"General metadata ABI");
#endif
