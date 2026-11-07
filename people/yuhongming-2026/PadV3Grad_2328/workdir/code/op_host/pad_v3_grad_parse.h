/*!
 * \file pad_v3_grad_parse.h
 * \brief 解析 mode 和 paddings。contiguous 为真时按 (front, back) 成对存放。
 */

#ifndef PAD_V3_GRAD_PARSE_H
#define PAD_V3_GRAD_PARSE_H

#include <cstdint>
#include <cstring>

constexpr int64_t kPadV3GradMaxRank = 8;
constexpr int64_t kPadV3GradModeConstant = 0;
constexpr int64_t kPadV3GradModeEdge = 1;
constexpr int64_t kPadV3GradModeReflect = 2;
constexpr int64_t kPadV3GradModeSymmetric = 3;
constexpr int64_t kPadV3GradModeCircular = 4;

inline int64_t PadV3GradParseMode(const char* mode)
{
    if (mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "reflect") == 0) {
        return kPadV3GradModeReflect;
    }
    if (std::strcmp(mode, "constant") == 0) {
        return kPadV3GradModeConstant;
    }
    if (std::strcmp(mode, "edge") == 0 || std::strcmp(mode, "replicate") == 0) {
        return kPadV3GradModeEdge;
    }
    if (std::strcmp(mode, "symmetric") == 0) {
        return kPadV3GradModeSymmetric;
    }
    if (std::strcmp(mode, "circular") == 0) {
        return kPadV3GradModeCircular;
    }
    return -1;
}

template <typename T>
inline bool PadV3GradReadPads(const T* data, int64_t count, int64_t rank, bool contiguous, int64_t* front,
    int64_t* back)
{
    if (data == nullptr || rank < 0 || rank > kPadV3GradMaxRank || count < rank * 2) {
        return false;
    }
    for (int64_t i = 0; i < rank; ++i) {
        int64_t f = 0;
        int64_t b = 0;
        if (contiguous) {
            f = static_cast<int64_t>(data[i * 2]);
            b = static_cast<int64_t>(data[i * 2 + 1]);
        } else {
            f = static_cast<int64_t>(data[i]);
            b = static_cast<int64_t>(data[rank + i]);
        }
        if (f < 0 || b < 0) {
            return false;
        }
        front[i] = f;
        back[i] = b;
    }
    return true;
}

#endif
