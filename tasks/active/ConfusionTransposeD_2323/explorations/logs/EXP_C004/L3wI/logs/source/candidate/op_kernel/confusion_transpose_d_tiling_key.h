/*!
 * \file confusion_transpose_d_tiling_key.h
 * \brief Tiling 模板参数定义
 *
 * schMode:
 *   0 — 2 字节元素（fp16/bf16/int16/uint16），kernel 用 half 位拷贝
 *   1 — 4 字节元素（fp32/int32/uint32），kernel 用 float 位拷贝
 *   2 — 1 字节元素（int8/uint8）
 *   3 — 8 字节元素（int64/uint64）
 */

#ifndef __CONFUSIONTRANSPOSED_TILING_KEY_H__
#define __CONFUSIONTRANSPOSED_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define CONFUSIONTRANSPOSED_TPL_SCH_MODE_0 0
#define CONFUSIONTRANSPOSED_TPL_SCH_MODE_1 1
#define CONFUSIONTRANSPOSED_TPL_SCH_MODE_2 2
#define CONFUSIONTRANSPOSED_TPL_SCH_MODE_3 3

ASCENDC_TPL_ARGS_DECL(
    ConfusionTransposeD,
    ASCENDC_TPL_UINT_DECL(
        schMode, 2, ASCENDC_TPL_UI_LIST,
        CONFUSIONTRANSPOSED_TPL_SCH_MODE_0,
        CONFUSIONTRANSPOSED_TPL_SCH_MODE_1,
        CONFUSIONTRANSPOSED_TPL_SCH_MODE_2,
        CONFUSIONTRANSPOSED_TPL_SCH_MODE_3));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(
        schMode, ASCENDC_TPL_UI_LIST,
        CONFUSIONTRANSPOSED_TPL_SCH_MODE_0,
        CONFUSIONTRANSPOSED_TPL_SCH_MODE_1,
        CONFUSIONTRANSPOSED_TPL_SCH_MODE_2,
        CONFUSIONTRANSPOSED_TPL_SCH_MODE_3)));

#endif
