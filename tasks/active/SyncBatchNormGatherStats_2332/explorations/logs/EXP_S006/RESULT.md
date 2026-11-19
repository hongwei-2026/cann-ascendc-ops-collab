# EXP_S006 — 云冒烟 + 舍入边界

| 项 | 内容 |
|----|------|
| tip | `ff6b681`（含 Kernel 改动 tip `b2271b4`） |
| 主机 | `dfdd97f8c4e84acbaa54982714e4d993` |
| `bash build.sh -j8 -e` | **PASS**；四输出 mismatched=**0/6**；末行 `PASSED` |
| `python3 tests/s006_rounding.py` | **PASS**；四输出 mismatched=**0/6**；末行 `PASSED` |
| 全文 | `build_e.log` / `rounding.log`（本目录） |

样例 ≠ 平台 50 点。下一步按 `deliverables/SUBMIT_LINKS_S006.md` 交平台。
