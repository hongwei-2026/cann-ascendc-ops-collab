# é“è·¯å°è´¦ â€” MultilabelMarginLoss_2327

| ID | æ—¥æœŸ | ä½œè€… | é˜¶æ®µ | å‡è®¾ | æ–¹æ³• | æ—¶é™ | ç»“æœ | ç»“è®º | è€—æ—¶ |
|----|------|------|------|------|------|------|------|------|------|
| M001 | 2026-11-03 | yuhongming-2026 | å‡†å¤‡ | é¢˜åŒ…å« reference ä¸éª¨æ¶å³å¯å¼€å†™ | è§£å‹ templateï¼›è¯» reference ä¸ ops-nn å…ˆä¾‹ | 0.5h | è¯­ä¹‰å·®å¼‚å·²ç¡®è®¤ | é‡‡çº³ | â€” |
| M002 | 2026-11-03 | yuhongming-2026 | å®ç° | è¡Œåˆ†æ ¸ + è¡Œå†…æ ‡é‡ O(CÂ²) å¯è¿‡åŠŸèƒ½/ç²¾åº¦æ ·ä¾‹ | InferShape/Tiling/Kernelï¼›fp16/fp32/bf16 keyï¼›workspace è¡ŒæŸå¤±å½’çº¦ | æœ¬è¿­ä»£ | æœ¬åœ°è¯­ä¹‰ PASSï¼›å¾…äº‘æµ‹ | é‡‡çº³ä¸º baseline | design.md |
| M003 | 2026-11-03 | hongwei-2026 | å®ç° | èåˆ teammate + design_final classTileï¼›äº‘ç«¯é¢˜ç›®éš”ç¦» | æ‹·åº§ä½ï¼›Host/Kernel æ»‘çª—ï¼›å†™ SYNC_CODEARTS | æœ¬ä¼šè¯ | æœ¬åœ° semantic PASSï¼›åˆç‰ˆè½ç›˜ | é‡‡çº³ï¼›è¿›äº‘æµ‹ | FIX_NOTES |
| M004 | 2026-11-03 | hongwei-2026 | äº‘æµ‹ | é¦–ç‚¹ -e å¯è¿‡ | CodeArts build.sh -e | æœ¬ä¼šè¯ | build OKï¼›is_target PASSï¼›**y WA 1/1** | æ ¹å› ï¼šGM æ ‡é‡å†™ Cache ä¸ä¸€è‡´ | EXP_M001 |
| M004b | 2026-11-03 | hongwei-2026 | ä¿®å¤ | DataCopyPad å†™ y/rowLoss + idle SyncAll + usedCores | push `3ebda45` | æœ¬ä¼šè¯ | å¾…äº‘å¤æµ‹ | é‡‡çº³ | EXP_M001_FIX |
| M004c | 2026-11-03 | hongwei-2026 | äº‘æµ‹ | DMA ä¿®åé¦–æµ‹ | CodeArts -e | æœ¬ä¼šè¯ | **COMPILE FAIL**ï¼š`ScanPrefix() const` â†’ TBuf::Get | å» constï¼›è§ EXP_M002 | EXP_M002 |
| M005 | 2026-11-04 | yuhongming-2026 | éªŒè¯ | åŒæ­¥ tip åæœ¬åœ° -e å¯è¿‡ | åº§ä½åŒæ­¥ hongwei DMA+const ä¿®ï¼›910B4 build + run.sh | æœ¬ä¼šè¯ | **PASSED** y 0/1 Â· is_target 0/24 | **é‡‡çº³**ï¼›å¯äº‘å¤æµ‹ | [EXP_M003](explorations/EXP_M003-local-910b4-pass.md) Â· [ä¼šè¯å¡ç‰‡](explorations/AI-SESSION-20261104-card.md) |
| M006 | 2026-11-04 | hongwei-2026 | ÓÅ»¯ | ÎüÊÕ M003£»ÎŞÕıÀà/È«Ä¿±ê j ¿éÔçÍË£»À© semantic | ±¾»á»° | semantic PASS | ²ÉÄÉ£»´ıÔÆ -e | FIX_NOTES |
| M007 | 2026-11-04 | hongwei-2026 | ÔÆ²â | M006 tip ÔÆ -e | CodeArts | **PASSED** y0/1 is_target0/24 | ²ÉÄÉ | EXP_M004 |
| M008 | 2026-11-04 | hongwei-2026 | Æ½Ì¨ | Ìá½» M006 tip | Æ½Ì¨ÓÃÀı 1-5 | **Pass** 0% ´í£»ÓÃÊ±~713us vs ×îÓÅ~5us | ¹¦ÄÜ²ÉÄÉ£»½øĞÔÄÜÕ¢ | PLATFORM_CASES_1-5 |

| M009 | 2026-11-04 | hongwei-2026 candidate | performance | Unused fixed 16MB workspace and tiny cross-core finalize | On-demand workspace; tiny single-core ordered reduction | <=2h | CPU semantic PASS; actual Process 36 / workspace 42 cases PASS; 72 reference cases generated; CANN/NPU/perf NOT RUN | candidate; M006 remains default | EXP_M009 |
| M010 | 2026-11-04 | hongwei-2026 | ĞŞ¸´ | ¶àºË SyncAll ĞèÏµÍ³ GM£»M009 È¥ 16MB ¡ú ACL 507035 | ¶àºË»Ö¸´ WS_SYS+GetUserWorkspace£»±£Áô none/tiny=0 | ±¾»á»° | ±¾µØ semantic+flow PASS£»´ıÔÆ¸´²â | ºòÑ¡ĞŞ¸´ | EXP_M009 ANALYSIS |
| M010b | 2026-11-04 | hongwei-2026 | ÔÆ²â | 204303 °üÈÔ fe00ab9f/416B | ÅĞ¶¨Î´²¿Êğ M010 | ±¾»á»° | Í¬ 507035£»·Ç»Ø¹é | Ğè pull+rsync ¸´²â | EXP_M009_NGAk |
