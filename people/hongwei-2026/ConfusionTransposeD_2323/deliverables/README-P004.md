# P004 平台复测代码包

本地文件：`ConfusionTransposeD_2323_P004_4761239_code.zip`，根目录为 `code/`，含44个已提交源文件，无编译产物。

精确代码 commit：4761239216bd4ec888609fe84b6a2f6e7dd19b94。已在910B3/CANN9.1.0通过样例及22/22 bitwise回归，同机连续行加速2.09～6.07倍；平台性能待复测。来源及逐文件校验见 `P004-SUBMISSION-MANIFEST.json`。

ZIP为本地生成件；可在仓库根复建相同源码内容（ZIP二进制校验可能因打包工具而不同）：

```bash
git archive --format=zip --prefix=code/ --output=people/hongwei-2026/ConfusionTransposeD_2323/deliverables/ConfusionTransposeD_2323_P004_4761239_code.zip 4761239:people/hongwei-2026/ConfusionTransposeD_2323/workdir/code
```

上传此代码包重测平台点1～5并回收性能表。如果平台仍显示ms量级，先核对实际提交版本、case和计时方式。本包是精确已测4761239，不包含随后仅修正日志解析字段的变更。
