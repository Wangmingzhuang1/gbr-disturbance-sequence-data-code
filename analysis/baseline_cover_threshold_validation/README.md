# Baseline cover threshold validation

## 问题

验证基线硬珊瑚覆盖率 10% 是否具有分布或模型层面的分层意义，以及 prior heatwave recurrence 与 absolute loss / raw proportional retention 的关联是否随基线组变化。10% 被视为待检验切点，不预设为生态断点。

## 数据与既有定义

- 输入矩阵：`output/legacy_load_analysis_matrix.csv`（当前终稿 525 reef-years）。
- 变量字典：`output/tables/table_s1_variable_definitions.csv`。
- 主模型：`scripts/05_run_lmm_legacy_model.py`。
- 样本和协变量沿用当前主模型：reef-cluster robust OLS；当前 DHW、当前 wind、5-year heatwave years、5-year storm years、return interval、event type；absolute loss 模型保留 baseline cover 调整。
- raw proportional retention 定义为 `nadir_hc / baseline_hc`，不裁剪、不转换。

## 方法

1. 10% 分组样本、5%–20% 分布/取整/局部密度、协变量标准化差异、缺失与 common support。
2. 以阈值为结点的连续 hinge 模型，检验结点前后 baseline slope 差。
3. recurrence × baseline-group 交互，直接检验两组 recurrence association 差异；另给组内描述模型和调整预测。
4. 对 8%、10%、12%、15% 重复；P 值保留原值，并对同一响应/检验族给出 BH-FDR，绝不据此选择阈值。
5. 单一 4-df natural cubic spline × recurrence 检查平滑异质性，不进行模型搜索。

## 复现

在项目根目录运行：

```powershell
python analysis/baseline_cover_threshold_validation/run_threshold_validation.py
```

脚本只向本目录的 `results/`、`figures/`、报告和检查文件写入。诊断图为英文标签、Times New Roman。
