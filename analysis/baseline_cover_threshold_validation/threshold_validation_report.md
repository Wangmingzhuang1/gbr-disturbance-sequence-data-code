# Baseline cover threshold validation report

## 决策结论

- **不支持把 10% 当作生态状态分界。** 10% 附近没有可见分布断点；loss/retention 的 recurrence × group 直接交互均不显著；absolute-loss hinge 的名义 P 值不构成阈值特异证据。
- **存在的信号更符合连续几何，而非 10% 离散断点。** absolute-loss 的固定 4-df spline 交互提示平滑异质性，不能反推 10% 或任何机制。
- **组间不可作可比状态对照。** <10% 仅 60 reef-years，且 common support 达到预设的严重受限规则。
- 5%/10% baseline-cutoff retention 仅保留为既有敏感性分析；不作为本验证的替代响应。

## 数据事实

- 当前分析矩阵：525 reef-years、92 个独立 reefs；年份 1987–2025。
- 基线 <10%：60 reef-years、33 reefs（Heatwave Only 38、Storm Only 18、Concurrent 4）；基线 ≥10%：465 reef-years、92 reefs（Heatwave Only 263、Storm Only 161、Concurrent 41）。
- 10% 左右各 1 percentage point 的样本数为 12/11，密度比（右/左）=0.917，等密度二项检验 P=1.0000。精确整数值占比 0.000，精确 0.5% 倍数占比 0.000；诊断图未见 10% 处断裂或堆积。
- 最大连续协变量绝对标准化差异为 Cumulative storm wind (5-year)：|SMD|=0.451。标准化差异用于衡量不平衡，不以 P 值代替。
- propensity-score overlap coefficient=0.699，AUC=0.685；低/高组落在共同范围外的比例分别为 0.017 / 0.215。组间 common support 判定：严重受限。因此分层回归不能解释为可比生态状态之间的对照。
- 当前 525 行矩阵未保留 baseline year / nadir year，未自行重建；可用 observation-window 变量为 `yrs_since_last_dist`。缺失详情见 `results/missingness.csv`。

## 统计结果


### absolute loss

- 10% hinge 的斜率差（阈值上方减下方）：0.639，95% CI [0.017, 1.262]，P=0.0441。
- 对应四阈值检验族的 BH-FDR P=0.0588；不能把未经校正的单一切点结果当作断点证据。
- recurrence × 高基线组交互：-1.208，95% CI [-3.362, 0.945]，P=0.2714。这是组间异质性的直接检验。
- 描述性分组回归：低组 recurrence 系数 -0.942 [95% CI -3.405, 1.522]，R²=0.108；高组 -2.597 [-3.856, -1.338]，R²=0.401。一组显著、另一组不显著不构成组间差异证据。

### raw proportional retention

- 10% hinge 的斜率差（阈值上方减下方）：0.138，95% CI [-0.012, 0.289]，P=0.0719。
- 对应四阈值检验族的 BH-FDR P=0.0851；不能把未经校正的单一切点结果当作断点证据。
- recurrence × 高基线组交互：0.098，95% CI [-0.347, 0.543]，P=0.6672。这是组间异质性的直接检验。
- 描述性分组回归：低组 recurrence 系数 -0.146 [95% CI -1.051, 0.759]，R²=0.071；高组 0.061 [0.007, 0.114]，R²=0.124。一组显著、另一组不显著不构成组间差异证据。

### 阈值敏感性与连续检查

- 8%、10%、12%、15% 全部预先列出并重复相同 hinge、交互和分组流程；原始 P 值与同一响应/检验族的 BH-FDR 值见 `results/threshold_sensitivity.csv`。未扫描其他切点，也未选择“最佳阈值”。
- absolute-loss 交互在四个阈值均 P<0.05：否；raw-retention 交互在四个阈值均 P<0.05：否。
- loss_abs：baseline spline × recurrence 联合检验 P=0.0001（固定 4 df spline，无模型搜索）。
- retention：baseline spline × recurrence 联合检验 P=0.1488（固定 4 df spline，无模型搜索）。

### 四类 retention 结果的边界

- raw proportional retention（全 525 行）recurrence 系数 -0.003，95% CI [-0.096, 0.091]，P=0.9572。
- TOST 是对上述系数的等效性检验，不是新响应：margin=0.100，90% CI [-0.081, 0.076]，TOST P=0.0201。
- baseline-cutoff retention 是分母敏感性：baseline >5% 时系数 0.048，baseline >10% 时 0.061；它们不能替代 raw retention 或组间交互。

## 对前提假设的回答

1. **10% 是否有可见分布断点？** 否。5%–20% 直方图连续，9%–10% 与 10%–11% 为 12/11，且无精确 10% 或规则取整堆积。
2. **基线—响应几何是否在 10% 改变？** 由连续 hinge 的斜率差直接回答，不能由两个独立分组模型代替。
3. **recurrence association 是否随 10% 分组变化？** 由 interaction coefficient 及其 CI/P 直接回答；分组显著性只作描述。
4. **结论是否依赖人为切点？** 由四阈值结果和固定 spline 检查共同判断，不按显著性挑选阈值。

## 生态解释边界

- 10% 缺少作为离散生态状态分界的经验支持：分布无可见断点，直接 recurrence 交互不显著，absolute-loss hinge 的单点名义证据经四阈值 BH-FDR 后不再低于 0.05，且连续样条显示的是平滑异质性而非 10% 跳变。建议只把 5%/10% baseline-cutoff retention 保留为敏感性分析。
- 本分析只支持或不支持“统计关联异质性”，不证明遗传多样性、共生体灵活性、功能冗余、藻类优势或繁殖体受限等机制。
- 不推断 phase shift、community transition、physiological adaptation 或单一分类过滤机制。
- 不使用 composition explains/mediates/causes lower loss 的因果措辞。
- absolute loss、raw proportional retention、既有 TOST 与 baseline-cutoff retention 是四个不同问题；登记见 `results/response_metric_context.csv`。

## 统计谬误核查（11/11）

- Simpson：已用 event type/sector 分布和分组模型检查方向；未把聚合与分组差异互换。
- Ecological fallacy：推断单位保持 reef-year/reef，不外推个体珊瑚机制。
- Berkson/collider：样本由现有事件矩阵构造，选择机制仍可能限制外推；未新增事后控制变量。
- Base-rate neglect：不适用诊断准确率问题。
- Regression to mean：baseline 与 absolute loss 存在数学/选择几何，故用 hinge 并保留 baseline 调整，不作适应性解释。
- Survivorship：baseline/nadir 可用性选择继承自主矩阵，无法由本验证消除。
- Look-elsewhere：只检验预定 8/10/12/15，并报告全部结果及 FDR。
- Forking paths：固定主协变量、固定 4-df spline，不做模型/切点搜索。
- Correlation ≠ causation：全程使用 association。
- Reverse causality：prior recurrence 在时间上先于 target event，但观察性混杂仍不能排除。
