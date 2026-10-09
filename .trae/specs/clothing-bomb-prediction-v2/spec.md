# 服装爆款 AI 预测系统 v2 — 产品需求文档

## Overview

- **Summary**: 对现有"虚拟消费者评审系统"进行根本性重构。旧系统基于 Spearman 排序相关和 3 Loop Lasso 回归 calibration，存在 calibration 学歪、分级规则单信号触发误判、记忆模块缺失、输入特征不足等问题。新系统将彻底重写 calibration 层（AUC 优化 + LightGBM 爆款分类器）、新增品牌记忆库（加权相似度匹配）、升级 PatternMiner 为结构化规则引擎、扩充输入特征（F11-F20 版型文本规则提取）、重构分级输出（爆款概率 + 置信区间 + 独立 P 款识别）、改造 UI 适配多角色。
- **Purpose**: 解决 mipo 品牌在新品内审阶段无法精准识别爆款潜质款的问题。目标是让系统在 60 个历款校准后，能在新品一批次中识别出 4-5 个真爆款（假设批次有 6 个），支持供应链提前备货和运营精准投放。
- **Target Users**: 品牌负责人（季末复盘年度汇报）、运营负责人（投流预算分配）、供应链负责人（备货量决策）。核心使用场景是**新品内审阶段**（上市前，内审评级前）。

## Goals

1. 系统能在新品内审阶段（没有内审评级、没有销量数据时）输出**爆款概率**，阈值由 calibration 从历史数据自动学习，**不硬编码批次内比例**（20 个款可以 0 个爆款，也可以 8 个）。
2. 系统预测性能超过内审分级 baseline。Phase 0 baseline 验证显示：Baseline E（品类+季节+价格+F11-F20 版型文本特征）AUC-ROC 0.78，Top-10 Recall 75%（3/4 爆款），已接近内审分级 S>A>P 水平。新系统加 VLM 图片特征 + 人设投票 + 记忆库 + LightGBM 分类器后，目标 AUC-ROC ≥ 0.85，Top-10 Recall ≥ 80%。
3. 每个款式独立分级，分级依据爆款概率绝对值，不依赖批次内相对排名。
4. 新增品牌记忆库：预测时对每个新款式从历史款式库中找 top-5 相似款（加权相似度，权重来自 calibration 的特征重要性），输出邻居爆款率作为预测信号和可解释性依据。
5. PatternMiner 升级为结构化爆款规则引擎：从历史数据中挖爆款基因规则（如"面料科技含量高 + 版型经典 + 颜色安全 → 爆款率 82%"），规则作为分类器输入特征，同时在 UI 上可解释展示。
6. P 款独立识别（设计展示款，独立于跑量款概率分级）：P 款可能卖得好也可能卖得差，不参与爆款概率排序，但需要被识别出来给供应链和运营不同的行动建议。
7. 内审分级是系统的**预测目标**之一（新品内审前系统要预测给内审参考），不是输入特征（新品时还没有内审分级）。
8. 版型/设计描述列（Excel 列 E）通过规则引擎提取 F11-F20 结构化特征，同时喂给 VLM prompt 和人设 prompt，增强视觉打分和购买意愿判断的上下文。

## Non-Goals

- 不解决实时销量预测问题（系统是新品预测，不是上市后调整）。
- 不做渠道级（淘宝/抖音/线下）的精细化投放预算分配（只给"重点投流"或"常规运营"的粗粒度建议）。
- 不做面料成本精确估算（只做成本感知 ordinal 特征）。
- 不替换 VLM backend（Mock backend 暂时保留用于本地开发，生产使用真实 VLM）。
- 不做 Streamlit 外的独立前端或 API 服务（继续用 Streamlit）。

## Background & Context

### Phase 0 Baseline 验证结果（2026-10-09）

用 mipo 品牌 60 个历款的真实数据跑了多个 baseline：

| Baseline | 特征来源 | AUC-ROC | Top-10 Recall | 说明 |
|---|---|---|---|---|
| D | 品类+季节+价格（新品元数据） | 0.36 | 0% | ❌ 完全没用 |
| E | D + F11-F20 版型文本规则提取 | 0.78 | 75% | ✅ 达到内审分级水平 |
| F | 内审分级 S>A>P（人类专家） | — | 75% | 目标 baseline |
| G | 30天标签（作弊） | — | 100% | 理论上限 |

**关键洞察**：
- 纯品类/价格/季节本身没有预测力（AUC=0.36，比随机还差），必须加版型内容信息。
- 只用规则引擎从版型描述文本提取 10 个特征（F11-F20），RandomForest 就能达到 AUC-ROC 0.78，说明文本里藏着明确的爆款信号。
- 4 个真爆款中 Baseline E 识别出 3 个（MPEWQWT06 外套、MPEWDYR27 羽绒服、MPELQKZ20 长裤），还有 1 个（MPEWXKZ10 短裤）没被任何 baseline 识别出来，大概率需要 VLM 图片特征或人设投票才能补。

### Ground truth 数据画像

60 个历款，正样本 4 个（爆款率 6.7%，类别严重不平衡）：
- MPEWQWT06 外套 9000件(爆) ← 用户指定的经典版型爆款
- MPELQKZ20 长裤 7000件(旺)
- MPEWXKZ10 短裤 5000件(旺)
- MPEWDYR27 羽绒服 3000件(爆)

爆款定义：最终销量 ≥ 品类 top10% 阈值 AND 30 天标签 ∈ {爆, 旺}

### 旧系统问题根因

1. **Spearman 排序相关 ≠ 爆款识别**：Spearman 优化的是"排序对齐"，但用户要的是"识别爆款"（分类问题）。60 样本 + 5 正样本下 Spearman 不稳定。
2. **3 Loop Lasso 维度灾难**：Loop2 拟合 30 人设权重用 10 样本 → 全选 0；Loop3 拟合引擎权重学到负相关 → calibration 后排序反了。
3. **Path B 分级规则单信号触发**：`nat<6 OR live<6 OR opp>15%` 任一命中就判风险，calibration 微调坏一个信号就把爆款送上死刑台。
4. **没有记忆模块**：每次独立预测，不利用品牌历史款式库的爆款基因。
5. **输入特征不足**：版型/设计描述文本没有结构化提取，没有喂给 VLM 和人设。

## Functional Requirements

### 数据管线层

- **FR-1**: 系统能从 Excel 批量加载历款或新品，提取款号、品类、季节、售价、年份、版型/设计描述文本等字段。
- **FR-2**: 版型/设计描述文本通过规则引擎（关键词匹配 + 正则）提取 F11-F20 共 10 个结构化特征：
  - F11 面料科技含量（0-10）：科技/机能/速干/吸湿/弹力/SGS 检测等关键词数量加权
  - F12 面料成本感知（0-10）：灯芯绒/羊毛/混纺/科技面料 → 高分；棉/涤 → 低分
  - F13 版型宽松度（0-10）：宽松/落肩/直筒 → 高分；修身/紧身 → 低分
  - F14 版型经典度（0-10）：经典/基础/复古 → 高分；独家/立体/茧型 → 低分
  - F15 功能细节数量（计数）：口袋/抽绳/弹性/印花等设计细节关键词命中数
  - F16 版型描述长度（计数）：版型段落字数
  - F17 面料描述长度（计数）：面料段落字数
  - F18 颜色数量（计数）：文本中颜色分割后的数量
  - F19 颜色安全度（0-10）：黑白灰藏青 → 高分；粉黄荧光 → 低分
  - F20 检测认证数（计数）：SGS/ISO/OEKO-TEX 等认证关键词命中数
- **FR-3**: 规则引擎提取失败时（如文本为空、格式异常）有 fallback 策略：使用默认值 + 标记为缺失，不阻断流程。
- **FR-4**: 新版型/设计描述文本（FR-2）能被拼接到 VLM prompt 和人设 prompt 中，增强视觉打分和购买意愿判断的上下文。

### 品牌记忆库

- **FR-5**: 系统构建并维护一个品牌记忆库（Brand Memory DB），存储格式：历款每条 = {style_id, 品类, 季节, 售价, VLM 10 特征分, F11-F20 版型文本特征, 人设加权分, 30天标签, 最终销量, is_bomb 爆款标签, P款标记}。
- **FR-6**: 记忆库初始化时，calibration 产物（校准后的特征重要性权重）被加载。相似度搜索使用加权欧氏距离，每个特征的权重来自 calibration 的 permutation importance。
- **FR-7**: 对每个新款式，从记忆库中搜索 top-5 相似款（全局搜索，但与新款式同品类的邻居权重 × 2，不同品类的 × 0.5）。
- **FR-8**: 相似度搜索输出三个记忆信号：
  - neighbor_bomb_rate: 邻居中爆款的比例（0-1）
  - neighbor_avg_sales: 邻居平均最终销量
  - neighbor_category_consistency: 邻居与新款式同品类的比例（作为"跨品类参考价值低"的信号）
- **FR-9**: 记忆库可增量更新：每次 calibration 跑完后，新数据追加进库；calibration 跑完后特征重要性权重更新。

### Calibration 层（彻底重写）

- **FR-10**: Calibration 从 Excel 加载历史数据后，构建 Ground truth：
  - 最终销量 → 数值化（"9000+" → 9000）
  - 爆款标签 is_bomb = (最终销量 ≥ 品类 top10% 阈值) AND (30天标签 ∈ {爆, 旺})
- **FR-11**: Calibration 不再使用 Spearman rank correlation 或 3 Loop Lasso 回归，替换为四步流程：
  1. 特征重要性评估（permutation importance / mutual information）：输出每个特征对爆款识别的贡献度 → 给记忆库相似度计算当权重
  2. 爆款分类器训练（LightGBM 或 Logistic Regression，具体算法待验证选优）：输入 = 所有 VLM 特征 + F11-F20 + 品类/季节/价格 + 人设分 + 记忆信号 + PatternMiner 规则命中，目标 = is_bomb（二分类），优化指标 = AUC-ROC + AUC-PR，5-fold stratified CV
  3. PatternMiner 规则提取（从训练好的分类器 + 历史数据）
  4. P 款识别规则阈值校准（规则从内审分级 P 款的特征分布自动学习，不是硬编码）
- **FR-12**: Calibration 产物保存到 `brand_profiles/{brand}/calibrated/` 目录，包含：
  - calibration_report.json（CV 结果、AUC、特征重要性、分类器阈值）
  - classifier_model.pkl（训练好的 LightGBM/LogisticRegression 模型）
  - feature_weights.json（校准后的特征权重，给记忆库用）
  - pattern_rules.json（PatternMiner 挖的爆款基因规则）
  - memory_db.parquet（更新后的品牌记忆库）
  - p_rule.json（P 款识别规则阈值）
- **FR-13**: Calibration 流程有完整的 validation 报告：展示 baseline（内审分级）vs 校准后分类器的 AUC、Precision@K、Recall@K 对比，让用户看到校准是否有效。
- **FR-14**: 旧 calibration（3 Loop Lasso）标记为 deprecated，保留文件但不再被调用。

### PatternMiner 规则引擎（重定义角色）

- **FR-15**: PatternMiner 从校准后的分类器和历史数据中提取结构化爆款基因规则，规则格式：
  - 每条规则 = {条件: 特征组合 + 阈值, 支持度: 命中样本数, 置信度: 命中后爆款率}
  - 例: "F08面料感知≥7 AND F09品牌调性<4 AND F11科技含量≥5 → 爆款率 82% (支持 6 样本)"
- **FR-16**: 规则匹配引擎：对每个新款式逐条匹配 PatternMiner 规则，输出 rule_hits = [{rule_id, confidence, matched_conditions}] 数组，作为分类器的额外输入特征。
- **FR-17**: PatternMiner 规则在单款报告中可解释展示，告诉用户"这款命中了哪些爆款基因规则，每条规则的置信度"。

### 爆款概率 + 分级输出

- **FR-18**: 每个款式独立跑 prediction（不依赖批次内其他款式），输出爆款概率（0-100%）和置信区间（bootstrap 计算，如 72% ± 10% → 62%-82%）。
- **FR-19**: 分级标签完全由爆款概率（calibration 学出来的阈值）推导，**不硬编码批次内比例**：
  - 爆款概率 ≥ calibration 学出的 S 阈值（例 70%）→ "多备货 + 重点投流"
  - 爆款概率 ≥ calibration 学出的 A+ 阈值 AND < S 阈值 → "正常备货 + 常规运营"
  - 爆款概率 < A+ 阈值 → "保守备货 + 观察"
  - P 款命中 → 独立标记 "设计展示款"（供应链小批量 + 运营品牌展示为主，不进上面三档）
- **FR-20**: P 款识别规则从 calibration 中学习，不是硬编码。P 款 = 设计定位款，可能卖得好也可能卖得差，独立于跑量款概率分级。

### PredictionPipeline 改造

- **FR-21**: PredictionPipeline 在跑 prediction 前，加载 calibration 产物（如果存在）；如果不存在，用 default baseline（Baseline E：品类+季节+价格+F11-F20，不用 VLM/人设/记忆库/PatternMiner）。
- **FR-22**: PredictionPipeline 输出扩展后的预测结果，包含：
  - 旧的 VLM 10 特征分 + 人设投票 + 渠道分
  - 新增的 F11-F20 版型文本特征
  - 新增的记忆信号（neighbor_bomb_rate, neighbor_avg_sales, neighbor_category_consistency + top-5 邻居款式列表）
  - 新增的 PatternMiner 规则命中
  - 新增的爆款概率 + 置信区间
  - 新增的 P 款标记
  - 新增的内审分级预测（作为系统预测目标之一）

### UI 改造

- **FR-23**: 首页新增"品牌记忆库"状态展示：已积累 N 款，覆盖 X 品类，最近校准时间，校准后 AUC 提升多少。
- **FR-24**: 单款报告新增以下模块：
  - 🧠 **记忆匹配**: "最像的 5 款历款: A(爆, 92%), B(旺, 87%)... 邻居爆款率 60%"，附邻居款式卡片
  - 🧬 **爆款基因**: "命中 S-01(置信度82%), S-03(置信度75%)..."，附规则详情
  - 📊 **爆款概率**: 72%（置信区间 62-82%），进度条可视化
  - 👗 **版型基因雷达图**: F11-F20 十维雷达图，展示与品牌爆款基因的相似性
- **FR-25**: 分级总表重构：
  - 去掉"Top 20% = S"硬编码，每个款独立分级
  - 新增"爆款概率"列（带进度条）
  - 新增"P 款"标记列
  - 新增"记忆邻居爆款率"列
  - 排序改为按爆款概率降序（不是 final_score）
- **FR-26**: 支持三视角切换（供应链 / 运营 / 品牌负责人），各自默认展示不同的列聚焦：
  - 供应链：备货建议（概率阈值映射的备货量）+ P款小批量标记 + 可解释记忆依据
  - 运营：投流建议（重点投流 vs 常规运营）+ 渠道匹配度 + 人设加权分
  - 品牌负责人：分级分布统计 + 预测爆款清单 + 与内审分级预测对比
- **FR-27**: Calibration 页改造：
  - 展示 calibration validation 报告（baseline vs 校准后 AUC、Precision@K、Recall@K 对比）
  - 展示特征重要性排名
  - 展示 PatternMiner 挖出来的爆款基因规则
  - 旧 3 Loop 标记 deprecated，不再展示或运行
- **FR-28**: 季末复盘页改造：上传实际销量后，系统自动对比预测爆款清单和实际爆款，输出准确率指标（Recall@K、Precision@K、AUC），供品牌负责人年度汇报。

### Prompt 增强

- **FR-29**: VLM prompt 和人设 prompt 都注入版型/设计描述文本，增强视觉打分和购买意愿判断的上下文。VLM prompt 模板：[品牌专业知识] + [VLM 特征定义] + [图片] + [版型面料描述文本] → 10 个特征分。人设 prompt 模板：[人设定义] + [款式基本信息] + [版型面料描述] + [记忆邻居爆款基因参考] → 购买/不购买 + 加权理由。

## Non-Functional Requirements

- **NFR-1**: 现有 60 个 unit test 继续通过（无回归）。
- **NFR-2**: PredictionPipeline 在 Mock backend 下，单款 prediction 运行时间 < 2 秒（不包含 VLM API 调用）。
- **NFR-3**: Calibration 流程在 60 个历款上运行时间 < 30 秒（不包含 PatternMiner 规则提取）。
- **NFR-4**: Streamlit app 启动时间 < 10 秒（localhost 下）。
- **NFR-5**: 所有 calibration 产物文件有版本号和向后兼容机制（旧 calibration 产物不能导致新系统 crash）。

## Constraints

### Technical

- 项目使用 Streamlit + Python 3.x + pandas/numpy/scikit-learn 栈。
- VLM backend 有 Mock 和 真实 API 两种，Mock 用于本地开发测试。
- LightGBM 或 Logistic Regression（sklearn）二选一作为爆款分类器（待验证选优）。
- 不能引入重型依赖（如 PyTorch）。

### Business

- calibration 训练样本只有 60 个（正样本 4 个），严重类别不平衡。calibration 算法必须能处理（scale_pos_weight / class_weight='balanced' / stratified CV）。
- P 款定义（设计展示款）是业务概念，需要内审分级历史数据来学习识别规则阈值。

### Dependencies

- 真实 VLM API（Mock 可替代，但生产需要）。
- 用户上传的 Excel 格式必须包含：款号、品类、内审分级、30天销售标签、版型/设计描述、季节、售价、最终销量、年份。

## Assumptions

- 用户能保证 Excel 字段格式稳定（特别是版型/设计描述列的"面料/版型/颜色"三段式结构）。
- 60 个历款数据里 4 个正样本足够 baseline，calibration 能学到初步信号；后续积累更多样本后效果会提升。
- PatternMiner 在只有 4 个正样本时挖不到太多稳健规则，但框架先搭好，后续加样本就能用。
- 记忆库的相似度权重在 calibration 后更新，但 calibration 本身需要记忆库的初始状态——这里有鸡生蛋问题，先解耦：calibration 只用全局特征重要性（permutation importance），记忆库相似度等 calibration 跑完再加载。

## Acceptance Criteria

### AC-1: Baseline 验证通过（已完成）
- **Type**: `rule`
- **Given**: mipo 品牌 60 个历款 Excel，包含版型/设计描述列
- **When**: 运行 Baseline E（品类+季节+价格+F11-F20 版型文本特征）
- **Then**: AUC-ROC ≥ 0.78，Top-10 Recall ≥ 75%（3/4 真爆款）
- **Pass Condition**: baseline 脚本输出满足上述阈值
- **Evidence**: Phase 0 运行日志（已存在 `/workspace/`）

### AC-2: 新系统 PredictionPipeline AUC > Baseline E
- **Type**: `rule`
- **Given**: 60 个历款数据，calibration 跑完并生成分类器
- **When**: 用 5-fold CV 评估 calibration 后分类器（输入 = VLM 特征 + F11-F20 + 记忆信号 + PatternMiner 规则命中）
- **Then**: AUC-ROC > 0.78（Baseline E 水平）
- **Pass Condition**: calibration validation 报告中校准后 AUC > Baseline AUC
- **Evidence**: calibration_report.json 中的 comparison 字段

### AC-3: 新系统 Top-10 Recall ≥ 80%
- **Type**: `rule`
- **Given**: 60 个历款，calibration 后分类器
- **When**: 按爆款概率降序排序，取 Top-10
- **Then**: 真爆款召回率 ≥ 80%（≥ 3.2 个，实际就是 4/4）
- **Pass Condition**: Top-10 里至少命中 4 个真爆款
- **Evidence**: calibration_report.json 中的 precision_recall_at_k 字段

### AC-4: 分级不依赖批次内相对排名
- **Type**: `rule`
- **Given**: 一批次 N 个款式（N = 任意正整数）
- **When**: 系统对这 N 个款式跑 prediction
- **Then**: 分级标签完全由每个款式自身的爆款概率（calibration 学出的阈值）决定，**批次内可以有 0 个爆款，也可以有 N 个**
- **Pass Condition**: 分级逻辑中不存在任何 `rank`、`percentile`、`top_k` 等批次相对排名操作
- **Evidence**: 代码审查 + 测试用例（构造一批全"烂"数据看是否正确输出 0 个爆款）

### AC-5: 每个款独立输出爆款概率 + 置信区间
- **Type**: `rule`
- **Given**: 任意一个款式（新品或历款）
- **When**: 系统跑 prediction
- **Then**: 输出爆款概率（0-100%）和置信区间（如 72% ± 10%）
- **Pass Condition**: prediction result 对象包含 `bomb_probability: float` 和 `bomb_probability_ci: (low, high)` 字段
- **Evidence**: unit test + 单款报告 UI 展示

### AC-6: 品牌记忆库相似度搜索返回 top-5 邻居
- **Type**: `rule`
- **Given**: 至少 30 个历款的记忆库
- **When**: 对一个新款式跑相似度搜索
- **Then**: 返回 top-5 邻居款式列表，每条含 style_id、相似度分数、品类标签、爆款标签，以及三个记忆信号（neighbor_bomb_rate, neighbor_avg_sales, neighbor_category_consistency）
- **Pass Condition**: 返回结果长度 = 5（或记忆库不足 5 时返回全部），包含上述三个记忆信号
- **Evidence**: unit test + 单款报告 UI 展示"最像的 5 款历款"

### AC-7: F11-F20 规则引擎对任意 Excel 版型描述文本能输出 10 个特征
- **Type**: `rule`
- **Given**: Excel 列 E 版型/设计描述文本（任意格式）
- **When**: 规则引擎处理该文本
- **Then**: 输出 F11-F20 共 10 个数值特征，缺失时有 fallback 默认值，不抛异常
- **Pass Condition**: 对 60 个历款的版型描述文本，规则引擎全部成功输出，无 crash
- **Evidence**: unit test + Phase 0 baseline 运行已验证

### AC-8: P 款独立识别（不进爆款概率排序）
- **Type**: `rule`
- **Given**: 一批次包含 P 款和非 P 款
- **When**: 系统完成所有 prediction 和分级
- **Then**: P 款被独立标记为"设计展示款"，不参与爆款概率排序，但输出供应链建议（小批量备货）和运营建议（品牌展示为主）
- **Pass Condition**: P 款列表与非 P 款列表分离显示；一款被标记 P 款时，其爆款概率显示在括号里但不作为主分级标签
- **Evidence**: UI 展示 + unit test

### AC-9: PatternMiner 规则匹配输出 rule_hits
- **Type**: `rule`
- **Given**: calibration 产物包含 PatternMiner 规则
- **When**: 对一个新款式跑规则匹配引擎
- **Then**: 输出 rule_hits = [{rule_id, confidence, matched_conditions}]，每条命中规则含置信度
- **Pass Condition**: rule_hits 数组非空（有爆款特征命中时）或空数组（无命中时）
- **Evidence**: unit test + 单款报告 UI 展示"爆款基因"

### AC-10: Calibration 生成完整 validation 报告
- **Type**: `rule`
- **Given**: 60 个历款数据
- **When**: Calibration 流程完成
- **Then**: 生成 calibration_report.json，包含 baseline（内审分级）vs 校准后分类器的 AUC、Precision@K、Recall@K 对比，以及特征重要性排名
- **Pass Condition**: 报告包含上述所有字段，baseline 和校准后结果并列对比
- **Evidence**: calibration_report.json 文件 + Calibration 页 UI 展示

### AC-11: 旧 calibration（3 Loop Lasso）标记为 deprecated 不被调用
- **Type**: `rule`
- **Given**: 新 calibration 模块已上线
- **When**: 系统启动和运行
- **Then**: 旧的 `optimization_kernel.py` 中 3 Loop 相关代码不被任何模块 import 或调用
- **Pass Condition**: grep 检查 app.py 和 src/ 目录，无 3 Loop 相关 import
- **Evidence**: 代码静态分析

### AC-12: 所有现有 unit test 无回归
- **Type**: `rule`
- **Given**: 改动后的代码
- **When**: 运行 `python -m pytest tests/ -v`
- **Then**: 所有现有 test case 通过
- **Pass Condition**: pytest exit code = 0（或有明确标记为 deprecated 的 test case 被跳过）
- **Evidence**: pytest 运行日志

### AC-13: 用户能通过 UI 选择 calibration 产物并查看 validation 报告
- **Type**: `rubric`
- **Dimension**: Calibration UI 可用性
- **Scale**: 1-5
- **Anchors**: 1 = calibration 页无任何新内容；3 = 有 validation 数字但不可解释；5 = 有 baseline vs 校准后 AUC 对比图 + 特征重要性排名 + PatternMiner 规则可视化 + 一键加载 calibrated 产物
- **Pass Threshold**: >= 4
- **Evidence**: Calibration 页 UI 截图

### AC-14: 单款报告包含记忆匹配 + 爆款基因 + 爆款概率三模块
- **Type**: `rubric`
- **Dimension**: 单款报告可解释性
- **Scale**: 1-5
- **Anchors**: 1 = 无任何新增模块；3 = 有新增模块但展示不完整；5 = 三个模块完整展示，每个模块有清晰的可解释文案（如"邻居爆款率 60%，最像 MPEWQWT06(爆)"）
- **Pass Threshold**: >= 4
- **Evidence**: 单款报告 UI 截图

### AC-15: 分级总表支持三视角切换
- **Type**: `rubric`
- **Dimension**: 分级总表多角色适配
- **Scale**: 1-5
- **Anchors**: 1 = 无视角切换，所有人看同一张表；3 = 有切换但各视角差异不大；5 = 供应链/运营/品牌负责人三视角各自聚焦不同列（备货建议/投流建议/分级统计），爆款概率列始终在显著位置
- **Pass Threshold**: >= 4
- **Evidence**: 分级总表 UI 截图

## Open Questions

- [ ] P 款识别规则的具体特征阈值怎么从内审分级 P 款数据中学出来？（calibration 时分析内审 P 款的特征分布，取各特征的 P 分位数作为阈值）
- [ ] PatternMiner 在只有 4 个正样本时能挖出多少条有效规则？（可能不够，框架先搭好，后续积累更多样本后生效）
- [ ] 记忆库相似度的三个信号（neighbor_bomb_rate 等）如何进 LightGBM 分类器？（作为额外特征列，和其他特征一起训练）
- [ ] calibration 产物的向后兼容：旧版 calibration 目录下的 .yaml 文件会被新版自动忽略，需要明确的版本号标记。
