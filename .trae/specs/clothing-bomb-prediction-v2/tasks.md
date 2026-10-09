# 服装爆款 AI 预测系统 v2 — 实施计划

## Task 1: F11-F20 版型文本特征规则引擎
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 新建 `src/text_feature_extractor.py`，实现 10 个结构化特征的规则提取
  - 输入：版型/设计描述文本（Excel 列 E）
  - 输出：字典 `{'F11': float, 'F12': float, ..., 'F20': float}`
  - 规则引擎优先（关键词匹配 + 正则），提取失败时有 fallback 默认值 + 缺失标记
  - 支持"面料:XX / 版型:XX / 颜色:XX"三段式结构解析
- **Acceptance Criteria Addressed**: AC-7
- **Test Requirements**:
  - `rule` TR-1.1: 对 60 个历款的版型描述文本，全部成功输出 F11-F20，无 crash；Evidence: pytest 运行日志 + 手动脚本输出
  - `rule` TR-1.2: 空文本、格式异常文本、缺失列 → 输出默认值（F11-F20 各有合理默认）且标记 missing_flags，不抛异常；Evidence: unit test 边界 case
  - `rule` TR-1.3: F18 颜色数量能正确从"颜色：藏青/芥黄/粉色/紫色/蓝色/黄色"提取出 6；Evidence: unit test
  - `rubric` TR-1.4: Baseline E 复现（F11-F20 + 品类+季节+价格 → RandomForest AUC ≥ 0.78）；Scale 1-5; Anchors: 1 = AUC < 0.6; 3 = 0.6-0.78; 5 = ≥ 0.78; Threshold >= 4; Evidence: 运行 60 历款 baseline 脚本
- **Notes**: 已在 Phase 0 中用 inline 代码验证过效果，现在模块化。

## Task 2: Ground Truth 映射模块
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 新建 `src/ground_truth.py`
  - 实现 Excel → is_bomb 二分类标签映射：
    - 最终销量数值化（"9000+" → 9000）
    - 品类 top10% 阈值计算（或用户指定经验值 fallback）
    - `is_bomb = (销量 ≥ 品类阈值) AND (30天标签 ∈ {爆, 旺})`
  - 同时实现内审 P 款识别：`is_p_style = (内审分级 == 'P')`
  - 输出 pandas Series/DataFrame，包含 is_bomb, is_p_style, label_ordinal, sales_num 等字段
- **Acceptance Criteria Addressed**: AC-10
- **Test Requirements**:
  - `rule` TR-2.1: 60 个历款数据跑映射后，is_bomb.sum() = 4（MPEWQWT06, MPELQKZ20, MPEWXKZ10, MPEWDYR27）；Evidence: unit test assert
  - `rule` TR-2.2: 最终销量数值化能处理 "9000+"、"25"、"1,200+" 等混合格式；Evidence: unit test
  - `rule` TR-2.3: P 款识别能正确标记 16 个内审 P 款；Evidence: unit test assert

## Task 3: 新版 calibration 核心模块（替换 3 Loop）
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1, Task 2
- **Description**:
  - 新建 `src/calibration_v2.py`，实现四步 calibration 流程：
    1. 特征重要性评估（permutation importance，用 sklearn 的 `permutation_importance`）
    2. 爆款分类器训练（LightGBM 优先，Logistic Regression 作 fallback；5-fold stratified CV；优化指标 AUC-ROC + AUC-PR）
    3. P 款识别规则阈值学习（从内审 P 款数据取各特征分位数）
    4. Validation 报告生成（baseline vs 校准后对比）
  - calibration 产物保存到 `brand_profiles/{brand}/calibrated/` 目录，包含：calibration_report.json, classifier_model.pkl, feature_weights.json, p_rule.json
  - 对外暴露统一入口函数 `run_calibration(history_df, brand_name) -> CalibrationResult`
- **Acceptance Criteria Addressed**: AC-10, AC-11
- **Test Requirements**:
  - `rule` TR-3.1: calibration 跑完后生成 calibration_report.json，包含 baseline AUC、校准后 AUC、Precision@K、Recall@K、特征重要性排名；Evidence: 文件内容检查
  - `rule` TR-3.2: 5-fold stratified CV 每折至少有 1 个正样本（is_bomb），不出现某折全负导致 AUC 无法计算；Evidence: CV 中间日志
  - `rule` TR-3.3: calibration 产物目录结构正确（4 个文件都存在）；Evidence: os.path.isfile 检查
  - `rubric` TR-3.4: 校准后 AUC-ROC > Baseline E (0.78)；Scale 1-5; Anchors: 1 = ≤ 0.78; 3 = 0.79-0.84; 5 = ≥ 0.85; Threshold >= 4; Evidence: calibration_report.json

## Task 4: 品牌记忆库（Brand Memory DB）
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 3
- **Description**:
  - 新建 `src/memory_store.py`
  - 存储格式：parquet 文件，每条历款 = {style_id, 品类, 季节, 售价, 年份, VLM10特征, F11-F20, 人设加权分, is_bomb, is_p_style, sales_num}
  - 初始化时从 calibration 产物加载 feature_weights.json，用作相似度加权
  - 相似度搜索接口 `search_neighbors(query_style, top_k=5)`：
    - 加权欧氏距离（权重来自 calibration 特征重要性）
    - 同品类邻居权重 × 2，不同品类 × 0.5
    - 返回 top-5 邻居列表 + 三个记忆信号（neighbor_bomb_rate, neighbor_avg_sales, neighbor_category_consistency）
  - 增量更新接口 `update_memory(new_styles_df)`
- **Acceptance Criteria Addressed**: AC-5, AC-6, AC-7, AC-8
- **Test Requirements**:
  - `rule` TR-4.1: 60 个历款建库后，对任意一个款式搜索 top-5 邻居能返回 5 条（或记忆库不足时返回全部）；Evidence: unit test
  - `rule` TR-4.2: 记忆信号计算正确——neighbor_bomb_rate = 邻居中 is_bomb=True 的比例；Evidence: unit test assert
  - `rule` TR-4.3: 同品类邻居的相似度分数高于不同品类邻居（品类匹配加权生效）；Evidence: unit test 手动构造
  - `rubric` TR-4.4: 60 历款建库后，对 4 个真爆款搜索邻居，邻居爆款率平均 ≥ 40%（如果记忆库里同款自己是爆款，邻居也应该离爆款近）；Scale 1-5; Anchors: 1 = < 20%; 3 = 20-40%; 5 = ≥ 40%; Threshold >= 3; Evidence: 脚本运行输出

## Task 5: PatternMiner 重定义为结构化规则引擎
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: Task 3
- **Description**:
  - 改造现有 `src/pattern_miner.py`（保留文件名和基本框架，改输出格式）
  - 规则提取：从 LightGBM 模型的树结构（feature splits）中提取"爆款基因"规则
  - 规则格式：{rule_id, conditions: [(feature, op, threshold), ...], support, confidence}
  - 规则匹配引擎 `match_rules(new_style_features) -> rule_hits`：逐条匹配，输出命中规则列表（含置信度）
  - calibration 完成后自动提取规则，保存到 calibrated/pattern_rules.json
- **Acceptance Criteria Addressed**: AC-9, AC-16
- **Test Requirements**:
  - `rule` TR-5.1: PatternMiner 能从 LightGBM 模型中提取出至少 3 条规则（正样本只有 4 个时保底至少能从树 splits 中提）；Evidence: rule_hits 非空
  - `rule` TR-5.2: 规则匹配引擎对新款式能正确命中或不命中各规则，输出 rule_hits 数组；Evidence: unit test 手动构造
  - `rubric` TR-5.3: 规则质量——命中规则的平均 confidence ≥ 60%；Scale 1-5; Anchors: 1 = < 40%; 3 = 40-60%; 5 = ≥ 60%; Threshold >= 3; Evidence: pattern_rules.json 分析

## Task 6: PredictionPipeline 改造（集成所有新模块）
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1, Task 3, Task 4, Task 5
- **Description**:
  - 改造 `src/pipeline.py` 中的 `PredictionPipeline` 类：
    - 启动时检测 calibration 产物：如果有 → 加载 classifier、memory_store、pattern_miner、feature_weights；如果没有 → 降级到 Baseline E（品类+季节+价格+F11-F20 直接用 RandomForest/LogisticRegression）
    - 对每个款式的 prediction 流程：
      1. 调 VLM → 10 个视觉特征分（同时注入版型文本到 prompt）
      2. 调人设 → 30 人设投票（同时注入版型文本 + 记忆邻居爆款基因）
      3. 文本规则引擎 → F11-F20
      4. 记忆库搜索 → top-5 邻居 + 记忆信号
      5. PatternMiner 规则匹配 → rule_hits
      6. 通道打分（自然/直播）
      7. 分类器预测 → 爆款概率 + 置信区间（bootstrap）
      8. P 款识别
      9. 内审分级预测
    - 输出扩展后的 prediction result 对象
  - 标记旧 3 Loop 为 deprecated（`src/core/optimization_kernel.py` 保留文件但不再被调用，顶部加 deprecation 注释）
- **Acceptance Criteria Addressed**: AC-2, AC-3, AC-11, AC-18, AC-19, AC-20, AC-21, AC-22
- **Test Requirements**:
  - `rule` TR-6.1: 有 calibration 产物时加载 classifier，无产物时降级到 Baseline E（不 crash）；Evidence: 手动删除 calibrated/ 目录后跑 prediction
  - `rule` TR-6.2: prediction result 对象包含所有新字段：bomb_probability, bomb_probability_ci, memory_neighbors, memory_signals, rule_hits, is_p_style, predicted_internal_grade；Evidence: unit test assert
  - `rule` TR-6.3: 置信区间用 1000 次 bootstrap 计算（train on 90% sample, predict 10%），输出 (low, high) 元组；Evidence: 代码审查 + 数值合理性检查
  - `rubric` TR-6.4: 60 历款 5-fold CV 下，校准后 AUC-ROC > 0.78（Baseline E）；Scale 1-5; Anchors: 1 = ≤ 0.78; 3 = 0.79-0.84; 5 = ≥ 0.85; Threshold >= 4; Evidence: calibration_report.json

## Task 7: Grading 模块彻底重写
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 6
- **Description**:
  - 改造 `src/grading.py` 中的 `decide_grade` 函数和 `_grade_from_dual_dimension` 等辅助函数
  - 新分级逻辑：**完全由爆款概率（calibration 学出的阈值）推导**，不依赖批次内相对排名
  - P 款独立识别（不进概率分级）
  - 去掉旧的 Path A/B 单信号触发机制（OR 逻辑）
  - 分级标签：
    - 爆款概率 ≥ calib_S_threshold → "多备货 + 重点投流"
    - 爆款概率 ≥ calib_Aplus_threshold AND < S_threshold → "正常备货 + 常规运营"
    - 爆款概率 < calib_Aplus_threshold → "保守备货 + 观察"
    - P 款命中 → "设计展示款"（独立标记）
  - calibration 完成后阈值自动学习
- **Acceptance Criteria Addressed**: AC-4, AC-19, AC-20
- **Test Requirements**:
  - `rule` TR-7.1: 分级逻辑中不存在任何 rank/percentile/top_k 操作（grep 检查）；Evidence: 代码静态分析
  - `rule` TR-7.2: 构造 20 个"全烂"测试款（爆款概率全部 < 0.4），系统正确输出 0 个"多备货"款；Evidence: unit test
  - `rule` TR-7.3: 构造 20 个"全好"测试款（爆款概率全部 ≥ 0.7），系统正确输出 20 个"多备货"款（不限死比例）；Evidence: unit test
  - `rule` TR-7.4: P 款独立标记，有自己的分级标签"设计展示款"；Evidence: unit test

## Task 8: 新 calibration 入口 + Calibration 页 UI 改造
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 3, Task 6
- **Description**:
  - 在 `src/pipeline.py` 或单独模块暴露 `run_calibration_from_excel()` 入口函数
  - 改造 `app.py` Calibration 页：
    - 上传 Excel → 显示 Excel 解析结果（款数、品类分布、真爆款数）
    - 一键运行 calibration → 实时日志
    - 展示 calibration validation 报告：baseline vs 校准后 AUC 对比图、Precision@K/Recall@K 对比表、特征重要性排名条形图、PatternMiner 规则列表
    - "加载 calibration 产物"按钮（让 PredictionPipeline 使用校准后的分类器）
    - 旧 3 Loop 标记 deprecated，隐藏或灰掉
- **Acceptance Criteria Addressed**: AC-10, AC-13, AC-27
- **Test Requirements**:
  - `rule` TR-8.1: Calibration 页能成功运行 calibration 并生成产物；Evidence: 手动上传 Excel + 运行
  - `rule` TR-8.2: Validation 报告展示 baseline（内审分级）vs 校准后对比；Evidence: UI 截图
  - `rubric` TR-8.3: Calibration 页 UI 可用性；Scale 1-5; Anchors: 1 = 无新内容; 3 = 有数字但不可解释; 5 = 有对比图+条形图+规则可视化+一键加载; Threshold >= 4; Evidence: UI 截图

## Task 9: PredictionPipeline 入口 + 首页 UI 改造
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: Task 4, Task 6
- **Description**:
  - 改造 `app.py` 首页：
    - 新增"品牌记忆库"状态卡片：已积累 N 款、覆盖 X 品类、最近校准时间、校准后 AUC
    - 新增 calibration 状态指示器（有校准产物用 ✅，没有用 ⚠️ Baseline）
- **Acceptance Criteria Addressed**: AC-23
- **Test Requirements**:
  - `rule` TR-9.1: 首页显示记忆库状态和 calibration 状态；Evidence: UI 截图
  - `rubric` TR-9.2: 首页信息层次清晰；Scale 1-5; Anchors: 1 = 无新卡片; 5 = 状态卡片显著位置、信息一目了然; Threshold >= 4; Evidence: UI 截图

## Task 10: 分级总表重构（三视角 + 爆款概率）
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 7, Task 6
- **Description**:
  - 改造 `app.py` 分级总表页：
    - 去掉"Top 20% = S"硬编码排序逻辑，改为按爆款概率降序
    - 新增列：爆款概率（带进度条）、爆款概率 CI、P款标记、记忆邻居爆款率
    - 支持三视角切换（供应链/运营/品牌负责人），各自默认显示不同列
    - P 款单独高亮行背景色
- **Acceptance Criteria Addressed**: AC-24, AC-25, AC-26, AC-15
- **Test Requirements**:
  - `rule` TR-10.1: 排序改为按爆款概率降序（不是 final_score 或内审分级）；Evidence: 代码审查 + 手动验证
  - `rule` TR-10.2: 三视角切换能正常工作，各自默认列不同；Evidence: UI 交互测试
  - `rule` TR-10.3: 爆款概率列带进度条，P 款列有明确标记；Evidence: UI 截图
  - `rubric` TR-10.4: 分级总表多角色适配；Scale 1-5; Anchors: 1 = 无视角切换; 3 = 切换但差异不大; 5 = 三视角各聚焦不同列; Threshold >= 4; Evidence: UI 截图

## Task 11: 单款报告重构（记忆匹配 + 爆款基因 + 爆款概率 + 版型雷达）
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 6
- **Description**:
  - 改造 `src/report.py` + `app.py` 单款报告页：
    - 新增 🧠 **记忆匹配** 模块：top-5 邻居款式卡片（爆款标记 + 相似度分数）、邻居爆款率、解释文案
    - 新增 🧬 **爆款基因** 模块：命中的 PatternMiner 规则列表（规则条件 + 置信度）
    - 新增 📊 **爆款概率** 模块：0-100% 进度条 + 置信区间标注
    - 新增 👗 **版型基因雷达图**：F11-F20 十维雷达图
    - 新增 👁️ **内审分级预测**：系统预测的内审等级 + 与用户输入对比（如果有）
- **Acceptance Criteria Addressed**: AC-14, AC-24
- **Test Requirements**:
  - `rule` TR-11.1: 单款报告能展示所有 5 个新增模块（记忆/基因/概率/雷达/内审预测）；Evidence: UI 截图
  - `rule` TR-11.2: 邻居款式卡片显示 style_id、品类、爆款标记、相似度分数；Evidence: UI 内容检查
  - `rule` TR-11.3: 雷达图数据正确（F11-F20 值与文本规则引擎输出一致）；Evidence: 数值校验
  - `rubric` TR-11.4: 单款报告可解释性；Scale 1-5; Anchors: 1 = 无新增模块; 3 = 有但展示不全; 5 = 五模块完整 + 清晰文案; Threshold >= 4; Evidence: UI 截图

## Task 12: Prompt 增强（注入版型文本 + 记忆基因）
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: Task 1, Task 4, Task 5
- **Description**:
  - 改造 `src/feature_extraction.py`（VLM prompt 模板）和 `src/persona_voting.py`（人设 prompt 模板）：
    - VLM prompt 注入版型/设计描述文本，格式清晰分段
    - 人设 prompt 注入版型文本 + 记忆邻居爆款基因参考（"这款最像历款 MPEWQWT06(爆), 面料:XX, 版型:XX..."）
    - 保留旧模板作为 fallback（如果文本缺失）
- **Acceptance Criteria Addressed**: AC-29
- **Test Requirements**:
  - `rule` TR-12.1: 有版型文本时 VLM prompt 包含文本内容；Evidence: prompt dump 检查
  - `rule` TR-12.2: 有人设投票时人设 prompt 包含版型文本 + 记忆邻居基因（如果有记忆库）；Evidence: prompt dump 检查
  - `rule` TR-12.3: 文本缺失时 fallback 到旧模板不 crash；Evidence: unit test

## Task 13: 季末复盘页 + 内审分级预测
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: Task 6, Task 10
- **Description**:
  - 改造季末复盘页：上传实际销量后系统自动对比预测爆款清单 vs 实际爆款，输出 Recall@K/Precision@K/AUC 等年度汇报指标
  - PredictionPipeline 输出内审分级预测（S/A/P），calibration 时从内审分级历史数据中学习预测模型（可以是简单 ordinal regression 或 LightGBM 多分类）
- **Acceptance Criteria Addressed**: AC-3, AC-14
- **Test Requirements**:
  - `rule` TR-13.1: 季末复盘页能正确计算 Recall@K/Precision@K/AUC 对比指标；Evidence: 手动上传 60 历款 Excel 后运行
  - `rule` TR-13.2: 内审分级预测有输出（S/A/P 三选一）；Evidence: unit test

## Task 14: 全量回归测试 + 验证
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1-13 全部或大部分
- **Description**:
  - 运行完整 pytest 套件，确保所有现有 test case 继续通过
  - 手动端到端测试：上传 60 历款 Excel → 跑 calibration → 看 calibration validation 报告 → 上传新品 Excel → 跑 prediction → 看分级总表和单款报告
  - 验证 calibration 后 AUC > Baseline E
  - 验证 4 个真爆款都被识别出来
- **Acceptance Criteria Addressed**: AC-12
- **Test Requirements**:
  - `rule` TR-14.1: `python -m pytest tests/ -v` exit code = 0；Evidence: pytest 运行日志
  - `rule` TR-14.2: 端到端流程无 crash（calibration → prediction → UI 展示）；Evidence: 手动端到端运行
  - `rubric` TR-14.3: 端到端预测性能；Scale 1-5; Anchors: 1 = AUC ≤ 0.78, Top-10 Recall < 75%; 3 = AUC 0.79-0.84, Recall 75-80%; 5 = AUC ≥ 0.85, Recall ≥ 80%; Threshold >= 4; Evidence: calibration_report.json + 手动端到端验证
