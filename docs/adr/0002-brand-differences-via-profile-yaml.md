# ADR-0002: 品牌差异走「适配包 YAML」而非代码分支

## 状态
已接受

## 背景
同一套预测引擎（特征提取→人设投票→渠道评分→分级）要服务多个品牌
（MIPO 童装户外、womenswear 女装等）。品牌间差异巨大：客群结构、
人设分布、价格带、品类树、决策层级都不同。

## 决策
所有品牌差异收敛到 `brand_profiles/<brand_id>/` 下的 YAML 适配包
（profile / personas / features_bars / category_registry / scoring_weights），
核心引擎代码保持品牌无关。新品牌以 `mipo/` 适配包为起手模板复制改造
（`_template/` 与 `tongzhuang-outdoor/` 已删除，系统当前只服务 mipo）。

## 备选方案
- 每品牌一个代码分支：维护成本随品牌数线性爆炸，未选。
- 单一配置文件内 if-else：品牌差异是结构性的（层数、轴数都不同），
  不是参数级的，未选。
- 数据库存品牌配置：无 DB 部署场景（本地 Streamlit 优先），未选。

## 后果
- 正面：新品牌上线 = 复制模板 + 填 YAML，零代码改动；适配包可整体
  进 git 做版本评审；校准产物也落在适配包内，随品牌隔离。
- 负面：① YAML 结构成为公共契约，改锚定 schema 会影响所有品牌，
  必须同步改所有适配包；② 历史默认 brand_id 的兜底调用点
  （已全部收敛为 mipo，后续应逐步清理为必传参数）。
- 遗留已解决（v1.3.0）：persona 投票数据结构硬编码「妈妈/孩子」的问题
  已重构为通用决策层——`PersonaVote` 按 layer_id 存分数/理由
  （layer_scores/layer_reasons），投票 prompt 按层动态渲染，否决层轴
  扫描按 layer.persona_axis_key 取数，age_weight_rules 支持任意层数的
  layer_weights 直配。见术语表「决策层」与
  tests/test_generic_decision_layers.py（三层女装结构验证）。
