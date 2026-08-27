# fashion-hit-engine | 服装爆款预测通用引擎

> 通用CORE预测引擎 + 品牌适配包 Brand Profile + 3Loop核心优化内核
> 跨童装/女装/男装/快消，换一套YAML即贴合新品类，销售数据喂进来→越跑越准

---

## 项目简介

fashion-hit-engine 是一款服装款式分级辅助工具。它的核心思路很简单：**模拟你品牌的典型消费者，让他面对这个款的图片和价格，判断"我会不会买"。**

你上传款式图片、FAB说明和售价，AI 会调出30个根据你品牌客群画像生成的"虚拟消费者"——每个人都有明确的购物场景、审美偏好和价格敏感度。他们逐款看图、读信息、给出购买意愿分数和不买的理由。30个人投完票，系统综合双渠道适配性和价格价值，输出一个S/A/P分级建议，并且告诉你每个款好在哪、差在哪、怎么改。

**它不是一次性工具——会越用越准。** 每季销售结束后，把真实销量数据喂回去，系统会自动校准三件事：哪个视觉特征的判断和实际卖得更好更相关、你品牌的真实人群分布和预设的30人设有多少偏差、三大评分引擎各自该占多少话语权。每一步调完如果精度下降就自动回滚，永远不会越调越差。你不需要懂代码、不需要请数据科学家，每季传一份销售Excel就行。

---

## 核心架构

```
消费者三层决策模型 → CORE引擎模拟(视觉观感/价值评估/身份表达)
                         ↓
           3Loop优化内核（VLM校准↔人设分布拟合↔集成权重调优）
                         ↓
                    S/A+/A/P分级报告
                         ↓
              残差分离器（不可预知因素识别）
```

---

## 3大关键能力

### 1. 通用CORE引擎 100%跨品牌复用

三大异构引擎并行，错误来源互相独立，跨品牌零代码复用：
- **引擎一·身份表达**：人设投票模拟真实消费者购买决策
- **引擎二·视觉观感**：System1快速视觉判断（BARS量表10特征结构化输出）
- **引擎三·价值评估**：System2理性价值评估（感知价值vs价格百分位匹配）

### 2. 品牌适配包 Brand Profile

5个YAML文件完成品牌适配，换品牌=换目录：
- **身份三轴线**：轴线A场景 × 轴线B审美偏好 × 轴线C价格敏感度 → 5×3×2=30人设
- **BARS量表**：10特征锚定描述，可按品牌调性微调
- **品类价格带**：各品类历史价格分布百分位
- **分级阈值**：S/A+/A/P各档门槛按品牌策略自定义

### 3. 3Loop优化内核

每轮销售数据自动校准，Spearman每轮单调提升（有保护机制，永不越调越差）：
- Loop1：校准VLM特征偏置
- Loop2：Lasso稀疏拟合品牌真实人群分布
- Loop3：调三大引擎+双渠道相对权重

---

## 品牌适配包目录

```
brand_profiles/
  ├─ tongzhuang-outdoor/     ← 现有：潮童户外6-14岁，双层决策结构
  │   ├─ profile.yaml              决策结构（single_layer / double_layer）
  │   ├─ features_bars.yaml        10特征BARS锚定量表
  │   ├─ personas.yaml             30人设 + 身份三轴线定义
  │   ├─ scoring_weights.yaml      评分公式 + 权重 + S/A/P阈值
  │   ├─ category_registry.yaml    品类 + 价格带 + 别名
  │   └─ calibrated/               3Loop校准产物（自动生成）
  │
  └─ _template/              ← 女装通勤模板，single_layer，复制改名字即新品类
      ├─ profile.yaml
      ├─ features_bars.yaml
      ├─ personas.yaml
      ├─ scoring_weights.yaml
      ├─ category_registry.yaml
      └─ calibrated/
```

---

## 快速启动（30秒）

```bash
# ① 装依赖（首次）
pip install -r requirements.txt

# ② 离线冒烟测试（不需要API Key，验证核心模块OK）
python quick_smoke_test.py
#   预期输出：
#   ✅ 10款模拟数据跑通
#   Spearman(分级) ≥ 0.80

# ③ 填百炼API Key（首次跑真实款式）
copy .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY=
# Key申请：https://bailian.console.aliyun.com/

# ④ 启动Web UI
streamlit run app.py
# 浏览器自动打开 http://localhost:8501
```

---

## 4个页面说明

### 页面1 · 📤 上传批次
选品牌适配包 → 传Excel（款式+FAB+售价）+ 图片文件夹（按款号建目录）→ 自动校验必传列 → 点「开始评估」→ 自动跳转页面2。

### 页面2 · 📋 批次总表
S/A+/A/P分级饼图 + Spearman相关性预览（有历史数据时）。每款一行：分级/综合分/自然分/直播分/价值匹配/价格风险/主推渠道。筛选+排序+导出Excel+打包下载Markdown报告。点行跳转页面3。

### 页面3 · 🔍 单款详情
左栏图片轮播（正面/背面/look/细节）。右栏：改款建议（2-5条可执行）+ 三大引擎贡献拆解柱状图 + 10特征BARS雷达图 + 人设投票分布 + 双渠道评分明细。底部下载Markdown单款报告。

### 页面4 · 📊 回测校准
喂真实销量Excel → 点「运行3Loop校准」→ 显示Loop1/2/3各自Spearman前后对比（绿色提升/灰色持平未用/红色倒退拦截）+ 校准报告（超预期款ID列表 + 不及预期款ID列表，供运营复盘，**这些残差款不喂进校准**）。校准通过后新权重写入品牌包的 `calibrated/` 目录。

---

## 3Loop优化内核机制

**Loop1 = 校准VLM特征偏置**：统计每个VLM特征f与真实销量的Spearman ρ_f，除以平均ρ得到偏置系数，clamp到 [0.7, 1.3] 区间，防止单个特征被过度放大。

**Loop2 = Lasso稀疏拟合品牌真实人群分布**：用L1正则化回归从真实销量反推30人设的真实权重分布，稀疏性保证只有真正有区分力的人设被调高，避免过拟合。

**Loop3 = 调三大引擎+双渠道相对权重**：权重 ∝ max(0.05, ρ + 0.3) 归一化，保证每个引擎最少5%话语权，防止某一引擎被完全压制。

**保护机制**：每步Loop更新后必须满足「新Spearman ≥ 旧Spearman + MIN_IMPROVEMENT」，否则回滚返回旧值，确保Spearman单调不减，永不越调越差。

---

## 残差分离（不可预知因素怎么办）

残差 ε = y真实 - ŷ预测，按正态分布统计：
- **ε > μ + 2σ**：超预期款（卖得比AI预测好很多）→ 单独标红，供运营复盘找不可预知因素（KOL带火/热搜同款/竞品缺货…）
- **ε < μ - 2σ**：不及预期款（卖得比AI预测差很多）→ 单独标红，复盘外生负面因素（质量事故/差评扩散/竞品大促…）

**绝对不能把残差喂回3Loop**，否则会学到伪相关（比如误以为某颜色特征导致爆款，其实是KOL碰巧穿了）。残差是运营复盘素材，不是校准训练数据。

---

## 目录结构（重点）

```
pack/
├─ app.py                          ← Web UI入口（streamlit run）
├─ quick_smoke_test.py             ← 离线冒烟测试
├─ requirements.txt
├─ spec.md                         ← 方法论 + 数学规格
├─ docs/web-ui-design.md           ← UI设计文档
│
├─ src/
│  ├─ core/
│  │   ├─ optimization_kernel.py   ← 3Loop优化内核核心
│  │   └─ ensemble_engine.py       ← 三大引擎集成聚合
│  ├─ feature_extraction.py        ← 引擎二·视觉观感（VLM+BARS）
│  ├─ persona_voting.py            ← 引擎一·身份表达（30人设投票）
│  ├─ channel_scoring.py           ← 双渠道评分（自然/直播）
│  ├─ calibration.py               ← 残差分离 + 保护机制
│  ├─ grading.py                   ← S/A+/A/P分级
│  ├─ pipeline.py / report.py / ...
│
└─ brand_profiles/
   ├─ tongzhuang-outdoor/          ← 童装户外品牌包（现有）
   └─ _template/                   ← 女装通勤模板（复制即用）
      ├─ profile.yaml              决策结构
      ├─ features_bars.yaml        BARS量表
      ├─ personas.yaml             30人设
      ├─ scoring_weights.yaml      公式+阈值
      ├─ category_registry.yaml    品类价格带
      └─ calibrated/               校准产物
```

---

## LICENSE & FAQ

**开源协议**：MIT License，商业使用需联系作者授权品牌适配包定制服务。

**常见问题**：
- **Q：换一个新品类需要改代码吗？** A：不用，复制 `brand_profiles/_template/` 改5个YAML即可，Python代码一行不动。
- **Q：销售数据需要多细？** A：只要款号+真实销量（件数/排名都行），系统自动转品类内百分位标签。
- **Q：会不会越调越差？** A：3Loop每步有Spearman提升保护，不达标就回滚，Spearman单调递增。
- **Q：残差款为什么不喂回校准？** A：防止学到伪相关（KOL带火≠颜色特征好），残差是运营复盘素材。
