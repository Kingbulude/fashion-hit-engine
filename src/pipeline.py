"""流水线主入口：读取款式 → 特征提取 → 人设投票 → 双渠道评分 → 分级 → 报告

支持两种运行模式：
  # 1. 回测模式（默认，有真实销量/分级对比）
  python -m src.pipeline backtest --styles data/styles.xlsx --images data/images

  # 2. 预测模式（只有款式信息，无历史数据对比）
  python -m src.pipeline predict --styles data/styles_new.xlsx --images data/images

  # 3. 训练模式（对一批跑完回测的样本拟合校准权重）
  python -m src.pipeline train

v2.0：新增 PredictionPipeline 类，通过 BrandConfig 注入全部模块。
"""
from __future__ import annotations

import argparse
import logging
import sys
import zlib
from pathlib import Path

from .channel_scoring import calculate_channel_scores
from .config import AppConfig, load_brand_profile, load_config
from .core.ensemble_engine import synthesise_final_score
from .data_io import load_styles_from_excel, read_styles_excel, save_predictions_xlsx
from .feature_extraction import (
    FeatureExtractionEngine,
    extract_style_features,
)
from .grading import assign_relative_grades, decide_grade
from .llm_client import BailianClient, ZhipuClient, is_fatal_quota_error
from .persona_voting import run_persona_voting
from .report import generate_backtest_summary, generate_markdown_report, generate_report
from .types import (
    BrandConfig,
    ChannelScores,
    FullPrediction,
    GradeResult,
    StyleFeatures,
    StyleInfo,
    VotingResult,
    clamp,
)


log = logging.getLogger("pipeline")


# ========== 旧兼容函数（保持100%原签名）==========
def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)5s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_single(client, info, cfg, all_style_prices, calibrated_weights=None, progress=False):
    feats = extract_style_features(client, info, cfg, progress=progress)
    voting = run_persona_voting(client, info, feats, cfg, progress=progress)
    channels, _ = calculate_channel_scores(info, feats, voting, cfg, all_style_prices=all_style_prices)
    grade = decide_grade(info, feats, voting, channels, cfg, calibrated_weights=calibrated_weights)
    return FullPrediction(info=info, features=feats, voting=voting, channels=channels, grade=grade)


def run_batch(cfg, styles_path, images_dir, mode, out_dir, brand_id="mipo"):
    styles = read_styles_excel(styles_path, images_dir)
    if not styles:
        log.error("没有读取到任何款式，请检查 %s", styles_path)
        sys.exit(1)

    has_images = [s for s in styles if s.images]
    if not has_images:
        log.error("所有款式都未匹配到图片。请把款号命名的图片放到 %s，如 images/T251001.jpg", images_dir)
        sys.exit(1)
    if len(has_images) < len(styles):
        log.warning("%d 款中有 %d 款未匹配到图片，将跳过", len(styles), len(styles) - len(has_images))
    styles = has_images

    log.info("载入 %d 款，启动预测流水线...", len(styles))

    calib_path = out_dir / "calibration" / "feature_weights.yaml"
    calibrated_weights = None
    if calib_path.exists():
        try:
            import yaml
            d = yaml.safe_load(calib_path.read_text(encoding="utf-8"))
            calibrated_weights = d.get("feature_weights")
            log.info("已载入训练后的校准权重（R²≈%.3f）", d.get("meta", {}).get("r2_in_sample", 0.0))
        except Exception as e:
            log.warning("载入校准权重失败，将使用默认权重：%s", e)

    client = BailianClient(cfg.api)
    all_prices = [s.price for s in styles if s.price > 0]

    predictions: list[FullPrediction] = []
    for i, s in enumerate(styles, 1):
        log.info("=== [%d/%d] 款号 %s 开始处理 ===", i, len(styles), s.style_id)
        try:
            pred = run_single(
                client, s, cfg,
                all_style_prices=all_prices,
                calibrated_weights=calibrated_weights,
                progress=True,
            )
            # 本款 API 用量快照（take 会清零，按款切分）
            pred.metadata["api_usage"] = client.usage_tracker.take()
            predictions.append(pred)
        except Exception as e:
            log.exception("  ❌ 款%s处理失败: %s", s.style_id, e)
            # 额度耗尽/鉴权失效：继续跑剩余款只会反复失败，立即止损
            if is_fatal_quota_error(str(e)):
                log.error(
                    "⛔ 检测到 API 额度不足/鉴权失败，批次中止（剩余 %d 款跳过）。"
                    "请到百炼控制台充值后重跑。",
                    len(styles) - i,
                )
                break

    # 批次内相对分级：冷启动时绝对阈值失效（VLM 绝对分尺度未校准），
    # 但 VLM 的排序可靠 → 按批次内分位重定 S/A+/A/P，与人工内审语义对齐。
    # 幂等：绝对档快照存 metadata['absolute_grade']，重复调用无副作用。
    assign_relative_grades(predictions)

    # 报告生成放在相对分级之后，确保单款报告展示的是最终相对档位
    for pred in predictions:
        report_path = generate_report(pred, out_dir / "reports")
        log.info(
            "  ✅ 完成 → 分级=%s 最终分=%.1f 置信度=%.0%%  报告=%s",
            pred.grade.grade, pred.grade.final_score,
            pred.grade.confidence, report_path.name,
        )

    # 本批 API 用量汇总（调用次数 / token，回答「钱花哪了」）
    if predictions:
        calls = sum(int(p.metadata.get("api_usage", {}).get("calls", 0)) for p in predictions)
        in_tok = sum(int(p.metadata.get("api_usage", {}).get("input_tokens", 0)) for p in predictions)
        out_tok = sum(int(p.metadata.get("api_usage", {}).get("output_tokens", 0)) for p in predictions)
        log.info("💰 本批 API 用量：成功 %d 款 · %d 次调用 · 输入 %s tokens · 输出 %s tokens",
                 len(predictions), calls, f"{in_tok:,}", f"{out_tok:,}")

    xlsx_path = save_predictions_xlsx(predictions, out_dir / f"{mode}_summary_{len(predictions)}款.xlsx")
    log.info("✅ 批量总表已导出 → %s", xlsx_path)

    if mode == "backtest":
        from statistics import mean

        def _rank(lst):
            indexed = sorted(enumerate(lst), key=lambda x: x[1])
            ranks = [0] * len(lst)
            i = 0
            while i < len(indexed):
                j = i
                while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                    j += 1
                avg_rank = (i + 1 + j + 1) / 2
                for k in range(i, j + 1):
                    ranks[indexed[k][0]] = avg_rank
                i = j + 1
            return ranks

        def _spearman(a, b):
            if len(a) < 2 or len(a) != len(b):
                return None
            ra = _rank(a)
            rb = _rank(b)
            n = len(a)
            ma, mb = mean(ra), mean(rb)
            num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
            den_a = sum((ra[i] - ma) ** 2 for i in range(n))
            den_b = sum((rb[i] - mb) ** 2 for i in range(n))
            if den_a == 0 or den_b == 0:
                return 0.0
            return num / (den_a ** 0.5 * den_b ** 0.5)

        grade_map = {"S": 4, "A+": 3, "A": 2, "P": 1}
        preds_for_grade = [p for p in predictions if p.info.manual_grade in grade_map]
        preds_for_sales = [p for p in predictions if p.info.sales_qty > 0]
        sp_grade: float | None = None
        sp_sales: float | None = None
        acc_s: float | None = None
        if len(preds_for_grade) >= 3:
            pred_vals = [grade_map.get(p.grade.grade, 0) for p in preds_for_grade]
            true_vals = [grade_map[p.info.manual_grade] for p in preds_for_grade]
            sp_grade = _spearman(pred_vals, true_vals)
            s_correct = sum(1 for p in preds_for_grade if
                            (p.grade.grade == "S") == (p.info.manual_grade == "S"))
            acc_s = s_correct / len(preds_for_grade)
        if len(preds_for_sales) >= 3:
            pred_vals = [p.grade.final_score for p in preds_for_sales]
            true_vals = [p.info.sales_qty for p in preds_for_sales]
            sp_sales = _spearman(pred_vals, true_vals)

        log.info("================ 回测验证报告 ================")
        if sp_grade is not None:
            log.info("预测 vs 人工S/A/P分级 Spearman = %+.3f  (样本=%d)", sp_grade, len(preds_for_grade))
        else:
            log.info("⚠️ 人工分级样本<3，无法计算Spearman分级相关")
        if sp_sales is not None:
            log.info("预测 vs 真实销量 Spearman = %+.3f  (样本=%d)", sp_sales, len(preds_for_sales))
        else:
            log.info("⚠️ 销量数据不足3个，无法计算Spearman销量相关")
        if acc_s is not None:
            log.info("S款识别准确率（S vs 非S 二分类） = %.0%%", acc_s * 100)

        summary_path = generate_backtest_summary(
            predictions,
            spearman_grade=sp_grade or 0.0,
            spearman_sales=sp_sales or 0.0,
            accuracy_s_vs_not=acc_s or 0.0,
            out_path=out_dir / "backtest_summary.md",
        )
        log.info("回测详情报告 → %s", summary_path)

        # v1 train_calibration 已移除：校准统一由下方 v2 run_all_loops 3Loop 内核完成
        # （产物写 brand_cfg.calibrated_dir，下次评估自动加载）

        # ---- v2 过渡：额外跑 3Loop 优化内核（spec §5/§8/§9）----
        # 产物写到 brand_cfg.calibrated_dir（brand_profiles/<brand>/calibrated/），
        # 与 v1 旧路径 out_dir/calibration/feature_weights.yaml 物理隔离，
        # 不破坏 run_single 加载旧权重的逻辑。
        # 下次 v2 PredictionPipeline.run_one 会自动加载 3Loop 权重。
        try:
            from .config import load_brand_profile
            from .core.optimization_kernel import build_history_df, run_all_loops
            brand_cfg = load_brand_profile(brand_id)
            sales_lookup = {
                p.info.style_id: float(p.info.sales_qty)
                for p in predictions
                if p.info.sales_qty and p.info.sales_qty > 0
            }
            history_df = build_history_df(predictions, sales_lookup=sales_lookup)
            sales_col = "sales"
            if sales_col not in history_df.columns or history_df[sales_col].sum() == 0:
                log.warning(
                    "⚠️ v2 3Loop 跳过：predictions 全无销量（info.sales_qty=0），"
                    "无回归信号。请在 styles Excel 填销量列或用 sales_lookup 注入。"
                )
            else:
                artifacts_dir = Path(brand_cfg.calibrated_dir)
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                v2_result = run_all_loops(
                    brand_cfg=brand_cfg,
                    history_df=history_df,
                    prediction_artifacts_dir=artifacts_dir,
                    sales_col=sales_col,
                )
                log.info(
                    "✅ v2 3Loop 校准完成：写入 %d 个产物 → %s",
                    len(v2_result.output_files), artifacts_dir,
                )
                log.info(
                    "   Loop1 applied=%s (ρ %.3f→%.3f), Loop2 applied=%s (ρ %.3f→%.3f), "
                    "Loop3 applied=%s (ρ %.3f→%.3f)",
                    v2_result.loop1.applied,
                    v2_result.loop1.old_spearman_avg, v2_result.loop1.new_spearman_avg,
                    v2_result.loop2.applied,
                    v2_result.loop2.old_spearman, v2_result.loop2.new_spearman,
                    v2_result.loop3.applied,
                    v2_result.loop3.old_engine_spearman, v2_result.loop3.new_engine_spearman,
                )
                log.info(
                    "   残差 μ=%.4f σ=%.4f, over=%d, under=%d, bias=%s",
                    v2_result.residual.residual_mean, v2_result.residual.residual_std,
                    len(v2_result.residual.overperformers),
                    len(v2_result.residual.underperformers),
                    v2_result.residual.system_bias_flag,
                )
        except Exception as e:
            log.warning("v2 3Loop 校准失败（不影响 v1 主流程）：%s", e)

    return predictions


# ========== v2.0：PredictionPipeline（BrandConfig 注入）==========
class PredictionPipeline:
    """爆款预测主流水线 v2.0 —— 基于 BrandConfig 全链路注入。

    Usage:
        pipe = PredictionPipeline(brand_id="mipo", llm_backend="mock")
        results = pipe.run_smoke_test_data(n=10)
        for r in results:
            print(r.style_id, r.grade.grade, r.grade.final_score)
    """

    def __init__(
        self,
        brand_id: str = "mipo",
        llm_backend: str = "dashscope",
        api_key: str | None = None,
    ) -> None:
        self.brand_id = brand_id
        # llm_backend: "dashscope"(默认真实百炼) | "mock"(本地随机调试) | "volc"(火山)
        self.llm_backend = llm_backend
        self.api_key = api_key
        self.brand_cfg: BrandConfig = load_brand_profile(brand_id)
        self.feature_engine = FeatureExtractionEngine(
            brand_cfg=self.brand_cfg, llm_backend=llm_backend,
        )
        self._client: BailianClient | ZhipuClient | None = None

        # ===== 加载 3Loop 校准产物（让校准权重真正生效）=====
        from .calibration_loader import load_calibration
        self.calibration = load_calibration(self.brand_cfg.calibrated_dir)
        # 把校准权重注入 brand_cfg（synthesise_final / _default_aggregate 会读）
        if self.calibration.engine_weights:
            self.brand_cfg.engine_weights = self.calibration.engine_weights
        # channel_split 用独立属性避免覆盖默认值
        if self.calibration.channel_split:
            self.brand_cfg._calibrated_channel_split = self.calibration.channel_split
        if self.calibration.persona_weights:
            self.brand_cfg.personas_weights = self.calibration.persona_weights
        # feature_biases 存起来，run_one 时乘到特征分上
        if self.calibration.feature_biases:
            self.brand_cfg._feature_biases = self.calibration.feature_biases

        # ===== 历史批次持久化：跨批次累积价格/特征/分数分布 =====
        from .history_store import HistoryStore
        self.history = HistoryStore()
        # ===== 缓存 AppConfig.api（persona_models / feature_extraction_models 等）=====
        # brand_cfg.personas 是 list，persona_models 从 AppConfig.api 取
        try:
            from .config import load_config
            _api_cfg = load_config(override_api_key=api_key).api
            self._api_persona_models: list[str] = list(_api_cfg.persona_models)
            self._api_feature_models: list[str] = list(_api_cfg.feature_extraction_models)
        except Exception:
            self._api_persona_models = ["qwen-max", "deepseek-v3"]
            self._api_feature_models = ["qwen-vl-plus"]
        self._history_prices: list[float] = self.history.load_prices(brand_id)
        log.info("Pipeline init 完成: brand=%s, llm=%s, calibration=%s, history_prices=%d",
                 brand_id, llm_backend, self.calibration, len(self._history_prices))

    @property
    def client(self) -> BailianClient | ZhipuClient | OllamaClient:
        """主 client（VLM 特征提取用）

        hybrid 模式 → Zhipu 云端 VLM（专业图像理解是最大短板，必须用云端）
        local  模式 → Ollama 本地 VLM（零成本）
        zhipu 模式 → Zhipu 全云端
        """
        if self._client is None:
            if self.llm_backend == "local":
                from .llm_client import OllamaClient
                self._client = OllamaClient()
            else:
                try:
                    api_cfg = load_config(override_api_key=self.api_key).api
                except Exception:
                    from .config import APIConfig
                    api_cfg = APIConfig(dashscope_api_key=self.api_key or "")
                if self.llm_backend in ("hybrid", "zhipu"):
                    # hybrid 和 zhipu 的 VLM 都走智谱云端（glm-4.6v-flash 免费且专业）
                    api_cfg.zhipu_api_key = self.api_key or api_cfg.zhipu_api_key
                    self._client = ZhipuClient(api_cfg)
                else:
                    self._client = BailianClient(api_cfg)
        return self._client

    @property
    def text_client(self) -> BailianClient | ZhipuClient | OllamaClient:
        """文本 client（人设投票/渠道分用）

        hybrid → Ollama 本地（文本理解 7B 够用，零成本，不占智谱配额）
        local  → Ollama 本地
        其他   → 同 self.client
        """
        if self.llm_backend not in ("hybrid",):
            return self.client  # type: ignore[return-value]
        # hybrid 模式：文本走本地 Ollama
        if getattr(self, "_text_client", None) is None:
            from .llm_client import OllamaClient
            self._text_client = OllamaClient()
        return self._text_client

    # ===== 三大引擎合成（调用 ensemble_engine）=====
    def synthesise_final(
        self,
        voting: VotingResult,
        channels: ChannelScores,
    ) -> tuple[float, dict[str, float]]:
        persona_score = clamp(voting.weighted_score, 0.0, 10.0)
        channel_scores = {
            "natural": channels.natural_score,
            "live_stream": channels.live_score,
        }
        pv_norm = (channels.value_match + 1.0) * 5.0
        # engine_weights: 默认 → 校准覆盖
        engine_weights = dict(self.brand_cfg.default_engine_weights or {})
        if getattr(self.brand_cfg, "engine_weights", None):
            engine_weights = {**engine_weights, **self.brand_cfg.engine_weights}
        # channel_split: 默认 0.5/0.5 → 校准覆盖（Loop3 产物）
        channel_split = dict(self.brand_cfg.default_channel_split or {"natural": 0.5, "live_stream": 0.5})
        if getattr(self.brand_cfg, "_calibrated_channel_split", None):
            channel_split = {**channel_split, **self.brand_cfg._calibrated_channel_split}
        return synthesise_final_score(
            persona_score,
            channel_scores,
            pv_norm,
            engine_weights=engine_weights,
            channel_split=channel_split,
        )

    # ===== mock 人设投票（不依赖LLM，用于smoke test）=====
    def _mock_voting(self, style_id: str, feats: StyleFeatures) -> VotingResult:
        import random
        # 注：原 hash() 跨进程随机化（PYTHONHASHSEED）→ mock 投票每进程不同 →
        # smoke test 非确定性（sp_sales 在 0.90~0.99 漂移、阈值 0.92 随机失败）。
        # 改用 zlib.crc32 提供确定性 hash。
        random.seed(zlib.crc32(f"{style_id}_vote".encode()))
        target_age = int(getattr(self.brand_cfg.decision_structure, "default_target_age", 10))
        n_personas = 30
        votes = []
        all_scores: list[float] = []
        support = 0
        oppose = 0
        buy_reasons_map: dict[str, int] = {}
        oppose_reasons_map: dict[str, int] = {}
        reasons_pool_buy = [
            "面料看起来很舒服，功能设计也实用",
            "版型好看，孩子喜欢颜色和廓形",
            "场景百搭，上学户外都能穿",
            "品牌调性符合我们的审美",
            "设计独特不容易撞款",
        ]
        reasons_pool_oppose = [
            "颜色太艳/太大胆，不敢挑战",
            "感觉搭配难度有点高",
            "版型过于宽松，日常上学不太合适",
            "面料看起来偏薄，担心质量",
            "设计有点夸张，不够实穿",
        ]
        for i in range(n_personas):
            base = random.uniform(4.0, 8.0)
            avg_feat = sum(f.score for f in feats.features.values()) / max(1, len(feats.features))
            s = clamp(base * 0.5 + avg_feat * 0.5, 1.0, 10.0)
            # 各决策层独立扰动：决策者层±0.8，影响层±1.2（与层语义无关的数值扰动）
            layer_scores: dict[str, float] = {}
            for layer in self.brand_cfg.decision_structure.layers:
                jitter = 0.8 if layer.role == "decider" else 1.2
                layer_scores[layer.id] = round(clamp(s + random.uniform(-jitter, jitter), 1.0, 10.0), 1)
            all_scores.append(s)
            if s >= 7.0:
                support += 1
                r = random.choice(reasons_pool_buy)
                buy_reasons_map[r] = buy_reasons_map.get(r, 0) + 1
            elif s < 4.0:
                oppose += 1
                r = random.choice(reasons_pool_oppose)
                oppose_reasons_map[r] = oppose_reasons_map.get(r, 0) + 1
            from .types import PersonaVote
            votes.append(PersonaVote(
                persona_id=f"P{i+1:02d}",
                persona_name=f"人设{i+1}",
                layer_scores=layer_scores,
                final_score=round(s, 1),
            ))
        from statistics import pstdev
        top_buy = [r for r, _ in sorted(buy_reasons_map.items(), key=lambda x: -x[1])[:3]]
        top_opp = [r for r, _ in sorted(oppose_reasons_map.items(), key=lambda x: -x[1])[:3]]
        return VotingResult(
            style_id=feats.style_id,
            votes=votes,
            weighted_score=round(sum(all_scores) / len(all_scores), 2),
            opposition_rate=oppose / len(all_scores),
            support_rate=support / len(all_scores),
            top_buy_reasons=top_buy,
            top_oppose_reasons=top_opp,
            score_std=round(pstdev(all_scores), 2) if len(all_scores) >= 2 else 0.0,
        )

    # ===== 单款运行（mock或真实，取决于 llm_backend）=====
    def run_one(
        self,
        info: StyleInfo,
        *,
        use_mock: bool | None = None,
        fixed_feature_scores: list[float] | None = None,
        image_paths_map: dict[str, list[str]] | None = None,
        all_style_prices: list[float] | None = None,
    ) -> FullPrediction:
        if use_mock is None:
            use_mock = (self.llm_backend == "mock")

        # 跨批次累积价格分布：历史 + 当前批次
        price_pool: list[float] = list(self._history_prices)
        if all_style_prices:
            price_pool.extend(all_style_prices)
        # 去重后保持合理规模（最新 2000 条足够做百分位）
        price_pool = sorted(set(p for p in price_pool if p > 0))[-2000:]
        # 当前款自身价格必须在里面，否则后续 percentile 可能为空
        if info.price > 0 and info.price not in price_pool:
            price_pool.append(info.price)
            price_pool.sort()

        # 把 image_paths_map 注入 info.images（真实VLM路径需要读取图片）
        if image_paths_map and info.style_id in image_paths_map and not info.images:
            from pathlib import Path
            info.images = [Path(p) for p in image_paths_map[info.style_id]]
        # 🔍 VLM 关键诊断日志：确认图片是否真的传入
        if info.images:
            _imgs = [str(p) for p in info.images]
            log.info("🎨 [%s] VLM 图片已注入: %d 张 (%s...)", info.style_id, len(_imgs), _imgs[0][-40:] if _imgs else "")
        else:
            log.warning("⚠️  [%s] VLM 图片为空! 检查 image_paths_map 是否有此款号 → VLM 会无图瞎猜", info.style_id)
        if use_mock:
            feats = self.feature_engine.extract_mock(
                info.style_id, fixed_feature_scores=fixed_feature_scores
            )
            # ===== 应用 Loop1 feature_biases（校准偏置）=====
            biases = getattr(self.brand_cfg, "_feature_biases", None)
            if biases:
                # biases key 是短名 "F01"，feats key 是 "F01_silhouette" → 前缀匹配
                for feat_id, feat in feats.features.items():
                    bias = biases.get(feat_id) or biases.get(feat_id[:3], 1.0)
                    if abs(bias - 1.0) > 0.001:
                        feat.score = round(
                            clamp(feat.score * bias, 0.0, 10.0), 3
                        )
            voting = self._mock_voting(info.style_id, feats)
            channels, _debug = calculate_channel_scores(
                info, feats, voting,
                cfg=None, all_style_prices=price_pool, brand_cfg=self.brand_cfg,
                category_id=info.category or None,
            )
            grade = decide_grade(
                info, feats, voting, channels,
                cfg=None, brand_cfg=self.brand_cfg,
            )
            # 使用 v2.0 合成覆盖最终分（0-10→0-100）
            final_0_10, _breakdown = self.synthesise_final(voting, channels)
            final_0_100 = round(clamp(final_0_10 * 10.0, 0.0, 100.0), 1)
            grade = GradeResult(
                demand_potential=grade.demand_potential,
                brand_portfolio_value=grade.brand_portfolio_value,
                grade_note=grade.grade_note,
                style_id=grade.style_id,
                grade=grade.grade,
                final_score=final_0_100,
                confidence=grade.confidence,
                strengths=grade.strengths,
                weaknesses=grade.weaknesses,
                improvements=grade.improvements,
                recommended_channel=grade.recommended_channel,
                consumer_insights=grade.consumer_insights,
            )
            return FullPrediction(
                info=info, features=feats, voting=voting,
                channels=channels, grade=grade,
                metadata={
                    "llm_backend": "mock",
                    "is_mock": True,
                    "brand_id": self.brand_cfg.brand_id,
                    "tier": "mock_deterministic",
                },
            )
        import time as _time
        _t0 = _time.time()
        feats = extract_style_features(
            self.client, info, cfg=None,
            brand_cfg=self.brand_cfg, llm_backend=self.llm_backend,
        )
        # ===== 应用 Loop1 feature_biases（校准偏置）=====
        # Loop1 校准后，对 VLM 提取的 F01-F10 特征乘以偏置系数。
        # 偏置系数 = ρ_i / mean(ρ)，clamp 到 [0.7, 1.3]。
        # 注意：mock 路径已有此逻辑，真实 LLM 路径之前缺失（v1.4.39 bug fix）
        biases = getattr(self.brand_cfg, "_feature_biases", None)
        if biases:
            for feat_id, feat in feats.features.items():
                bias = biases.get(feat_id) or biases.get(feat_id[:3], 1.0)
                if abs(bias - 1.0) > 0.001:
                    feat.score = round(
                        clamp(feat.score * bias, 0.0, 10.0), 3
                    )
        # hybrid 模式：人设投票强制走 text_client（云端文本模型）；
        # local / 默认：VLM + 文本都走 self.client（Ollama 本地或云端）
        voting = run_persona_voting(
            self.text_client if self.llm_backend == "hybrid" else self.client,
            info, feats, None, brand_cfg=self.brand_cfg,
        )
        channels, _ = calculate_channel_scores(
            info, feats, voting, cfg=None, all_style_prices=price_pool, brand_cfg=self.brand_cfg,
            category_id=info.category or None,
        )
        grade = decide_grade(
            info, feats, voting, channels, cfg=None, brand_cfg=self.brand_cfg,
        )
        # ===== 用 Loop3 校准权重覆盖 final_score（v1.4.39 bug fix）=====
        # decide_grade 内的 _default_aggregate 用的是硬编码权重
        # (persona=0.40, natural=0.20, live=0.20, value=0.20)，
        # synthesise_final() 会优先用 Loop3 校准产物（engine_weights + channel_split），
        # 更能反映真实销量相关性。grade 字段保留 decide_grade 的分级结果
        # （分级规则里的 S/A+/A/P 阈值和反对率/渠道硬规则仍然有价值），
        # 只替换 final_score 这个数值。
        # 注意：mock 路径已有此覆盖，真实 LLM 路径之前缺失。
        final_0_10, _breakdown = self.synthesise_final(voting, channels)
        final_0_100 = round(clamp(final_0_10 * 10.0, 0.0, 100.0), 1)
        grade = GradeResult(
            style_id=grade.style_id,
            grade=grade.grade,
            final_score=final_0_100,
            confidence=grade.confidence,
            demand_potential=grade.demand_potential,
            brand_portfolio_value=grade.brand_portfolio_value,
            grade_note=grade.grade_note,
            strengths=grade.strengths,
            weaknesses=grade.weaknesses,
            improvements=grade.improvements,
            recommended_channel=grade.recommended_channel,
            consumer_insights=grade.consumer_insights,
        )
        _elapsed = round(_time.time() - _t0, 1)
        # 从 features 里提取 VLM 实际用了哪些模型（model_scores 会有）
        _feat_models: set[str] = set()
        for f in feats.features.values():
            for m in (f.model_scores or {}).keys():
                _feat_models.add(m)
        metadata = {
            "llm_backend": self.llm_backend,
            "is_mock": False,
            "feature_models": sorted(_feat_models) or ["qwen-vl-plus"],
            "persona_models": getattr(self, "_api_persona_models", None) or ["qwen-max", "deepseek-v3"],
            "elapsed_s": _elapsed,
            "brand_id": self.brand_cfg.brand_id,
            # 本款 API 用量（调用次数/token，从上次快照后累计），
            # 批次总表汇总展示「钱花哪了」
            "api_usage": self.client.usage_tracker.take(),
        }
        return FullPrediction(
            info=info, features=feats, voting=voting,
            channels=channels, grade=grade, metadata=metadata,
        )

    # ===== 历史批次持久化入口 =====
    def save_to_history(
        self,
        predictions: list[FullPrediction],
        notes: str = "",
    ) -> str:
        """把预测结果写入 SQLite + CSV，下次启动自动加载累积分布。"""
        is_mock = (self.llm_backend == "mock")
        return self.history.append(
            predictions,
            brand_id=self.brand_id,
            llm_backend=self.llm_backend,
            is_mock=is_mock,
            notes=notes,
        )

    # ===== Smoke test：直接喂 N 款 mock 款式 =====
    def run_smoke_test_data(
        self,
        n: int = 10,
    ) -> list[FullPrediction]:
        """直接生成 n 款 mock 数据并跑完整流水线（不读Excel、不调LLM）。

        Returns:
            list[FullPrediction]：每款的完整预测结果（已调用 synthesise_final_score 合成）
        """
        categories_pool: list[tuple[str, str, float]] = [
            ("jacket", "夹克外套", 259.0),
            ("tshirt", "T恤", 129.0),
            ("shorts", "短裤", 119.0),
            ("pants", "长裤", 149.0),
            ("vest", "背心", 69.0),
            ("down_jacket", "羽绒服", 399.0),
            ("sun_protection", "防晒衣", 179.0),
            ("shirt", "衬衫", 169.0),
            ("hoodie", "卫衣", 139.0),
            ("vest_padded", "马甲", 219.0),
        ]
        styles: list[StyleInfo] = []
        for i in range(n):
            cid, cname, price_base = categories_pool[i % len(categories_pool)]
            import random
            random.seed(zlib.crc32(f"smk_{self.brand_id}_{i}".encode()))
            price = round(price_base + random.uniform(-20, 30), 0)
            styles.append(StyleInfo(
                style_id=f"T251{i+1:03d}",
                images=[],
                fab_description=(
                    f"{cname}款：采用功能性面料，多口袋系统，适合6-18岁全客群"
                    f"{self.brand_cfg.decision_structure.default_target_age}岁儿童日常和户外穿着。"
                ),
                category=cid,
                price=price,
                season="春夏" if i % 2 == 0 else "秋冬",
                is_main_push=(i % 5 == 0),
                is_live_stream=(i % 3 == 0),
            ))
        results: list[FullPrediction] = []
        for info in styles:
            results.append(self.run_one(info, use_mock=True))
        return results

    def run_backtest_calibration(
        self,
        predictions: list[FullPrediction] | None = None,
        sales_lookup: dict[str, float] | None = None,
        out_dir: Path | None = None,
    ) -> "RunAllLoopsResult | None":
        """对一批预测结果跑 3Loop 优化内核 + 残差分离（spec §5/§8/§9）。

        v2 推荐的校准入口，替代 v1 run_batch 里仅跑 Loop2 的 train_calibration。
        产物写到 brand_cfg.calibrated_dir，下次 run_one 可加载 loop1/2/3 权重。

        Args:
            predictions: 已跑完的预测结果；为 None 时用 run_smoke_test_data(10) 兜底
            sales_lookup: 真实销量映射 {style_id: sales_qty}；mock 数据需注入真实销量
            out_dir: 预测产物目录；默认 brand_cfg.calibrated_dir 的父级 output

        Returns:
            RunAllLoopsResult 或 None（样本不足/失败时）
        """
        if predictions is None:
            predictions = self.run_smoke_test_data(n=10)

        from .core.optimization_kernel import build_history_df, run_all_loops
        history_df = build_history_df(predictions, sales_lookup=sales_lookup)

        # 销量全 0 → run_all_loops 无信号，提前拦截
        sales_col = "sales"
        if sales_col not in history_df.columns or history_df[sales_col].sum() == 0:
            log.warning(
                "⚠️ predictions 全无销量（info.sales_qty=0 且 sales_lookup 为空），"
                "3Loop 无回归信号，跳过校准。请用 sales_lookup 注入真实销量。"
            )
            return None

        if out_dir is None:
            # 默认输出到 brand calibrated_dir 的同级 output
            out_dir = Path(self.brand_cfg.calibrated_dir).parent / "output"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = run_all_loops(
                brand_cfg=self.brand_cfg,
                history_df=history_df,
                prediction_artifacts_dir=out_dir,
                sales_col=sales_col,
            )
            log.info(
                "✅ 3Loop 校准完成：写入 %d 个产物文件 → %s",
                len(result.output_files), self.brand_cfg.calibrated_dir,
            )
            log.info(
                "   Loop1 applied=%s (ρ %.3f→%.3f), Loop2 applied=%s (ρ %.3f→%.3f), "
                "Loop3 applied=%s (ρ %.3f→%.3f)",
                result.loop1.applied,
                result.loop1.old_spearman_avg, result.loop1.new_spearman_avg,
                result.loop2.applied,
                result.loop2.old_spearman, result.loop2.new_spearman,
                result.loop3.applied,
                result.loop3.old_engine_spearman, result.loop3.new_engine_spearman,
            )
            log.info(
                "   残差 μ=%.4f σ=%.4f, over=%d, under=%d, bias=%s",
                result.residual.residual_mean, result.residual.residual_std,
                len(result.residual.overperformers),
                len(result.residual.underperformers),
                result.residual.system_bias_flag,
            )
            return result
        except Exception as e:
            log.exception("3Loop 校准失败：%s", e)
            return None


# ========== CLI 入口（保持原逻辑）==========
def main(argv=None):
    parser = argparse.ArgumentParser(description="童装爆款预测AI工具 v2.0 阶段一")
    parser.add_argument("mode", choices=["backtest", "predict", "train"],
                        help="backtest=回测（需带销量/分级列），predict=预测新款式，train=仅重新训练校准层")
    parser.add_argument("--styles", default=None, help="款式Excel路径，默认 data/styles.xlsx")
    parser.add_argument("--images", default=None, help="图片目录，默认 data/images/")
    parser.add_argument("--output", default=None, help="输出目录，默认 output/")
    parser.add_argument("--env", default=None, help="自定义.env路径")
    parser.add_argument("--brand", default="mipo",
                        help="品牌ID（默认 mipo），将用 brand_profiles/<brand>/ 下配置")
    parser.add_argument("-v", "--verbose", action="store_true", help="调试日志")
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    cfg = load_config(Path(args.env) if args.env else None)
    styles_path = Path(args.styles) if args.styles else cfg.paths.styles_xlsx
    images_dir = Path(args.images) if args.images else cfg.paths.images_dir
    out_dir = Path(args.output) if args.output else cfg.paths.output_dir

    if args.mode == "train":
        pkl = out_dir / "calibration_train_predictions.pkl"
        if pkl.exists():
            import pickle
            preds = pickle.load(pkl.read_bytes())
            # v2 路径：3Loop 优化内核替代 v1 train_calibration
            sales_lookup = {p.info.style_id: float(p.info.sales_qty or 0) for p in preds}
            pl = PredictionPipeline(brand_id=args.brand, llm_backend="mock")
            result = pl.run_backtest_calibration(predictions=preds, sales_lookup=sales_lookup)
            if result is not None:
                log.info("✅ 3Loop 校准完成，产物写入 %s", pl.brand_cfg.calibrated_dir)
            else:
                log.info("ℹ️ 样本不足或无销量信号，校准未执行（冷启动模式继续）")
            return
        log.warning("训练模式需先跑过一次 backtest（会自动触发训练），或手动准备 %s", pkl)
        return

    run_batch(cfg, styles_path, images_dir, args.mode, out_dir, brand_id=args.brand)


if __name__ == "__main__":
    main()
