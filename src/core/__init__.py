"""fashion-hit-engine CORE 子包

三大评分引擎 + 优化内核：
- persona_voting（人设投票，来自 src.persona_voting）
- channel_scoring（双渠道评分，来自 src.channel_scoring）
- price_value（价格价值，来自 src.channel_scoring 或独立模块）
- ensemble_engine（三大引擎合成）
- optimization_kernel（3Loop 校准内核，Task3 实现）
"""
from __future__ import annotations

from .ensemble_engine import synthesise_final_score

__all__ = ["synthesise_final_score"]
