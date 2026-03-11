# ╔══════════════════════════════════════════════════════════════╗
# ║                         数据模型层                            ║
# ║         Shot → Storyboard → EvalResult 三层核心结构           ║
# ╚══════════════════════════════════════════════════════════════╝

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


# ┌──────────────────────────────────────────────────────────────┐
# │  Shot — 单个镜头，pipeline 最小执行单元                        │
# └──────────────────────────────────────────────────────────────┘

@dataclass
class Shot:
    index: int
    goal: str            # 镜头要传达的目标（也是 Verify Agent 的评估标准）
    emotion: str         # 情绪基调
    action: str          # 动作描述
    style: str           # 视觉风格
    # ── MEVG 叙事连接字段 ──────────────────────────────────────
    narrative_role: str = ""   # 这个镜头在叙事弧中的角色（如"开场建立基调"）
    follows_from: str = ""     # 从上一个镜头承接的叙事逻辑
    leads_to: str = ""         # 为下一个镜头埋下的铺垫
    # ── 运行时状态 ────────────────────────────────────────────
    prompt: str = ""
    status: str = "pending"    # pending | verified | failed | skipped
    eval_note: str = ""


# ┌──────────────────────────────────────────────────────────────┐
# │  Storyboard — 完整分镜计划 + 序列化支持（用于存盘/续跑）        │
# └──────────────────────────────────────────────────────────────┘

@dataclass
class Storyboard:
    title: str
    business_goal: str
    shots: list[Shot]
    reference_image: str = ""  # 产品参考图路径，不为空时所有镜头走 I2V 模式

    def save(self, path: str) -> None:
        """将当前进度序列化到 JSON 文件。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> Storyboard:
        """从 JSON 文件恢复进度。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["shots"] = [Shot(**s) for s in data["shots"]]
        return cls(**data)


# ┌──────────────────────────────────────────────────────────────┐
# │  EvalResult — Agent 4 输出，驱动重试或放行                     │
# └──────────────────────────────────────────────────────────────┘

@dataclass
class EvalResult:
    # ── 多维评分（0-10）────────────────────────────────────────
    visual_score: float      # 画面质量：清晰度、构图、光线
    alignment_score: float   # 语义一致性：画面是否符合镜头目标
    temporal_score: float    # 时间一致性：多帧主体稳定性、无闪烁
    overall: float           # 综合分：visual*0.3 + alignment*0.5 + temporal*0.2
    # ── 策略决策 ───────────────────────────────────────────────
    verdict: str             # "accept" | "show_user" | "retry"
    is_motion_shot: bool     # 运动镜头：true 时跳过运动评估，避免静帧误判
    # ── 诊断信息 ───────────────────────────────────────────────
    dimension: str
    reason: str
    fix_suggestion: str
