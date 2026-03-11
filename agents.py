# ╔══════════════════════════════════════════════════════════════╗
# ║                        四个 Agent 实现                        ║
# ║   意图提取 → 分镜规划 → Prompt 生成 → 视觉验证                 ║
# ║                                                              ║
# ║   设计原则（来自 Anthropic Multi-Agent Research）：            ║
# ║   · Teach Delegation：每个 Agent 收到明确的目标+边界+成功标准   ║
# ║   · Scale Effort：复杂度决定子任务数量，写入 prompt 而非硬编码  ║
# ║   · Tool Interface：Agent 间接口与人机接口同等重要             ║
# ╚══════════════════════════════════════════════════════════════╝

from __future__ import annotations

import base64
import json
import re

from anthropic import Anthropic
from models import EvalResult, Shot


# ┌──────────────────────────────────────────────────────────────┐
# │                      System Prompts                           │
# │   每个 prompt 包含：目标 / 输出格式 / 任务边界 / 成功标准        │
# └──────────────────────────────────────────────────────────────┘

# Agent 1：意图提取
# 任务边界：只做结构化提取，不做创意发挥；shot_count 由时长决定
_INTENT_SYSTEM = """你是商业视频策划专家，负责从业务简报中提取结构化目标。

任务边界：
- 只提取用户明确说明的信息，不要自行添加卖点
- shot_count 规则：视频 ≤15s → 3个镜头，16~30s → 4个，>30s → 5个

成功标准：输出的 key_messages 必须全部能在原始简报中找到对应依据。

只输出 JSON，格式：
{
  "video_type": "视频类型",
  "key_messages": ["卖点1", "卖点2"],
  "structure": ["开场", "功能演示", "品牌收尾"],
  "duration": "30s",
  "shot_count": 4
}"""

# Agent 2：分镜规划
# 任务边界：严格按 shot_count 生成，每个镜头必须对应 key_messages 中的至少一个卖点
_STORYBOARD_SYSTEM = """你是分镜规划专家，负责将业务目标转化为可执行的镜头列表。

任务边界：
- 严格按照 intent 中的 shot_count 生成对应数量的镜头，不多不少
- 每个镜头的 goal 必须对应 key_messages 中的至少一个卖点
- goal 就是 Verify Agent 的评估标准，必须具体可验证（避免"展示产品"这种模糊描述）

成功标准：所有 key_messages 至少在一个镜头中被覆盖。

只输出 JSON 数组，每个镜头必须包含叙事连接字段：
[
  {
    "index": 1,
    "goal": "特写鞋底防滑纹路，画面停留2秒后淡出",
    "emotion": "专业可靠",
    "action": "鞋底从模糊到清晰的推进特写",
    "style": "白底强光，产品电商风",
    "narrative_role": "开场建立产品专业感",
    "follows_from": "",
    "leads_to": "为下一镜头的动态演示奠定静谧基调，运镜减速结尾"
  }
]"""

# Agent 3：Prompt 生成
# 任务边界：处理单个镜头，但感知叙事上下文（MEVG）
_PROMPT_SYSTEM = """你是 Hailuo AI 视频 prompt 专家，负责将单个镜头目标转化为可直接使用的英文 prompt。

任务边界：
- 如果提供了「叙事上下文」，运镜节奏和过渡方式必须体现镜头间的逻辑衔接
- 如果提供了「风格锚点」，lighting/color palette/visual style 必须与锚点保持一致
- 如果提供了上次失败信息，只修改失败维度，其他部分保持不变

成功标准：prompt 必须包含以下全部要素：
  画面主体 / 核心动作 / 运镜方式 / 光线描述 / 视觉风格 / 时长（秒数）

构图约束（每个 prompt 必须遵守）：
  subject centered in frame, full subject visible, no aggressive crop,
  maintain head-to-shoulder or full-body framing as appropriate

负向屏蔽（prompt 中禁止引入以下效果）：
  no lens flare, no blue glow, no neon light, no sci-fi lighting,
  no color grading artifacts, no cyberpunk tones, no god rays,
  no dramatic color shift, no glitch effect

只输出英文 prompt，不要任何解释或格式标签。"""

# Edit Agent：局部编辑
# 用户对已生成视频有局部不满，通过自然语言指令精准修改，保留满意部分
_EDIT_SYSTEM = """你是视频 prompt 精准编辑专家。用户对已生成的视频有局部修改需求。

你会收到：
1. 当前视频帧截图（了解现有画面内容）
2. 用户的自然语言编辑指令（如"把光线改暖"、"背景虚化"）
3. 原始镜头目标

任务边界：
- 分析截图，识别所有视觉要素（角色、构图、动作、色调等）
- 只修改用户指令中明确提到的部分，其余全部保留
- 生成新的完整 Hailuo 英文 prompt

只输出英文 prompt，不要任何解释。"""

# Agent 4：视觉验证
# 多维评分 + 策略决策，避免单一 pass/fail 导致过度 retry
_VERIFY_SYSTEM = """你是视频质量评估专家，你会收到同一视频的多帧截图（首帧/中帧/尾帧），进行多维评分并给出策略建议。

第一步：判断 is_motion_shot
- 如果镜头目标包含运动词（升降/旋转/移动/转动/后仰等），is_motion_shot=true
- 运动镜头只能从静帧判断构图和风格，不能判断运动是否完成，因此 alignment 评分时忽略运动部分

评分维度（0-10）：
- visual_score：画面清晰度、构图、光线质量（基于所有帧的整体判断）
- alignment_score：画面与镜头目标语义一致性（is_motion_shot=true 时只评静态部分）
- temporal_score：时间一致性 —— 对比多帧，主体是否稳定、无闪烁、无突变（若只有一帧则给 5 分）
- overall = visual_score × 0.3 + alignment_score × 0.5 + temporal_score × 0.2

verdict 规则（严格按此执行）：
- overall > 7.0 → "accept"
- 5.0 < overall ≤ 7.0 → "show_user"
- overall ≤ 5.0 → "retry"

dimension 从以下选一个：视觉质量 / 语义对齐 / 时间一致性 / 运镜 / 场景细节 / 构图

只输出 JSON：
{
  "visual_score": 8.0,
  "alignment_score": 6.0,
  "temporal_score": 7.0,
  "overall": 6.8,
  "verdict": "show_user",
  "is_motion_shot": false,
  "dimension": "运镜",
  "reason": "一句话原因",
  "fix_suggestion": "具体修正建议"
}"""


# ┌──────────────────────────────────────────────────────────────┐
# │                        工具函数                               │
# └──────────────────────────────────────────────────────────────┘



def _parse_json(text: str):
    """
    Robust JSON parser for LLM outputs.
    自动修复常见 JSON 错误：
    - markdown ```json
    - 前后解释文字
    - trailing comma
    - 单引号
    """

    if not text:
        raise ValueError("LLM returned empty response")

    text = text.strip()

    # 1️⃣ 去掉 markdown code block
    if text.startswith("```"):
        text = re.sub(r"```json|```", "", text).strip()

    # 2️⃣ 提取 JSON 对象或数组
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if not match:
        print("\n⚠️ 未找到 JSON：")
        print(text)
        raise ValueError("No JSON found in LLM output")

    json_text = match.group(1)

    # 3️⃣ 修复 trailing comma
    json_text = re.sub(r",\s*}", "}", json_text)
    json_text = re.sub(r",\s*]", "]", json_text)

    # 4️⃣ 修复单引号（LLM常犯）
    if "'" in json_text and '"' not in json_text:
        json_text = json_text.replace("'", '"')

    # 5️⃣ 尝试解析
    try:
        return json.loads(json_text)

    except json.JSONDecodeError as e:
        print("\n❌ JSON解析失败")
        print("错误:", e)
        print("\n===== LLM RAW OUTPUT =====")
        print(text)
        print("==========================")

        raise

def _encode_image(path: str) -> tuple[str, str]:
    """将本地图片编码为 base64，返回 (data, media_type)。"""
    ext = path.rsplit(".", 1)[-1].lower()
    media_type = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode(), media_type


def _call(client: Anthropic, system: str, user: str, max_tokens: int = 512) -> str:
    """统一的文本 Agent 调用入口。"""
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


# ┌──────────────────────────────────────────────────────────────┐
# │                        四个 Agent                             │
# └──────────────────────────────────────────────────────────────┘

def run_intent_agent(client: Anthropic, brief: str) -> dict:
    """Agent 1：业务简报 → 结构化意图（含 shot_count）。"""
    return _parse_json(_call(client, _INTENT_SYSTEM, brief))


def run_storyboard_agent(client: Anthropic, intent: dict) -> list[Shot]:
    """Agent 2：结构化意图 → 分镜列表，严格按 shot_count 生成。"""
    user = f"业务目标：\n{json.dumps(intent, ensure_ascii=False, indent=2)}"
    raw = _parse_json(_call(client, _STORYBOARD_SYSTEM, user, max_tokens=1024))
    return [Shot(**s) for s in raw]


def run_prompt_agent(
    client: Anthropic,
    shot: Shot,
    fix_hint: str = "",
    anchor_prompt: str = "",
) -> str:
    """Agent 3：镜头目标 → Hailuo 英文 prompt。
    · anchor_prompt：镜头1的 prompt，锁定视觉风格（关键帧锚定）
    · shot.follows_from / leads_to：叙事上下文，影响运镜节奏（MEVG）
    """
    user = f"目标：{shot.goal}\n情绪：{shot.emotion}\n动作：{shot.action}\n风格：{shot.style}"
    # MEVG：叙事上下文注入
    if shot.narrative_role:
        user += f"\n\n【叙事上下文】"
        user += f"\n叙事角色：{shot.narrative_role}"
        if shot.follows_from:
            user += f"\n承接自：{shot.follows_from}"
        if shot.leads_to:
            user += f"\n引导至：{shot.leads_to}"
    # 关键帧锚定
    if anchor_prompt:
        user += f"\n\n【风格锚点 - lighting/color/style 必须保持一致】\n{anchor_prompt}"
    if fix_hint:
        user += f"\n\n上次验证失败，修正方向：{fix_hint}"
    return _call(client, _PROMPT_SYSTEM, user, max_tokens=256)


def run_edit_agent(
    client: Anthropic,
    shot: Shot,
    frame_path: str,
    edit_instruction: str,
    anchor_prompt: str = "",
) -> str:
    """Edit Agent：看当前视频帧 + 用户指令 → 精准局部修改后的新 prompt。"""
    image_data, media_type = _encode_image(frame_path)
    user_text = f"编辑指令：{edit_instruction}\n原镜头目标：{shot.goal}"
    if anchor_prompt:
        user_text += f"\n风格锚点（保持一致）：{anchor_prompt}"
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=256,
        system=_EDIT_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                {"type": "text", "text": user_text},
            ],
        }],
    )
    return resp.content[0].text.strip()


# ┌──────────────────────────────────────────────────────────────┐
# │                    Agent 5：跨镜头一致性                       │
# └──────────────────────────────────────────────────────────────┘

_CONSISTENCY_SYSTEM = """你是视频跨镜头一致性审核专家。你会收到多个镜头各自的代表帧。

评估三个维度（每条一行输出）：
1. 主体一致性：产品/角色外观是否在各镜头间保持一致（形状/颜色/细节）
2. 视觉风格一致性：光线、色调、背景风格是否统一
3. 叙事连贯性：镜头构图、视线引导是否有视觉逻辑

最后一行：整体结论（达标 / 需修正）+ 最突出的问题（如有）。
仅输出报告文本，不要 JSON 或格式标签。"""


def run_consistency_agent(
    client: Anthropic, shot_frame_map: dict[int, list[str]]
) -> str:
    """Agent 5：跨镜头一致性检查，每个镜头取首帧作为代表帧。"""
    content: list = [{"type": "text", "text": "请评估以下各镜头代表帧的跨镜头一致性："}]
    for shot_idx in sorted(shot_frame_map.keys()):
        frame_path = shot_frame_map[shot_idx][0]
        image_data, media_type = _encode_image(frame_path)
        content.append({"type": "text", "text": f"\n[镜头 {shot_idx}]"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}})
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        system=_CONSISTENCY_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text.strip()


def run_verify_agent(client: Anthropic, shot: Shot, frame_paths: list[str]) -> EvalResult:
    """Agent 4：多帧截图 + 镜头目标 → 评估结果（Claude Vision）。
    frame_paths 通常为首/中/尾三帧，覆盖 Temporal Consistency 评估。
    """
    labels = ["首帧", "中帧", "尾帧"]
    content: list = []
    for i, path in enumerate(frame_paths):
        label = labels[i] if i < len(labels) else f"帧{i}"
        image_data, media_type = _encode_image(path)
        content.append({"type": "text", "text": f"[{label}]"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}})
    goal_text = f"镜头目标（评估标准）：{shot.goal}\n情绪：{shot.emotion}\n动作：{shot.action}\n风格：{shot.style}"
    content.append({"type": "text", "text": goal_text})
    resp = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=512,
        system=_VERIFY_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return EvalResult(**_parse_json(resp.content[0].text))
