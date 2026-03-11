# ╔══════════════════════════════════════════════════════════════╗
# ║                   AI 视频 Agent Pipeline                      ║
# ║   意图理解 → 分镜规划 → Prompt 生成 → Hailuo 生成 → 视觉验证   ║
# ╚══════════════════════════════════════════════════════════════╝

from __future__ import annotations

import os
import re
import sys
import termios
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from agents import run_consistency_agent, run_edit_agent, run_intent_agent, run_prompt_agent, run_storyboard_agent, run_verify_agent
import hailuo
from models import EvalResult, Shot, Storyboard

client = Anthropic()
SAVE_PATH = "storyboard.json"


# ┌──────────────────────────────────────────────────────────────┐
# │                        展示辅助函数                            │
# └──────────────────────────────────────────────────────────────┘

def show_intent(intent: dict) -> None:
    print("\n【Agent 1 — 目标提取】")
    print(f"  类型：{intent.get('video_type')}  时长：{intent.get('duration')}  镜头数：{intent.get('shot_count')}")
    print(f"  卖点：{', '.join(intent.get('key_messages', []))}")
    print(f"  结构：{' → '.join(intent.get('structure', []))}")


def show_storyboard(shots: list[Shot]) -> None:
    print(f"\n【Agent 2 — 分镜计划，共 {len(shots)} 个镜头】")
    for s in shots:
        print(f"  [{s.index}] {s.goal}")


def show_eval(result: EvalResult) -> None:
    motion_tag = " [运动镜头-跳过运动评估]" if result.is_motion_shot else ""
    print(f"  视觉质量：{result.visual_score}/10  语义对齐：{result.alignment_score}/10  时间一致性：{result.temporal_score}/10  综合：{result.overall:.1f}/10{motion_tag}")
    icons = {"accept": "✅", "show_user": "🟡", "retry": "❌"}
    print(f"  建议：{icons.get(result.verdict, '?')} {result.verdict.upper()} — {result.reason}")
    if result.fix_suggestion:
        print(f"  修正：{result.fix_suggestion}")


# ┌──────────────────────────────────────────────────────────────┐
# │                  并行生成所有镜头 prompt                        │
# └──────────────────────────────────────────────────────────────┘

def generate_all_prompts(shots: list[Shot]) -> None:
    """生成所有镜头 prompt，采用关键帧锚定策略：
    先生成镜头1作为视觉锚点，再并行生成其余镜头并传入锚点保持风格一致。
    """
    print(f"\n[Agent 3] 生成 {len(shots)} 个镜头的 prompt（关键帧锚定）...")

    # 第一步：单独生成镜头1，作为全片视觉锚点
    anchor_shot = shots[0]
    anchor_shot.prompt = run_prompt_agent(client, anchor_shot)
    anchor_shot.status = "prompted"
    print(f"  镜头 1 prompt 就绪（视觉锚点）")

    if len(shots) == 1:
        return

    # 第二步：并行生成其余镜头，全部携带镜头1的 prompt 作为风格参考
    anchor_prompt = anchor_shot.prompt

    def _gen(shot: Shot) -> tuple[int, str]:
        return shot.index, run_prompt_agent(client, shot, anchor_prompt=anchor_prompt)

    with ThreadPoolExecutor(max_workers=len(shots) - 1) as executor:
        futures = {executor.submit(_gen, s): s for s in shots[1:]}
        for future in as_completed(futures):
            idx, prompt = future.result()
            shot = next(s for s in shots if s.index == idx)
            shot.prompt = prompt
            shot.status = "prompted"
            print(f"  镜头 {idx} prompt 就绪")


# ┌──────────────────────────────────────────────────────────────┐
# │          单个镜头：自动生成视频 → 验证 → 不达标自动重试           │
# └──────────────────────────────────────────────────────────────┘

def process_shot(shot: Shot, anchor_prompt: str = "", reference_image: str = "") -> list[str]:
    """
    自动闭环：Hailuo 生成 → 截帧 → Agent 4 验证 → 三种结果：
      ✅ 达标继续
      ❌ 不达标 → 自动修正 prompt 重试
      ✏️  局部编辑 → Edit Agent 看图理解指令 → 精准修改 prompt 重试
    返回最后一次成功生成的 frame_paths，供上层做 Frame Chaining 和一致性检查。
    """
    fix_hint = shot.eval_note
    max_retries = 3
    attempt = 0
    best: tuple[float, str] = (0.0, shot.prompt)  # (score, prompt) 历史最佳
    last_frame_paths: list[str] = []

    while attempt < max_retries:
        print(f"\n── 镜头 {shot.index}（{shot.narrative_role or ''}）──────────────")
        print(f"目标：{shot.goal}\n")
        print(f"【Prompt】\n{shot.prompt}\n")

        attempt += 1
        print(f"  第 {attempt}/{max_retries} 次尝试")
        try:
            video_path, frame_paths = hailuo.generate(
                shot.prompt, reference_image_path=reference_image
            )
            last_frame_paths = frame_paths
            print(f"  视频：{video_path}")
        except Exception as e:
            print(f"  生成失败：{e}")
            if input("  跳过此镜头？(y/n)：").strip().lower() == "y":
                shot.status = "skipped"
                break
            continue

        print("  Agent 4 验证中（多帧评估）...")
        result = run_verify_agent(client, shot, frame_paths)
        show_eval(result)

        # 记录历史最佳 prompt
        if result.overall > best[0]:
            best = (result.overall, shot.prompt)

        # ── 策略决策层 ──────────────────────────────────────────
        if result.verdict == "accept":
            shot.status = "verified"
            break

        elif result.verdict == "show_user":
            print("\n  综合分在可接受范围，建议接受。")
            print("  [1] 接受  [2] 局部编辑  [3] 自动修正  [4] 跳过")
            choice = input("  选择 (1/2/3/4)：").strip()
            if choice == "1":
                shot.status = "verified"
                break
            elif choice == "4":
                shot.status = "skipped"
                break
            elif choice == "2":
                instruction = input("  编辑指令：").strip()
                shot.prompt = run_edit_agent(client, shot, frame_path, instruction, anchor_prompt)
            else:
                fix_hint = f"[{result.dimension}] {result.fix_suggestion}"
                shot.prompt = run_prompt_agent(client, shot, fix_hint, anchor_prompt)

        else:  # retry
            fix_hint = f"[{result.dimension}] {result.fix_suggestion}"
            shot.eval_note = fix_hint
            shot.status = "failed"
            print("\n  自动修正 prompt...")
            shot.prompt = run_prompt_agent(client, shot, fix_hint, anchor_prompt)

    # 超过最大重试次数 → 回退历史最佳，不丢失已有成果
    if shot.status not in ("verified", "skipped"):
        shot.prompt = best[1]
        print(f"\n  ⚠️  达到重试上限，已回退到历史最佳（综合分 {best[0]:.1f}）。")
        if input("  强制标记为通过？(y/n)：").strip().lower() == "y":
            shot.status = "verified"
        else:
            shot.status = "skipped"

    return last_frame_paths


# ┌──────────────────────────────────────────────────────────────┐
# │                      状态存盘 / 续跑                           │
# └──────────────────────────────────────────────────────────────┘

def try_resume() -> Storyboard | None:
    if not os.path.exists(SAVE_PATH):
        return None
    answer = input(f"发现未完成进度（{SAVE_PATH}），续跑？(y/n)：").strip().lower()
    return Storyboard.load(SAVE_PATH) if answer == "y" else None


# ┌──────────────────────────────────────────────────────────────┐
# │                           主入口                              │
# └──────────────────────────────────────────────────────────────┘

def run() -> None:
    print("=== AI 视频 Agent Pipeline ===\n")

    board = try_resume()

    if board is None:
        brief = input("业务简报：").strip()
        if not brief:
            return

        print("\n[Agent 1] 提取目标...")
        intent = run_intent_agent(client, brief)
        show_intent(intent)

        print("\n[Agent 2] 规划分镜...")
        shots = run_storyboard_agent(client, intent)
        show_storyboard(shots)

        print("\n产品参考图（可选，传入后所有镜头锁定同一外观）")
        termios.tcflush(sys.stdin, termios.TCIFLUSH)  # 清空残留的缓冲输入
        raw = input("图片路径（无则按 Enter 跳过）：")
        # 从输入中提取路径（兼容拖拽带引号、粘贴时混入前文等情况）
        match = re.search(r"['\"]?(/[^'\"]+\.(jpg|jpeg|png|webp))['\"]?", raw, re.IGNORECASE)
        reference_image = match.group(1).strip() if match else ""
        if reference_image and not os.path.exists(reference_image):
            print("  路径不存在，已忽略参考图。")
            reference_image = ""

        board = Storyboard(
            title=intent.get("video_type", "视频"),
            business_goal=str(intent.get("key_messages", [])),
            shots=shots,
            reference_image=reference_image,
        )

        if input("\n确认开始？(Enter 继续 / q 退出)：").strip().lower() == "q":
            return

        generate_all_prompts(board.shots)
        board.save(SAVE_PATH)

    # 跨镜头上下文策略选择（有参考图时自动使用 I2V，无参考图时让用户选）
    if board.reference_image:
        strategy = "0"  # I2V 固定参考图
    else:
        print("\n跨镜头一致性策略（无参考图）：")
        print("  [1] 叙事衔接 — 上一镜头尾帧作为下一镜头首帧（故事连续感）")
        print("  [2] 身份锚定 — 镜头1首帧作为所有后续镜头参考（产品一致性）")
        print("  [3] 无锚定   — 每个镜头独立生成（T2V 原始模式）")
        strategy = input("  选择 (1/2/3，默认2)：").strip() or "2"

    # 逐镜头自动生成 + 验证
    pending = [s for s in board.shots if s.status not in ("verified", "skipped")]
    print(f"\n[Agent 4] 开始自动生成验证，待处理 {len(pending)} 个镜头")

    anchor_prompt = board.shots[0].prompt if board.shots else ""
    auto_reference = board.reference_image
    shot_frame_map: dict[int, list[str]] = {}  # 收集各镜头帧，供 Agent 5 使用

    for i, shot in enumerate(pending):
        frame_paths = process_shot(shot, anchor_prompt=anchor_prompt, reference_image=auto_reference)

        if frame_paths:
            shot_frame_map[shot.index] = frame_paths

        # 根据策略更新下一镜头的参考帧
        if shot.status == "verified" and frame_paths:
            if strategy == "1":  # 叙事衔接：尾帧 → 下一镜头首帧
                auto_reference = frame_paths[-1]
                print(f"  [Frame Chaining] 镜头{shot.index}尾帧 → 镜头{shot.index + 1}首帧")
            elif strategy == "2" and i == 0:  # 身份锚定：仅用镜头1首帧
                auto_reference = frame_paths[0]
                print(f"  [Identity Memory] 镜头1首帧已设为全局参考 → {auto_reference}")

        board.save(SAVE_PATH)

    verified = sum(1 for s in board.shots if s.status == "verified")
    skipped = sum(1 for s in board.shots if s.status == "skipped")
    print(f"\n=== 完成 ✅ {verified} 达标  ⏭ {skipped} 跳过  共 {len(board.shots)} 个镜头 ===")

    # Agent 5：跨镜头一致性检查（至少 2 个镜头有帧数据才运行）
    if len(shot_frame_map) >= 2:
        print("\n[Agent 5] 跨镜头一致性检查...")
        report = run_consistency_agent(client, shot_frame_map)
        print("\n【一致性报告】")
        print(report)


if __name__ == "__main__":
    run()
