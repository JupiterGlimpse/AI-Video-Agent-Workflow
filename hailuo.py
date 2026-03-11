# ╔══════════════════════════════════════════════════════════════╗
# ║                  Hailuo / MiniMax 视频生成 API               ║
# ║          提交任务 → 轮询状态 → 下载视频 → 截取验证帧             ║
# ╚══════════════════════════════════════════════════════════════╝

from __future__ import annotations

import base64
import os
import subprocess
import time

import requests
import loading

API_KEY = os.environ.get("MINIMAX_API_KEY", "")
BASE = "https://api.minimax.io/v1"


def _headers(json: bool = False) -> dict:
    h = {"Authorization": f"Bearer {API_KEY}"}
    if json:
        h["Content-Type"] = "application/json"
    return h


# ┌──────────────────────────────────────────────────────────────┐
# │                      三步 API 调用                            │
# └──────────────────────────────────────────────────────────────┘

def _encode_image_to_data_url(path: str) -> str:
    """将本地图片编码为 data URL，供 first_frame_image 使用。"""
    ext = path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _submit(prompt: str, duration: int, reference_image_path: str = "") -> str:
    """提交生成任务，返回 task_id。
    reference_image_path 不为空时走 I2V 模式，锁定产品外观一致性。
    """
    body: dict = {
        "model": "MiniMax-Hailuo-2.3",
        "prompt": prompt,
        "prompt_optimizer": False,
        "duration": duration,
    }
    if reference_image_path:
        body["first_frame_image"] = _encode_image_to_data_url(reference_image_path)

    resp = requests.post(f"{BASE}/video_generation", headers=_headers(json=True), json=body)
    data = resp.json()
    code = data.get("base_resp", {}).get("status_code")
    if code != 0:
        raise RuntimeError(f"提交失败 [{code}]：{data['base_resp']['status_msg']}")
    return data["task_id"]


def _poll(task_id: str, timeout: int) -> str:
    """轮询任务状态，返回 file_id。网络异常自动重试，不中断等待。"""
    start = time.time()
    deadline = start + timeout
    while time.time() < deadline:
        try:
            data = requests.get(
                f"{BASE}/query/video_generation",
                headers=_headers(),
                params={"task_id": task_id},
                timeout=15,
            ).json()
            status = data.get("status", "")
            if status == "Success":
                print()  # 换行，收尾进度条
                return str(data["file_id"])
            if status == "Fail":
                raise RuntimeError(f"视频生成失败：{data}")
            loading.tick(time.time() - start, timeout)
        except RuntimeError:
            raise
        except Exception as e:
            print(f"    [网络异常，自动重试] {e}")
        time.sleep(5)
    raise TimeoutError(f"超时 {timeout}s，task_id={task_id}")


def _get_download_url(file_id: str) -> str:
    """通过 file_id 拿到有效期 1 小时的下载链接。"""
    data = requests.get(
        f"{BASE}/files/retrieve",
        headers=_headers(),
        params={"file_id": file_id},
    ).json()
    return data["file"]["download_url"]


# ┌──────────────────────────────────────────────────────────────┐
# │                      下载 + 截帧                              │
# └──────────────────────────────────────────────────────────────┘

def _download(url: str, path: str) -> None:
    """流式下载视频到本地。"""
    with requests.get(url, stream=True) as r:
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def _extract_frames(video_path: str, base_path: str, timestamps: list[str]) -> list[str]:
    """截取多个时间点的帧，返回所有帧路径。
    覆盖 Temporal Consistency 和 Motion Quality 评估需要的时序信息。
    """
    paths = []
    for i, ts in enumerate(timestamps):
        frame_path = f"{base_path}_f{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-ss", ts, "-vframes", "1", frame_path, "-y"],
            check=True,
            capture_output=True,
        )
        paths.append(frame_path)
    return paths


# ┌──────────────────────────────────────────────────────────────┐
# │                        对外接口                               │
# └──────────────────────────────────────────────────────────────┘

def generate(
    prompt: str,
    out_dir: str = "outputs",
    duration: int = 6,
    reference_image_path: str = "",
) -> tuple[str, list[str]]:
    """
    完整生成流程：提交 → 轮询 → 下载 → 截多帧。
    reference_image_path 不为空时走 I2V，锁定产品外观。
    返回 (video_path, frame_paths)，frame_paths 包含首/中/尾三帧供 Agent 4 多维评估。
    """
    os.makedirs(out_dir, exist_ok=True)

    mode = "I2V（参考图锚定）" if reference_image_path else "T2V（纯文本）"
    print(f"  提交到 Hailuo API [{mode}]...")
    task_id = _submit(prompt, duration, reference_image_path)

    print(f"  task_id={task_id}，等待生成（最长 5 分钟）...")
    file_id = _poll(task_id, timeout=300)

    print("  获取下载链接...")
    url = _get_download_url(file_id)

    video_path = os.path.join(out_dir, f"{task_id}.mp4")
    base_path = os.path.join(out_dir, f"{task_id}")

    print(f"  下载视频 → {video_path}")
    _download(url, video_path)

    # 截三帧：首帧（1s）/ 中帧（3s）/ 尾帧（5s）
    # 覆盖 Temporal Consistency 和 Motion Quality 评估
    print("  截取验证帧（首/中/尾）...")
    frame_paths = _extract_frames(video_path, base_path, ["00:00:01", "00:00:03", "00:00:05"])

    return video_path, frame_paths
