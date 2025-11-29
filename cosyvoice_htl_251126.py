#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch zero-shot TTS for CosyVoice2 — JSONL + WAV directory, random sampling, no spkinfo.

功能概述：
- 输入：
    - 一个 JSONL 文件：每行是一段对话，里面包含 "role": "user" 的文本
    - 一个 WAV 目录：包含大量参考语音，按文件名排序，与 JSONL 行号对齐
- 只从中随机抽取 num-samples 条样本（而不是读全量进内存）
- 对每条样本：
    - 使用该行的 user 文本作为 TTS 目标文本
    - 使用对应行号的 WAV 作为 zero-shot 参考音频
    - 调用 cosyvoice.inference_zero_shot(text, prompt_text, prompt_speech)
- 输出：
    - 若干生成的 WAV 文件：utt_0000000.wav、utt_0000001.wav、...
    - 一个 metadata.jsonl，记录：
        - id / index / text
        - ref_audio_path（参考音频路径）
        - audio_path（生成音频路径）
        - sample_rate / num_samples / duration_sec / gen_time_sec

备注：
- 单 GPU 脚本，多线程用于 I/O 和调度；模型本身只初始化一次。
- 为了线程安全，推理段加了一个锁（默认 instances>1 也不会在模型内部并行），
  真要极限吃满多 GPU，建议上多进程 + CUDA_VISIBLE_DEVICES。
"""

import argparse
import os
import sys
import time
import json
import random
import queue
import inspect
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
import swanlab  # new

# 放在全局位置（比如和 counter_lock 类似的级别）
SWAN_LOG_LOCK = Lock()
import torch
import torchaudio

# -----------------------------------------------------------------------------
# 性能相关的一些小优化（有就用，没有就算了）
# -----------------------------------------------------------------------------
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "third_party" / "Matcha-TTS"))

# 尝试注册 vLLM 模型（如果环境里有的话）
try:
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    try:
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
        print("[vLLM] ModelRegistry registered CosyVoice2ForCausalLM")
    except Exception:
        pass
except Exception:
    pass

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def parse_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y"}


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int):
    wf = waveform.detach().to("cpu", non_blocking=True)
    safe_mkdir(path.parent)
    torchaudio.save(str(path), wf, sample_rate)


# def sample_items_from_jsonl_and_wav(
#     jsonl_path: Path,
#     wav_dir: Path,
#     num_samples: int = -1,
#     seed: int | None = None,
# ):
#     """
#     从 jsonl 和 wav_dir 中随机抽取 num_samples 个 pair：
#       (idx, text, ref_audio_path)

#     设计原则：不把全部 jsonl 行读进内存，只做“两遍扫描”：
#       1. 第一遍：只数行 -> total_lines
#       2. 计算可匹配的 pair 数：max_pairs = min(total_lines, len(wav_files))
#       3. 从 [0, max_pairs) 中随机 sample K 个索引（K = num_samples 或 max_pairs）
#       4. 第二遍：顺序扫 jsonl，仅在命中的行上解析 / 提取文本，组装 items

#     JSON 结构假设（示例）：
#     {
#       "conversation": [
#         {
#           "role": "user",
#           "content": [ ... ]
#         },
#         {
#           "role": "user",
#           "content": [
#             { "type": "text", "text": "这里是要做 TTS 的文本" }
#           ]
#         },
#         ...
#       ]
#     }
#     """
#     if not jsonl_path.is_file():
#         raise RuntimeError(f"[Data] jsonl file not found: {jsonl_path}")
#     if not wav_dir.is_dir():
#         raise RuntimeError(f"[Data] wav-dir not found: {wav_dir}")

#     wav_files = sorted(wav_dir.glob("*.wav"))

#     # 1) 数行
#     print(f"[Data] Counting lines in {jsonl_path} ...")
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         total_lines = sum(1 for _ in f)
#     print(f"[Data] jsonl total_lines = {total_lines}, wav_files = {len(wav_files)}")

#     max_pairs = min(total_lines, len(wav_files))
#     if max_pairs == 0:
#         raise RuntimeError("[Data] No valid pairs: jsonl has 0 lines or wav_dir is empty.")

#     # 2) 确定抽样数量
#     if num_samples < 0 or num_samples > max_pairs:
#         num_samples = max_pairs
#     print(f"[Data] Will sample {num_samples} pairs out of {max_pairs}.")

#     # 3) 随机索引
#     if seed is not None:
#         random.seed(seed)
#     selected_indices = sorted(random.sample(range(max_pairs), num_samples))

#     # 4) 第二遍：只解析命中的行
#     items = []
#     current_ptr = 0
#     target_idx = selected_indices[current_ptr]

#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line_idx, line in enumerate(f):
#             if line_idx > selected_indices[-1]:
#                 break
#             if line_idx != target_idx:
#                 continue

#             line = line.strip()
#             if not line:
#                 print(f"[Data][WARN] line {line_idx}: empty, skip.")
#             else:
#                 obj = None
#                 try:
#                     obj = json.loads(line)
#                 except Exception as e:
#                     print(f"[Data][WARN] line {line_idx}: json decode failed: {e}")

#                 if obj is not None:
#                     conv = obj.get("conversation", [])
#                     user_turn = None
#                     for turn in conv:
#                         if turn.get("role") == "user":
#                             user_turn = turn
#                             break
#                     text = None
#                     if user_turn is not None:
#                         for c in user_turn.get("content", []):
#                             if c.get("type") == "text":
#                                 text = c.get("text", "").strip()
#                                 if text:
#                                     break
#                     if not text:
#                         print(f"[Data][WARN] line {line_idx}: no user text, skip this index.")
#                     else:
#                         ref_audio_path = wav_files[line_idx]
#                         items.append((line_idx, text, str(ref_audio_path)))

#             # 准备下一个 target_idx
#             current_ptr += 1
#             if current_ptr >= len(selected_indices):
#                 break
#             target_idx = selected_indices[current_ptr]

#     print(f"[Data] Actually loaded {len(items)} sampled pairs.")
#     return items

def sample_items_from_jsonl_and_wav(
    jsonl_path: Path,
    wav_dir: Path,
    num_samples: int = -1,
    seed: int | None = None,
):
    """
    从 jsonl 和 wav_dir 中随机抽取 num_samples 个 pair：
      (idx, text, ref_audio_path)

    逻辑：
      1. 先统计 jsonl 行数 total_lines
      2. 根据 total_lines 和 wav 数量决定最大可配对数 max_pairs
      3. 在 [0, max_pairs) 里随机 sample 若干行号
      4. 第二遍扫描 jsonl，只解析这些被抽到的行：
         - 如果该行的 output_language != "English" 就跳过
         - 否则取 user 文本 + 对应行号的 wav 路径
    注意：
      - 为了省时间和内存，这里只解析“被抽中的行”，不会把全量 jsonl 读进内存。
      - 如果抽中的行里英文样本不够多，最后返回的 items 数量会 < num_samples。
    """
    if not jsonl_path.is_file():
        raise RuntimeError(f"[Data] jsonl file not found: {jsonl_path}")
    if not wav_dir.is_dir():
        raise RuntimeError(f"[Data] wav-dir not found: {wav_dir}")

    wav_files = sorted(wav_dir.glob("*.wav"))

    # 1) 数行
    print(f"[Data] Counting lines in {jsonl_path} ...")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"[Data] jsonl total_lines = {total_lines}, wav_files = {len(wav_files)}")

    max_pairs = min(total_lines, len(wav_files))
    if max_pairs == 0:
        raise RuntimeError("[Data] No valid pairs: jsonl has 0 lines or wav_dir is empty.")

    # 2) 确定抽样数量
    if num_samples < 0 or num_samples > max_pairs:
        num_samples = max_pairs
    print(f"[Data] Will sample {num_samples} indices out of {max_pairs} total pairs (before language filtering).")

    # 3) 随机索引
    if seed is not None:
        random.seed(seed)
    selected_indices = sorted(random.sample(range(max_pairs), num_samples))

    # 4) 第二遍：只解析命中的行 + 过滤 output_language
    items = []
    current_ptr = 0
    target_idx = selected_indices[current_ptr]

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx > selected_indices[-1]:
                break
            if line_idx != target_idx:
                continue

            line = line.strip()
            if not line:
                print(f"[Data][WARN] line {line_idx}: empty, skip.")
            else:
                obj = None
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"[Data][WARN] line {line_idx}: json decode failed: {e}")

                if obj is not None:
                    # ---- 语言过滤：只要 output_language == "English" ----
                    out_lang = str(obj.get("output_language", "")).strip().lower()
                    if out_lang != "english":
                        # 你给的样例是 "Chinese"；这里统一只要不是 English 就跳过
                        print(f"[Data][LANG] line {line_idx}: output_language='{out_lang}', skip.")
                    else:
                        # 从 conversation 里找 user 文本
                        conv = obj.get("conversation", [])
                        user_turn = None
                        for turn in conv:
                            if turn.get("role") == "user":
                                user_turn = turn
                                break

                        text = None
                        if user_turn is not None:
                            for c in user_turn.get("content", []):
                                if c.get("type") == "text":
                                    text = c.get("text", "").strip()
                                    if text:
                                        break

                        if not text:
                            print(f"[Data][WARN] line {line_idx}: no user text, skip this index.")
                        else:
                            ref_audio_path = wav_files[line_idx]
                            items.append((line_idx, text, str(ref_audio_path)))

            # 准备下一个 target_idx
            current_ptr += 1
            if current_ptr >= len(selected_indices):
                break
            target_idx = selected_indices[current_ptr]

    print(
        f"[Data] Actually loaded {len(items)} sampled pairs "
        f"after filtering output_language=='English'."
    )
    return items
# -----------------------------------------------------------------------------
# CosyVoice2 初始化 & worker 线程
# -----------------------------------------------------------------------------
def init_cosyvoice(args):
    """
    根据命令行参数创建一个 CosyVoice2 实例，带 vLLM / JIT / TRT 这些开关。
    """
    vllm_kwargs = dict(
        tensor_parallel_size=args.vllm_tp,
        gpu_memory_utilization=args.vllm_gpu_mem,
        max_model_len=args.vllm_max_len,
    )
    print(f"[Init] vLLM kwargs = {vllm_kwargs}")

    sig = inspect.signature(CosyVoice2.__init__)
    can_split = all(k in sig.parameters for k in ["vllm_gpu_mem", "vllm_max_len", "vllm_tp"])

    try:
        if can_split:
            print("[Init] This CosyVoice2 version exposes vLLM config; applying user settings.")
            cosyvoice = CosyVoice2(
                args.model_dir,
                load_jit=args.jit,
                load_trt=args.trt,
                load_vllm=args.vllm,
                fp16=args.fp16,
                vllm_gpu_mem=args.vllm_gpu_mem,
                vllm_max_len=args.vllm_max_len,
                vllm_tp=args.vllm_tp,
                vllm_enforce_eager=False,
            )
        else:
            print("[Init][WARN] This CosyVoice2 version exposes no vLLM config; using library defaults.")
            cosyvoice = CosyVoice2(
                args.model_dir,
                load_jit=args.jit,
                load_trt=args.trt,
                load_vllm=args.vllm,
                fp16=args.fp16,
                vllm_enforce_eager=False,
            )
    except TypeError as e:
        print("[Init][ERROR] Unexpected CosyVoice2 signature; falling back without vLLM tuning:", e)
        cosyvoice = CosyVoice2(
            args.model_dir,
            load_jit=args.jit,
            load_trt=args.trt,
            load_vllm=args.vllm,
            fp16=args.fp16,
        )

    print("[Init] CosyVoice2 model ready.")
    return cosyvoice


def worker_thread(
    worker_id: int,
    args,
    cosyvoice,
    model_lock: Lock,
    task_queue: "queue.Queue",
    io_pool: ThreadPoolExecutor,
    counter,
    counter_lock: Lock,
    futures_list,
    metadata_list,
    meta_lock: Lock,
    warmup_ref_audio_path: str | None = None,
):
    """
    单个 worker 线程：
    - 从队列里拿 (idx, text, out_path, ref_audio_path)
    - 读取 ref_audio_path -> prompt_speech
    - 调用 inference_zero_shot 做 TTS
    - 将结果写入磁盘 + metadata_list
    """

    # warmup：用一条参考音频 + 固定句子做缓存预热（可选）
    if args.warmup > 0 and warmup_ref_audio_path is not None:
        warmup_text = args.warmup_text or "你好，世界。"
        try:
            prompt_speech = load_wav(warmup_ref_audio_path, 16000)
            for _ in range(args.warmup):
                try:
                    # with model_lock:  # 保护模型调用的线程安全
                    try:
                        gen = cosyvoice.inference_zero_shot(
                            warmup_text,
                            args.prompt_text,
                            prompt_speech,
                            stream=False,
                            text_frontend=args.text_frontend,
                        )
                    except TypeError:
                        gen = cosyvoice.inference_zero_shot(
                            warmup_text,
                            args.prompt_text,
                            prompt_speech,
                            stream=False,
                        )
                    _ = list(gen)
                except Exception as e:
                    print(f"[Warmup] worker#{worker_id} warmup failed: {e}")
                    break
            print(f"[Warmup] worker#{worker_id} done.")
        except Exception as e:
            print(f"[Warmup] worker#{worker_id} failed to load warmup ref audio: {e}")

    with torch.inference_mode():
        while True:
            try:
                item = task_queue.get(timeout=2.0)
            except queue.Empty:
                break
            if item is None:
                break

            idx, text, out_path, ref_audio_path = item
            try:
                t0 = time.time()

                # 每条样本自己的 zero-shot 参考音频
                prompt_speech = load_wav(ref_audio_path, 16000)

                # 调用 zero-shot TTS（不使用 spkinfo）
                # with model_lock:
                try:
                    gen = cosyvoice.inference_zero_shot(
                        text,
                        args.prompt_text,
                        prompt_speech,
                        stream=args.stream,
                        text_frontend=args.text_frontend,
                    )
                except TypeError:
                    gen = cosyvoice.inference_zero_shot(
                        text,
                        args.prompt_text,
                        prompt_speech,
                        stream=args.stream,
                    )

                if args.stream:
                    chunks = []
                    sr = cosyvoice.sample_rate
                    for pack in gen:
                        chunks.append(pack["tts_speech"].detach().cpu())
                    if not chunks:
                        task_queue.task_done()
                        continue
                    audio = torch.cat(chunks, dim=-1)
                else:
                    packs = list(gen)
                    if not packs:
                        task_queue.task_done()
                        continue
                    audio = packs[-1]["tts_speech"].detach().cpu()
                    sr = cosyvoice.sample_rate

                gen_time = time.time() - t0
                num_samples = audio.shape[-1]
                duration_sec = float(num_samples) / float(sr)

                utt_id = f"{args.prefix}{idx:07d}{args.suffix}"

                meta = {
                    "id": utt_id,
                    "index": idx,
                    "text": text,
                    "ref_audio_path": ref_audio_path,
                    "audio_path": str(out_path),
                    "sample_rate": int(sr),
                    "num_samples": int(num_samples),
                    "duration_sec": duration_sec,
                    "gen_time_sec": float(gen_time),
                }

                with meta_lock:
                    metadata_list.append(meta)

                fut = io_pool.submit(save_audio, out_path, audio, sr)
                futures_list.append(fut)

                with counter_lock:
                    counter["done"] += 1
                    d = counter["done"]
                    if d % 50 == 0:
                        print(f"[Prog] worker#{worker_id} -> {d} utterances queued for save")
                # 额外采集一次当前显存使用（可选）
                gpu_mem_mb = None
                if torch.cuda.is_available():
                    gpu_mem_mb = torch.cuda.memory_allocated() / 1024**2

                # SwanLab 日志：一条样本一个点
                with SWAN_LOG_LOCK:
                    metrics = {
                        "tts/gen_time_sec": float(gen_time),
                        "tts/duration_sec": float(duration_sec),
                        "tts/real_time_factor": float(duration_sec / gen_time) if gen_time > 0 else None,
                    }
                    if gpu_mem_mb is not None:
                        metrics["gpu/memory_allocated_mb"] = float(gpu_mem_mb)
                    # 你也可以顺手 log 一下 worker_id 或 idx 方便排查
                    metrics["meta/worker_id"] = worker_id
                    metrics["meta/index"] = idx

                    swanlab.log(metrics, step=d)
            except Exception as e:
                print(f"[Error] worker#{worker_id} failed on index {idx}: {e}")
            finally:
                task_queue.task_done()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Batch zero-shot TTS for CosyVoice2 using JSONL + WAV with random sampling."
    )
    p.add_argument("--model-dir", type=str, required=True, help="Path to CosyVoice2 model dir.")

    # 数据来源：JSONL + WAV 目录
    p.add_argument("--jsonl", type=str, required=True,
                   help="JSONL file where each line contains a conversation with user text.")
    p.add_argument("--wav-dir", type=str, required=True,
                   help="Directory containing reference wavs sorted by name, aligned by line index.")

    p.add_argument("--outdir", type=str, required=True,
                   help="Output directory for generated wavs and metadata.jsonl.")

    # 抽样控制
    p.add_argument("--num-samples", type=int, default=-1,
                   help="How many random pairs to sample from JSONL+wav. -1 means use all available.")
    p.add_argument("--random-seed", type=int, default=42,
                   help="Random seed for sampling from JSONL.")

    # TTS 相关
    p.add_argument("--prompt-text", type=str, default="",
                   help="Prompt text to condition the voice style / context (second arg of inference_zero_shot).")
    p.add_argument("--prefix", type=str, default="utt_",
                   help="Prefix for output wav filenames.")
    p.add_argument("--suffix", type=str, default=".wav",
                   help="Suffix/extension for output wav filenames.")

    # 运行参数
    p.add_argument("--resume", action="store_true",
                   help="Skip already existing output wavs (by index).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing wavs when resume is enabled.")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of I/O worker threads for saving audio.")
    p.add_argument("--instances", type=int, default=1,
                   help="Number of worker threads consuming the task queue (model is shared).")

    # CosyVoice / vLLM 开关
    p.add_argument("--fp16", type=parse_bool, default=True)
    p.add_argument("--text-frontend", type=parse_bool, default=True)
    p.add_argument("--stream", type=parse_bool, default=False)
    p.add_argument("--vllm", type=parse_bool, default=True)
    p.add_argument("--trt", type=parse_bool, default=False)
    p.add_argument("--jit", type=parse_bool, default=False)
    p.add_argument("--vllm-tp", type=int, default=1)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.9)
    p.add_argument("--vllm-max-len", type=int, default=8192)

    # warmup
    p.add_argument("--warmup", type=int, default=0,
                   help="Number of warmup runs per worker using one reference wav (0 to disable).")
    p.add_argument("--warmup-text", type=str, default="",
                   help="Warmup text, if empty will use '你好，世界。'")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    if device.type != "cuda":
        print("[Warning] CUDA not available. This script is designed for GPU use; CPU will be very slow.")
    # ---------------- SwanLab init ----------------
    run = swanlab.init(
        project="cv-htl",                      # 自己起个项目名
        workspace="tohkasensei",
        experiment_name=time.strftime("run-%Y%m%d-%H%M%S"),
        config={
            "model_dir": args.model_dir,
            "num_samples": args.num_samples,
            "instances": args.instances,
            "vllm": bool(args.vllm),
            "vllm_gpu_mem": args.vllm_gpu_mem,
        },
        # 如果想完全离线看板：
        # mode="local",
        # logdir="./swanlog_cosyvoice",
    )
    if torch.cuda.is_available():
        dev = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(dev)
        swanlab.log({
            "hardware/gpu_index": dev,
            "hardware/gpu_name": props.name,
            "hardware/gpu_total_mem_GB": props.total_memory / 1024**3,
        })
    else:
        swanlab.log({"hardware/device": "cpu"})
    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    jsonl_path = Path(args.jsonl)
    wav_dir = Path(args.wav_dir)

    # 1) 从 jsonl + wav-dir 中“随机抽样”若干 pair
    items = sample_items_from_jsonl_and_wav(
        jsonl_path=jsonl_path,
        wav_dir=wav_dir,
        num_samples=args.num_samples,
        seed=args.random_seed,
    )
    if not items:
        print("[Data] No items sampled; exiting.")
        return

    # 选第一条样本的 ref_audio 作为 warmup 的参考音频
    warmup_ref_audio_path = items[0][2] if args.warmup > 0 else None

    # 2) 准备队列、线程池、计数器、metadata 容器
    q = queue.Queue(maxsize=4 * max(1, args.instances))
    io_pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = []

    counter = {"done": 0}
    counter_lock = Lock()
    metadata_list = []
    meta_lock = Lock()

    # 模型锁：保护 inference_zero_shot 的并发安全
    model_lock = Lock()

    # 3) 初始化模型（只初始化一次）
    cosyvoice = init_cosyvoice(args)

    # 4) 启动 worker 线程
    instances = max(1, int(args.instances))
    threads = []
    t0 = time.time()
    for wid in range(instances):
        th = Thread(
            target=worker_thread,
            args=(
                wid,
                args,
                cosyvoice,
                model_lock,
                q,
                io_pool,
                counter,
                counter_lock,
                futures,
                metadata_list,
                meta_lock,
                warmup_ref_audio_path,
            ),
            daemon=True,
        )
        th.start()
        threads.append(th)

    # 5) 将任务放入队列
    total_tasks = 0
    for idx, text, ref_audio_path in items:
        if not text:
            continue
        out_path = outdir / f"{args.prefix}{idx:07d}{args.suffix}"
        if args.resume and out_path.exists() and not args.overwrite:
            continue
        q.put((idx, text, out_path, ref_audio_path))
        total_tasks += 1

    print(f"[Main] Enqueued {total_tasks} tasks.")

    # 6) 等待队列完成 & 线程退出
    q.join()
    for _ in range(instances):
        q.put(None)
    for th in threads:
        th.join()

    for fut in as_completed(futures):
        _ = fut.result()
    io_pool.shutdown(wait=True)

    elapsed = time.time() - t0
    utts = counter["done"]
    print(
        f"[Done] Wrote {utts} / {total_tasks} utterances to {str(outdir)} "
        f"in {elapsed/60:.2f} min "
        f"({(utts / elapsed) if elapsed > 0 else 0:.2f} utt/s)."
    )

    # 7) 写 metadata.jsonl
    meta_path = outdir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadata_list:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[Meta] Saved metadata for {len(metadata_list)} utterances to {meta_path}")


if __name__ == "__main__":
    main()