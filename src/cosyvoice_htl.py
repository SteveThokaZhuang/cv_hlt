
"""
Batch TTS for CosyVoice2 — single GPU, no text splitting, multi-instance, vLLM/TRT/JIT ready
First run saves spkinfo; later runs can skip prompt audio and use spkinfo only.
"""
import argparse, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Thread, Lock
import queue
import inspect
import torch, torchaudio

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
sys.path.append(f"{str(ROOT_DIR)}/third_party/Matcha-TTS")

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
import os, json, glob
import torch
from pathlib import Path

def load_spkinfo_into(cosy, model_dir: str):
    """
    从磁盘加载 <model_dir>/spk2info.pt（若存在）并注入到当前 CosyVoice2 实例。
    返回可用的 speaker id 列表。
    """
    path = Path(model_dir) / "spk2info.pt"
    keys = []
    if path.is_file():
        try:
            d = torch.load(str(path), map_location="cpu")
            # 典型结构：dict[spk_id] = {...info...}
            if hasattr(cosy, "frontend") and hasattr(cosy.frontend, "spk2info"):
                cosy.frontend.spk2info = d
            # 兼容某些版本用 cosy.spk_info 访问
            try:
                cosy.spk_info = d
            except Exception:
                pass
            keys = list(d.keys()) if isinstance(d, dict) else []
            print(f"[SPK] Loaded {len(keys)} speakers from {path}")
        except Exception as e:
            print(f"[SPK][WARN] Failed to load {path}: {e}")
    else:
        print(f"[SPK] No spk2info.pt at {path} (will create on first save)")
    return keys

def _has_spk_id_any(cosy, spk_id: str, model_dir: str) -> bool:
    # 先看内存对象
    for attr in ["spk_info", "spk2info", "speaker_dict"]:
        d = getattr(cosy, attr, None)
        if isinstance(d, dict) and (spk_id in d):
            return True
    # 再看磁盘（spk2info.pt）
    path = Path(model_dir) / "spk2info.pt"
    if path.is_file():
        try:
            d = torch.load(str(path), map_location="cpu")
            if isinstance(d, dict) and (spk_id in d):
                return True
        except Exception:
            pass
    return False

def parse_bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_lines(path: Path, start: int, end: int):
    with open(path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            if idx < start: continue
            if end >= 0 and idx >= end: break
            line = raw.strip()
            yield idx, line

def save_audio(path: Path, waveform: torch.Tensor, sample_rate: int):
    wf = waveform.detach().to("cpu", non_blocking=True)
    safe_mkdir(path.parent)
    torchaudio.save(str(path), wf, sample_rate)

# ---------- spkinfo helpers ----------
def _has_spk_id(cosyvoice_obj, spk_id: str) -> bool:
    try:
        spk_info = getattr(cosyvoice_obj, "spk_info", None)
        return isinstance(spk_info, dict) and (spk_id in spk_info)
    except Exception:
        return False


def ensure_spkinfo_on_disk(model_dir: str, spk_id: str, prompt_text: str, prompt_wav_path: str,
                           fp16: bool = True) -> None:
    print(f"[SPK] Ensuring spkinfo for id='{spk_id}' exists on disk...")
    cosy = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=fp16)

    # 关键：先把磁盘里的 spk2info.pt 注入到实例
    _ = load_spkinfo_into(cosy, model_dir)

    # 再判断是否已存在该 spk_id
    if _has_spk_id_any(cosy, spk_id, model_dir):
        print(f"[SPK] spk_id='{spk_id}' already present. No need to add.")
        return

    # 真缺才需要首跑的 prompt-wav
    if not prompt_wav_path:
        raise RuntimeError(f"[SPK] Missing --prompt-wav for first-time registration of spk_id='{spk_id}'.")

    prompt_speech = load_wav(prompt_wav_path, 16000)
    ok = cosy.add_zero_shot_spk(prompt_text or "", prompt_speech, spk_id)
    if not ok:
        raise RuntimeError(f"[SPK] add_zero_shot_spk failed for id='{spk_id}'.")
    cosy.save_spkinfo()
    print(f"[SPK] Saved spkinfo for id='{spk_id}' to model_dir.")
# ------------------------------------
def init_cosyvoice(args, prompt_speech):
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
                load_jit=args.jit, load_trt=args.trt, load_vllm=args.vllm, fp16=args.fp16,
                vllm_gpu_mem=args.vllm_gpu_mem,
                vllm_max_len=args.vllm_max_len,
                vllm_tp=args.vllm_tp,
                vllm_enforce_eager=False
            )
        else:
            print("[Init][WARN] This CosyVoice2 version exposes no vLLM config; using library defaults.")
            cosyvoice = CosyVoice2(
                args.model_dir,
                load_jit=args.jit, load_trt=args.trt, load_vllm=args.vllm, fp16=args.fp16,
                vllm_enforce_eager=False
            )
    except TypeError as e:
        print("[Init][ERROR] Unexpected CosyVoice2 signature; falling back without vLLM tuning:", e)
        cosyvoice = CosyVoice2(
            args.model_dir,
            load_jit=args.jit, load_trt=args.trt, load_vllm=args.vllm, fp16=args.fp16
        )
    # 确保推理用到最新 spk2info.pt
    load_spkinfo_into(cosyvoice, args.model_dir)
    # print(f"[Init] worker#{worker_id} model ready.")
    return cosyvoice
def worker_thread(worker_id: int, args, cosyvoice, prompt_speech, task_queue: "queue.Queue",
                  io_pool: ThreadPoolExecutor, counter, counter_lock, futures_list,
                  metadata_list, meta_lock):
    
    # warmup
    if args.warmup > 0:
        warmup_text = args.warmup_text or "你好，世界。"
        for _ in range(args.warmup):
            try:
                if args.use_spkinfo:
                    try:
                        gen = cosyvoice.inference_zero_shot(
                            warmup_text, "", "",
                            zero_shot_spk_id=args.spk_id,
                            stream=False, text_frontend=args.text_frontend
                        )
                    except TypeError:
                        gen = cosyvoice.inference_zero_shot(
                            warmup_text, "", "",
                            zero_shot_spk_id=args.spk_id,
                            stream=False
                        )
                else:
                    try:
                        gen = cosyvoice.inference_zero_shot(
                            warmup_text, args.prompt_text, prompt_speech,
                            stream=False, text_frontend=args.text_frontend
                        )
                    except TypeError:
                        gen = cosyvoice.inference_zero_shot(
                            warmup_text, args.prompt_text, prompt_speech,
                            stream=False
                        )
                _ = list(gen)
            except Exception as e:
                print(f"[Warmup] worker#{worker_id} warmup failed: {e}")
                break
        print(f"[Warmup] worker#{worker_id} done.")

    with torch.inference_mode():
        while True:
            try:
                item = task_queue.get(timeout=2.0)
            except queue.Empty:
                break
            if item is None:
                break
            idx, text, out_path = item
            try:
                t0 = time.time() # record individual speech processing start time
                # inference branch
                if args.use_spkinfo:
                    try:
                        gen = cosyvoice.inference_zero_shot(
                            text, "", "",
                            zero_shot_spk_id=args.spk_id,
                            stream=args.stream, text_frontend=args.text_frontend
                        )
                    except TypeError:
                        gen = cosyvoice.inference_zero_shot(
                            text, "", "",
                            zero_shot_spk_id=args.spk_id,
                            stream=args.stream
                        )
                else:
                    try:
                        gen = cosyvoice.inference_zero_shot(
                            text, args.prompt_text, prompt_speech,
                            stream=args.stream, text_frontend=args.text_frontend
                        )
                    except TypeError:
                        gen = cosyvoice.inference_zero_shot(
                            text, args.prompt_text, prompt_speech,
                            stream=args.stream
                        )

                if args.stream:
                    chunks = []
                    sr = cosyvoice.sample_rate
                    for pack in gen:
                        chunks.append(pack["tts_speech"].detach().cpu())
                    if not chunks:
                        continue
                    audio = torch.cat(chunks, dim=-1)
                else:
                    packs = list(gen)
                    if not packs:
                        continue
                    audio = packs[-1]["tts_speech"].detach().cpu()
                    sr = cosyvoice.sample_rate
                # record metadata
                gen_time = time.time() - t0 # time taken for this utterance
                num_samples = audio.shape[-1]
                duration_sec = float(num_samples) / float(sr)

                utt_id = f"{args.prefix}{idx}{args.suffix}" # utt_id
                # utt_id = f"{args.prefix}{idx:07d}"

                speaker_id = args.spk_id if args.use_spkinfo else "prompt_speech"
                meta = {
                    "id": utt_id,
                    "index": idx,
                    "text": text,
                    "speaker_id": speaker_id,
                    "audio_path": str(out_path),
                    "sample_rate": int(sr),
                    "num_samples": int(num_samples),
                    "duration_sec": duration_sec,
                    "gen_time_sec": float(gen_time),
                }

                # Collect meta data
                with meta_lock:
                    metadata_list.append(meta)

                fut = io_pool.submit(save_audio, out_path, audio, sr)
                futures_list.append(fut)

                with counter_lock:
                    counter["done"] += 1
                    d = counter["done"]
                    if d % 50 == 0:
                        print(f"[Prog] worker#{worker_id} -> {d} utterances queued for save")

            except Exception as e:
                print(f"[Error] worker#{worker_id} failed on line {idx}: {e}")
            finally:
                task_queue.task_done()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True)
    # 首跑注册 spkinfo 用；后续可省略
    p.add_argument("--prompt-wav", type=str, default="")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--prompt-text", type=str, default="")
    p.add_argument("--prefix", type=str, default="utt_")
    p.add_argument("--suffix", type=str, default=".wav")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--instances", type=int, default=2)
    p.add_argument("--fp16", type=parse_bool, default=True)
    p.add_argument("--text-frontend", type=parse_bool, default=True)
    p.add_argument("--stream", type=parse_bool, default=False)
    p.add_argument("--vllm", type=parse_bool, default=True)
    p.add_argument("--trt", type=parse_bool, default=False)
    p.add_argument("--jit", type=parse_bool, default=False)
    p.add_argument("--vllm-tp", type=int, default=1)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.9)
    p.add_argument("--vllm-max-len", type=int, default=8192)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--warmup-text", type=str, default="")
    # spkinfo
    p.add_argument("--use-spkinfo", type=parse_bool, default=True,
                   help="If true, use zero_shot_spk_id for inference; prompt-wav is only needed once to bootstrap spkinfo.")
    p.add_argument("--spk-id", type=str, default="my_zero_shot_spk",
                   help="Speaker id for spkinfo registry and later reuse.")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    if device.type != "cuda":
        print("[Warning] CUDA not available. This script is designed for single-GPU use.")

    outdir = Path(args.outdir); safe_mkdir(outdir)

    # 首跑：确保 spkinfo 已写盘（无则用 prompt-wav 抽取并保存）
    if args.use_spkinfo:
        ensure_spkinfo_on_disk(
            model_dir=args.model_dir,
            spk_id=args.spk_id,
            prompt_text=args.prompt_text,
            prompt_wav_path=args.prompt_wav,
            fp16=args.fp16
        )

    # 仅在不用 spkinfo 的模式下加载 prompt_speech
    prompt_speech = None
    if not args.use_spkinfo:
        if not args.prompt_wav:
            raise RuntimeError("[Arg] --use-spkinfo=false requires --prompt-wav for every run.")
        prompt_speech = load_wav(args.prompt_wav, 16000)

    q = queue.Queue(maxsize=4 * max(1, args.instances))
    io_pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    futures = []
    counter, counter_lock = {"done": 0}, Lock()
    # metadata collection
    metadata_list = []
    meta_lock = Lock()
    instances = max(1, int(args.instances))
    cosyvoice = init_cosyvoice(args, prompt_speech)
    threads = []
    t0 = time.time()
    for wid in range(instances):
        th = Thread(target=worker_thread,
                    args=(wid, args, cosyvoice, prompt_speech, q, io_pool, counter, counter_lock, futures,
                    metadata_list, meta_lock),
                    daemon=True)
        th.start()
        threads.append(th)

    total_tasks = 0
    for idx, text in read_lines(Path(args.input), args.start, args.end):
        if not text: continue
        out_path = outdir / f"{args.prefix}{idx:07d}{args.suffix}"
        if args.resume and out_path.exists() and not args.overwrite: continue
        q.put((idx, text, out_path))
        total_tasks += 1

    q.join()
    for _ in range(instances): q.put(None)
    for th in threads: th.join()

    for fut in as_completed(futures): _ = fut.result()
    io_pool.shutdown(wait=True)

    elapsed = time.time() - t0
    print(f"[Done] Wrote {counter['done']} / {total_tasks} utterances to {str(outdir)} in {elapsed/60:.2f} min "
          f"({(counter['done'] / elapsed) if elapsed > 0 else 0:.2f} utt/s).")
    # write down metadata.jsonl
    meta_path = outdir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadata_list:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[Meta] Saved metadata for {len(metadata_list)} utterances to {meta_path}")


if __name__ == "__main__":
    main()