import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def _clean_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(errors="ignore")

def _to_gb(val: float, unit: str) -> float:
    u = unit.upper()
    if u in ("GB", "GIB"):
        return val
    if u in ("MB", "MIB"):
        return val / 1024.0
    if u in ("KB", "KIB"):
        return val / (1024.0 * 1024.0)
    return val

def _grab_last_float(pattern: str, text: str) -> Optional[float]:
    m = re.findall(pattern, text)
    if not m:
        return None
    try:
        return round(float(m[-1]), 2)
    except Exception:
        return None

def _clean_cuda_visible_devices(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    # берём числа только до конца строки или до пробела/запятой
    line = raw.strip().split("\n")[0]
    nums = re.findall(r"\b\d+\b", line)
    return ", ".join(nums) if nums else None

def _pick_parallel_mode(deepspeed_stage: Optional[int], fsdp_string: Optional[str]) -> Optional[str]:
    if deepspeed_stage is not None:
        return f"DeepSpeed (ZeRO-{deepspeed_stage})"
    if fsdp_string:
        return f"FSDP {fsdp_string}"
    return None

def _precision_str(bf16_flag: Optional[str], fp32_flag: Optional[str]) -> str:
    # ожидаем "True"/"False" или None
    bf16_on = (bf16_flag or "").lower() == "true"
    fp32_on = (fp32_flag or "").lower() == "true"
    if bf16_on:
        return "bf16"
    if fp32_on:
        return "fp32"
    return None

def parse_one_log(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"File name": path.name, "Path": str(path)}
    text = _clean_ansi(_read_text(path))
    lines = text.splitlines()

    # --- Run / W&B ------------------------------------------------------------
    m = re.search(r"\brun[_ ]?name[=:]\s*([^\n]+)", text, flags=re.IGNORECASE)
    out["Run name"] = m.group(1).strip() if m else None
    m = re.search(r"(https://wandb\.ai/[^\s]+)", text)
    out["W&B URL"] = m.group(1) if m else None

    # --- CUDA_VISIBLE_DEVICES -------------------------------------------------
    m = re.search(r"CUDA_VISIBLE_DEVICES\s*[:=]\s*([0-9,\s^\n]+)", text)
    out["CUDA visible devices"] = _clean_cuda_visible_devices(m.group(1)) if m else None

    # --- World size (несколько вариантов) ------------------------------------
    ws = None
    for pat in [
        r"\bWorld size\s*[:=]\s*(\d+)",
        r"\bdistributed_world_size[)=: ]+(\d+)",
        r"\bnproc_per_node[)=: ]+(\d+)",
        r"\bnum_processes[)=: ]+(\d+)",
        r"\bWORLD_SIZE\s*[:=]\s*(\d+)",
    ]:
        m = re.search(pat, text)
        if m:
            ws = int(m.group(1))
            break
    out["World size"] = ws

    # --- Batch / GA -----------------------------------------------------------
    m = re.search(r"\bper_device_train_batch_size[)=: ]+(\d+)", text)
    per_device = int(m.group(1)) if m else None
    out["Per-device batch size"] = int(per_device) if per_device is not None else None

    m = re.search(r"\bgradient_accumulation_steps[)=: ]+(\d+)", text)
    ga = int(m.group(1)) if m else 1
    out["Gradient accumulation"] = ga

    # --- DeepSpeed stage / FSDP string ---------------------------------------
    m = re.search(r"zero_optimization[^\n]*stage[\"']?\s*[:=]\s*(\d)", text)
    deepspeed_stage = int(m.group(1)) if m else None

    m = re.search(r"FSDP:\s*(.+)$", text, flags=re.MULTILINE)
    fsdp_string = m.group(1).strip() if m else None

    out["Parallel mode"] = _pick_parallel_mode(deepspeed_stage, fsdp_string)

    # --- Precision flags ------------------------------------------------------
    m = re.search(r"\bbf16\s*=\s*(True|False)", text)
    bf16 = m.group(1) if m else None
    m = re.search(r"\bfp32\s*=\s*(True|False)", text)
    fp32 = m.group(1) if m else None
    out["Precision"] = _precision_str(bf16, fp32)

    # --- Speed: собираем s/it (предпочтительно), иначе it/s -------------------
    EVAL_START_PAT = re.compile(r"(Финальная валидация|Running final evaluation|Final evaluation|Start eval|Evaluation)", re.IGNORECASE)
    cut_idx = None
    for i, l in enumerate(lines):
        if EVAL_START_PAT.search(l):
            cut_idx = i
            break
    train_lines = lines[:cut_idx] if cut_idx is not None else lines
    s_per_it_vals: List[float] = []
    # it_per_s_vals: List[float] = []
    for l in train_lines:
        ms = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*s/it", l)
        for x in ms:
            try:
                s_per_it_vals.append(float(x))
            except Exception:
                pass
        mi = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*it/s", l)
        # for x in mi:
        #     try:
        #         it_per_s_vals.append(float(x))
        #     except Exception:
        #         pass
    
    out["Train runtime"] = _grab_last_float(r"\btrain_runtime[\"']?\s*[:=]\s*([0-9\.\-eE]+)", text)
    out["Eval loss"] = _grab_last_float(r"\beval_loss[\"']?\s*[:=]\s*([0-9\.\-eE]+)", text)

    # медиана по s/it, затем превращаем в it/s
    iter_per_s = None
    steps_detected = None
    if s_per_it_vals:
        med_s_per_it = float(np.median(sorted(s_per_it_vals)))
        # iter_per_s = 1.0 / med_s_per_it if med_s_per_it > 0 else None
        steps_detected = int(out["Train runtime"] / med_s_per_it) if out["Train runtime"] is not None else None
    # elif it_per_s_vals:
    #     iter_per_s = float(np.median(sorted(it_per_s_vals)))
    #     steps_detected = len(it_per_s_vals)

    out["Iter/s"] = med_s_per_it
    out["Steps detected"] = steps_detected

    # --- Trainer метрики (если есть) -----------------------------------------


    # --- Effective batch size -------------------------------------------------
    ebs = None
    if per_device is not None and ga is not None and (ws is not None):
        ebs = per_device * ga * ws
    out["Effective batch size"] = ebs

    # --- Samples/s и total samples -------------------------------------------
    # samples_per_s = None
    # if iter_per_s is not None and ebs is not None:
    #     samples_per_s = iter_per_s * ebs
    out["Samples/s"] = round(out["Effective batch size"] / out["Iter/s"], 2) if out["Iter/s"] is not None and out["Effective batch size"] is not None else None

    # total_samples = None
    # if ebs is not None and steps_detected:
    #     total_samples = ebs * steps_detected
    # out["Total samples seen"] = int(total_samples) if total_samples is not None else None
    out["Total samples seen"] = int(out["Steps detected"] * out["Effective batch size"]) if out["Steps detected"] is not None and out["Effective batch size"] is not None else None

    # --- Peak GPU mem (GB) эвристика -----------------------------------------
    mem_candidates = []
    for l in lines:
        if re.search(r"(GPU|memory|allocated|reserved|VRAM|peak)", l, flags=re.IGNORECASE):
            for val, unit in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(GB|GiB|MB|MiB|KB|KiB)", l, flags=re.IGNORECASE):
                try:
                    mem_candidates.append(_to_gb(float(val), unit))
                except Exception:
                    pass
    out["Peak GPU mem (GB)"] = max(mem_candidates) if mem_candidates else None

    return out

def _expand_globs(arg: str) -> List[Path]:
    p = Path(arg)
    if any(ch in arg for ch in "*?[]"):
        return [Path(x) for x in Path(".").glob(arg)]
    if p.is_dir():
        return sorted(list(p.rglob("*.log")) + list(p.rglob("*.out")) + list(p.rglob("*.txt")))
    if p.is_file():
        return [p]
    return [Path(x) for x in Path(".").glob(arg)]

def parse_logs(dir_or_glob: str) -> pd.DataFrame:
    paths = _expand_globs(dir_or_glob)
    rows: List[Dict[str, Any]] = []
    for f in paths:
        try:
            rows.append(parse_one_log(f))
        except Exception as e:
            rows.append({"File name": f.name, "Path": str(f), "Error": str(e)})
    df = pd.DataFrame(rows)

    # упорядочим колонки
    preferred = [
        "File name", "Run name", "W&B URL",
        "CUDA visible devices", "World size",
        "Per-device batch size", "Gradient accumulation", "Effective batch size",
        "Parallel mode", "Precision",
        "Iter/s", "Samples/s", "Total samples seen",
        "Train runtime", "Eval loss",
        "Peak GPU mem (GB)",
        "Path",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]