#!/usr/bin/env python3
"""Export a HuggingFace causal LM checkpoint to ONNX with metadata."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import onnx
from onnx import helper as onnx_helper
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.onnx import FeaturesManager, export


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Export HF checkpoint to ONNX")
    parser.add_argument("--checkpoint", required=False, help="Path or name of HF checkpoint (optional)")
    parser.add_argument(
        "--output",
        required=False,
        help="Path to output ONNX file (e.g. /app/output_dir/onnx/exp/model.onnx)",
    )
    parser.add_argument(
        "--experiment",
        required=False,
        default=None,
        help="Experiment name to embed into ONNX metadata",
    )
    parser.add_argument(
        "--commit",
        required=False,
        default=None,
        help="Commit hash to embed into ONNX metadata (default: GIT_COMMIT env or unknown)",
    )
    parser.add_argument(
        "--date",
        required=False,
        default=None,
        help="Date to embed into ONNX metadata (default: now ISO)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for export (cpu/cuda). Export stays device-agnostic, device used only during tracing.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path (defaults to checkpoint)",
    )
    return parser.parse_args(args)


def _export(checkpoint: str, output: Path, opset: int, device: str, tokenizer_path: str | None):
    output.parent.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(checkpoint)
    feature = "causal-lm"
    onnx_config = FeaturesManager.get_config(config, feature=feature)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=opset,
        output=output,
        device=device,
    )
    # Сохраняем токенайзер рядом
    output_dir = output if output.is_dir() else output.parent
    tokenizer.save_pretrained(output_dir)


def _export_with_optimum(checkpoint: str, output: Path, task: str = "text-generation-with-past"):
    """
    Fallback экспорт через optimum-cli, когда transformers.onnx не поддерживает модель.
    optimum-cli требует путь к папке; внутри появится model.onnx.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    out_dir = output if output.is_dir() else output.parent
    tmp_dir = None
    try:
        # Если задан файл, экспортируем во временную папку и потом переносим model.onnx
        if output.suffix == ".onnx":
            tmp_dir_obj = tempfile.TemporaryDirectory()
            tmp_dir = Path(tmp_dir_obj.name)
            out_dir = tmp_dir
        cmd = [
            "optimum-cli",
            "export",
            "onnx",
            "--model",
            checkpoint,
            "--task",
            task,
            str(out_dir),
        ]
        subprocess.run(cmd, check=True)
        if output.suffix == ".onnx":
            src = out_dir / "model.onnx"
            if not src.exists():
                raise FileNotFoundError(f"Expected model.onnx in {out_dir}")
            # перенесем сам onnx
            shutil.move(str(src), str(output))
            # перенесем external data, если она есть
            data_files = list(out_dir.glob("model.onnx*"))
            for f in data_files:
                if f.name == "model.onnx":
                    continue
                shutil.move(str(f), str(output.parent / f.name))
            # перенесем сопутствующие файлы (токенайзер/конфиг), если есть
            for fname in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]:
                src_file = out_dir / fname
                if src_file.exists():
                    shutil.move(str(src_file), str(output.parent / fname))
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _write_metadata(onnx_path: Path, metadata: dict[str, str]) -> None:
    model = onnx.load(onnx_path)
    # Merge/overwrite metadata properties
    onnx_helper.set_model_props(model, metadata)
    # Сохраняем, сохраняя external data, если она была
    onnx.save(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
    )


def _latest_checkpoint(base_dir: Path) -> Path | None:
    candidates = sorted(base_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _next_experiment(base_dir: Path) -> str:
    """Return next exp name like exp1, exp2 based on existing expN folders."""
    base_dir.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.startswith("exp"):
            suffix = p.name[3:]
            if suffix.isdigit():
                max_id = max(max_id, int(suffix))
    return f"exp{max_id + 1}"


def main(argv=None):
    args = parse_args(argv)

    base_ckpt_dir = Path("/app/output_dir/gpt2-1b-russian")
    base_onnx_dir = Path("/app/output_dir/onnx")

    checkpoint = args.checkpoint or _latest_checkpoint(base_ckpt_dir)
    if checkpoint is None:
        raise SystemExit("Checkpoint not provided and no checkpoint-* found in /app/output_dir/gpt2-1b-russian")

    # Resolve output path / experiment
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix == "":
            output_path = output_path / "model.onnx"
    else:
        exp_name = args.experiment or _next_experiment(base_onnx_dir)
        output_path = base_onnx_dir / exp_name / "model.onnx"

    experiment_name = args.experiment or Path(checkpoint).name or output_path.parent.name
    exported = False

    print(f"[info] exporting from checkpoint: {checkpoint}")  # noqa: T201
    print(f"[info] output path: {output_path}")  # noqa: T201
    print(f"[info] experiment: {experiment_name}")  # noqa: T201

    try:
        _export(
            checkpoint=checkpoint,
            output=output_path,
            opset=args.opset,
            device=args.device,
            tokenizer_path=args.tokenizer,
        )
        exported = True
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] transformers.onnx export failed ({exc}); trying optimum-cli...")  # noqa: T201
        _export_with_optimum(
            checkpoint=checkpoint,
            output=output_path,
            task="text-generation-with-past",
        )
        exported = True

    meta = {
        "checkpoint": str(checkpoint),
        "experiment": str(experiment_name),
        "commit": args.commit or os.getenv("GIT_COMMIT", "unknown"),
        "date": args.date or datetime.now().isoformat(),
    }
    _write_metadata(output_path, meta)

    print(f"[info] ONNX export completed")  # noqa: T201
    print(f"[info] checkpoint: {checkpoint}")  # noqa: T201
    print(f"[info] output: {output_path}")  # noqa: T201
    print(f"[info] experiment: {experiment_name}")  # noqa: T201
    print(f"[info] metadata: {meta}")  # noqa: T201


if __name__ == "__main__":
    main()

