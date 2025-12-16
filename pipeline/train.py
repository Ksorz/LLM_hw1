"""DVC stage: train model (baseline) with limited steps."""

from __future__ import annotations

import os

import yaml

import train_distributed


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["train_model"]

    # Prefer single-GPU for baseline training unless explicitly overridden.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.get("cuda_visible_devices", os.getenv("CUDA_VISIBLE_DEVICES", "0")))

    metrics_out = "output_dir/metrics/train_model.json"
    history_out = "output_dir/plots/train_model_history.csv"
    tb_dir = "output_dir/tensorboard/train_model"

    args = [
        "--mode", params["mode"],
        "--batch-size", str(params["batch_size"]),
        "--grad-accum", str(params["grad_accum"]),
        "--timeout", str(params["timeout"]),
        "--data-dir", params["data_dir"],
        "--output-dir", params.get("output_dir", "output_dir/gpt2-1b-russian"),
        "--metrics-out", metrics_out,
        "--history-out", history_out,
        "--logging-dir", tb_dir,
        "--save-steps", str(params.get("save_steps", 2000)),
        "--eval-steps", str(params.get("eval_steps", 2000)),
        "--logging-steps", str(params.get("logging_steps", 50)),
    ]

    if params.get("bf16", False):
        args.append("--bf16")
    if params.get("torch_compile", False):
        args.append("--torch-compile")
    if params.get("max_steps") is not None:
        args += ["--max-steps", str(params["max_steps"])]
    if params.get("learning_rate") is not None:
        args += ["--learning-rate", str(params["learning_rate"])]
    if params.get("no_wandb", False):
        args.append("--no-wandb")
    if params.get("no_generation", False):
        args.append("--no-generation")
    if params.get("tensorboard", False):
        args.append("--tensorboard")

    train_distributed.main(args)


if __name__ == "__main__":
    main()

