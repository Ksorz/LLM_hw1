"""DVC stage: train model (baseline) with limited steps."""

from __future__ import annotations

import yaml

import train_distributed


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["train_model"]

    args = [
        "--mode", params["mode"],
        "--batch-size", str(params["batch_size"]),
        "--grad-accum", str(params["grad_accum"]),
        "--timeout", str(params["timeout"]),
        "--data-dir", params["data_dir"],
    ]

    if params.get("bf16", False):
        args.append("--bf16")
    if params.get("torch_compile", False):
        args.append("--torch-compile")
    if params.get("max_steps") is not None:
        args += ["--max-steps", str(params["max_steps"])]
    if params.get("learning_rate") is not None:
        args += ["--learning-rate", str(params["learning_rate"])]

    train_distributed.main(args)


if __name__ == "__main__":
    main()

