"""DVC stage: export HF checkpoint to ONNX."""

from __future__ import annotations

import yaml

import export_onnx


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["export_onnx"]

    args = [
        "--checkpoint", params["checkpoint"],
        "--output", params["onnx_out"],
        "--experiment", params["experiment"],
    ]
    export_onnx.main(args)


if __name__ == "__main__":
    main()

