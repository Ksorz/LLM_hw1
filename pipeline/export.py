"""DVC stage: export HF checkpoint to ONNX."""

from __future__ import annotations

import yaml

import export_onnx


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["export_onnx"]

    # export_onnx умеет сам подобрать последний checkpoint и следующий expN,
    # если checkpoint/output/experiment не заданы.
    args = []
    if params.get("checkpoint"):
        args += ["--checkpoint", params["checkpoint"]]
    if params.get("onnx_out"):
        args += ["--output", params["onnx_out"]]
    if params.get("experiment"):
        args += ["--experiment", params["experiment"]]
    export_onnx.main(args)


if __name__ == "__main__":
    main()

