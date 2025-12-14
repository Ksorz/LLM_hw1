"""Tests for ONNX metadata utilities."""

import tempfile

import onnx
from onnx import TensorProto, helper

from ml_service.inference.service import read_onnx_metadata


def _build_dummy_onnx(path: str):
    # Minimal identity graph
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "dummy", [X], [Y])
    model = helper.make_model(graph, producer_name="test")
    helper.set_model_props(
        model,
        {
            "commit": "abc123",
            "experiment": "exp",
            "checkpoint": "/tmp/chkpt",
            "date": "2025-12-06",
        },
    )
    onnx.save(model, path)


def test_read_onnx_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = f"{tmpdir}/model.onnx"
        _build_dummy_onnx(onnx_path)
        meta = read_onnx_metadata(onnx_path)
        assert meta["commit"] == "abc123"
        assert meta["experiment"] == "exp"
        assert meta["checkpoint"] == "/tmp/chkpt"
        assert meta["date"] == "2025-12-06"

