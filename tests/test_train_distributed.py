"""Tests for train_distributed CLI helpers."""

from train_distributed import parse_args


def test_parse_args_eval_and_save_steps():
    args = parse_args([
        "--eval-steps", "123",
        "--save-steps", "456",
        "--mode", "baseline",
    ])

    assert args.eval_steps == 123
    assert args.save_steps == 456


def test_parse_args_defaults_none():
    args = parse_args(["--mode", "baseline"])
    assert args.eval_steps is None
    assert args.save_steps is None

