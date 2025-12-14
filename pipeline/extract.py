"""DVC stage: extract/tokenize dataset shards."""

from __future__ import annotations

import yaml

from solution import prepare_dataset


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)["extract_data"]

    prepare_dataset(
        dataset_name=params["dataset_name"],
        dataset_config=params["dataset_config"],
        split=params["split"],
        max_length=params["max_length"],
        num_proc=params["num_proc"],
        output_dir=params["output_dir"],
        num_shards=params["num_shards"],
    )


if __name__ == "__main__":
    main()

