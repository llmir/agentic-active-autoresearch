"""Trusted plugin example: load a JSON array through the common dataset validator."""

from pathlib import Path

import pandas as pd

from agentic_al import register_dataset


def load_json_array(config):
    path = config.resolved_dataset_path()
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"JSON array dataset not found: {path}")
    return pd.read_json(path, orient="records")


register_dataset("json_array", load_json_array)
