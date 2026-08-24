# Custom datasets and oracles

## Built-in tabular formats

The CSV, newline-delimited JSON (`jsonl`), and optional Parquet loaders accept numerical and
categorical features. Select one with `dataset.kind`; Parquet additionally requires
`pip install 'agentic-active-autoresearch[parquet]'`. IDs must be unique; if the configured ID is absent,
`generate_id_if_missing: true` creates stable IDs from row positions. The target is kept out of
model features and revealed only through `TableOracle`.

For leakage-prone datasets, provide a `group_column` such as patient, batch, chemical series, or
molecular scaffold. Agentic Active AutoResearch then uses a group-disjoint validation split.

## Register another data source

A trusted loader receives the complete `AppConfig` and returns a pandas `DataFrame`. The engine
then applies the same ID, target-leakage, feature, group, cost, risk, and missing-value checks as it
does for built-in files.

```python
from pathlib import Path

import pandas as pd

from agentic_al import register_dataset


def load_json_array(config):
    path = config.resolved_dataset_path()
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"JSON array dataset not found: {path}")
    return pd.read_json(path, orient="records")


register_dataset("json_array", load_json_array)
```

Reference it from YAML:

```yaml
dataset:
  kind: json_array
  path: ../data/pool.json
  task: regression
  target_column: activity
  feature_columns: [descriptor_a, descriptor_b]
  parameters: {}  # loader-specific, secret-free settings only
```

Load the reviewed module for validation and execution:

```bash
agentic-autoresearch validate-config --config config.yaml --plugin my_lab.dataset_plugin
agentic-autoresearch run --config config.yaml --plugin my_lab.dataset_plugin
```

The repository's `examples/custom_dataset.py` is a copyable starting point; package your reviewed
loader in an importable project module such as `my_lab.dataset_plugin`.

Dataset plugins execute normal Python with the user's permissions. Never let a prompt, uploaded
file, or untrusted config choose the module. Keep credentials in environment variables or a secret
store—not in `dataset.parameters`. Secret-bearing parameter names are rejected; an `*_env` field
may hold an uppercase environment-variable name, never its value.

## External regression oracle with a genuinely unlabeled pool

The benchmark path keeps a target column in the table and hides it behind `TableOracle`. For a
regression pool whose target does not exist yet, pass both an explicit idempotent oracle and a
separate labeled `validation_data` DataFrame. Pool and validation IDs—and configured groups—must be
disjoint. The pool file may omit `target_column` entirely.

```python
from pathlib import Path
import pandas as pd
from agentic_al import CallableOracle, load_config, run_experiment


def measure(rows, id_column, target_column):
    # Replace with a trusted, reviewed integration.
    # Return every requested ID exactly once.
    return {
        str(row[id_column]): laboratory_lookup(str(row[id_column])) for _, row in rows.iterrows()
    }


config = load_config(Path("config.yaml"))
validation = pd.read_csv("data/private_validation.csv")
result = run_experiment(
    config,
    oracle=CallableOracle(measure),
    validation_data=validation,
)
```

Oracle implementations must be idempotent by ID. They should enforce authentication,
authorization, rate/cost limits, physical safety approval, and a human confirmation gate when an
observation triggers a real experiment. Agentic Active AutoResearch validates response IDs and missing values but
does not replace domain safety systems.

Version 0.1.0 limits the fully unlabeled external-pool path to regression. Classification still
uses a target-bearing benchmark table so the initial labeled set can be stratified; external
classification needs an explicit seed-label contract in a future release. Do not insert dummy
labels to bypass this limit.

`observations.csv` is required for resume and may be sensitive. It lives under the ignored output
directory; review it before sharing a run bundle.
