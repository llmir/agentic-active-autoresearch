"""Programmatic external-regression oracle with a separate validation table."""

import pandas as pd

from agentic_al import CallableOracle, load_config, run_experiment


def main() -> None:
    config = load_config("configs/external_oracle_regression.yaml")
    validation = pd.read_csv("data/external_validation.csv")

    def observe(rows, id_column, target_column):
        del target_column
        # Replace this deterministic placeholder with a reviewed, idempotent lab integration.
        return {
            str(row[id_column]): laboratory_lookup(str(row[id_column]))
            for _, row in rows.iterrows()
        }

    result = run_experiment(
        config,
        oracle=CallableOracle(observe),
        validation_data=validation,
    )
    print(result.report_path)


def laboratory_lookup(sample_id: str) -> float:
    raise RuntimeError(f"replace laboratory_lookup before requesting sample {sample_id!r}")


if __name__ == "__main__":
    main()
