from __future__ import annotations

import argparse
import csv
import glob
import logging
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline_interface.traits import resolve_trait

log = logging.getLogger(__name__)

_PROBE_TO_CONDITION: dict[str, str] = {
    "no_prompt": "none",
    "elicit_undesired": "respond",
}

_MODEL_SUFFIX = "Llama3.1(8.0, LR1e-04)_seed42"
_CSV_FIELDNAMES = ["mean", "lower_bound", "upper_bound", "count", "confidence", "group", "evaluation_id", "conditions"]


def _make_group(cond_prefix: str, des_adj: str, und_adj: str) -> str:
    if cond_prefix == "C1":
        return f"T({des_adj}, {und_adj}100%)I(Empty)_{_MODEL_SUFFIX}"
    elif cond_prefix == "C2":
        return f"T({des_adj}, {und_adj}100%)I({und_adj})_{_MODEL_SUFFIX}"
    else:  # R512
        return f"R512({des_adj}, {und_adj}100%)I({und_adj})_{_MODEL_SUFFIX}"


def _csv_row(mean: float, count: int, group: str, eval_id: str, condition: str) -> dict:
    return {
        "mean": mean,
        "lower_bound": mean,
        "upper_bound": mean,
        "count": count,
        "confidence": 0.95,
        "group": group,
        "evaluation_id": eval_id,
        "conditions": condition,
    }


def convert(configs_dir: Path, ipft_results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_trait: dict[str, list[dict]] = defaultdict(list)
    base_written: set[tuple[str, str, str]] = set()

    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        condition_name: str = cfg["condition_name"]
        cond_prefix = condition_name.split("_")[0]
        des_adj = resolve_trait(cfg["trait_pair"]["desired_trait"]).adjective
        und_adj = resolve_trait(cfg["trait_pair"]["undesired_trait"]).adjective
        base_model_id: str = cfg["training"]["base_model_id"]
        base_group = base_model_id.split("/")[-1]
        ft_group = _make_group(cond_prefix, des_adj, und_adj)

        pattern = str(ipft_results_dir / f"{condition_name}_*" / "metrics.csv")
        matches = glob.glob(pattern)
        if not matches:
            log.warning("No metrics.csv found for %s (pattern: %s)", condition_name, pattern)
            continue

        with open(matches[0], newline="") as f:
            metrics_rows = list(csv.DictReader(f))

        for row in metrics_rows:
            probe = row["probe_name"]
            if probe not in _PROBE_TO_CONDITION:
                continue

            condition = _PROBE_TO_CONDITION[probe]
            eval_id = row["dataset"]
            count = int(row["n_records"])
            model_id: str = row["model_id"]
            is_base = model_id == base_model_id or base_model_id.split("/")[-1] in model_id

            des_mean = float(row["desired_mean"])
            und_mean = float(row["undesired_mean"])

            if is_base:
                for trait_adj, mean_val in ((des_adj, des_mean), (und_adj, und_mean)):
                    key = (trait_adj, eval_id, condition)
                    if key not in base_written:
                        base_written.add(key)
                        rows_by_trait[trait_adj].append(
                            _csv_row(mean_val, count, base_group, eval_id, condition)
                        )
            else:
                rows_by_trait[des_adj].append(_csv_row(des_mean, count, ft_group, eval_id, condition))
                rows_by_trait[und_adj].append(_csv_row(und_mean, count, ft_group, eval_id, condition))

    for trait_adj, rows in rows_by_trait.items():
        out_path = output_dir / f"TD_ci_{trait_adj}_last.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Wrote %d rows to %s", len(rows), out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--configs-dir", required=True, type=Path)
    parser.add_argument("--ipft-results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    convert(args.configs_dir, args.ipft_results_dir, args.output_dir)


if __name__ == "__main__":
    main()
