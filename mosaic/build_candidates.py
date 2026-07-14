from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


QUANTIZATION_ORDER = ("Q4_K_M", "Q5_K_M", "Q8_0")


def find_unique_file(directory: Path, filename: str) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    matches = list(directory.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {directory}"
        )

    if len(matches) > 1:
        locations = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Found multiple copies of {filename}: {locations}"
        )

    return matches[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def require_float(
    row: dict[str, str],
    field: str,
    source: Path,
) -> float:
    raw_value = row.get(field)

    if raw_value is None or raw_value.strip() == "":
        raise ValueError(
            f"Missing {field} in {source}: {row}"
        )

    try:
        value = float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"Invalid numeric value for {field}: {raw_value}"
        ) from error

    if value <= 0:
        raise ValueError(
            f"{field} must be positive, received {value}"
        )

    return value


def quantization_slug(quantization: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "-",
        quantization.casefold(),
    ).strip("-")


def load_quantization_results(
    path: Path,
) -> dict[str, dict[str, float]]:
    rows = read_csv(path)
    results: dict[str, dict[str, float]] = {}

    for row in rows:
        quantization = row.get("quantization", "").strip()

        if not quantization:
            raise ValueError(
                f"Missing quantization name in {path}: {row}"
            )

        if quantization in results:
            raise ValueError(
                f"Duplicate quantization in {path}: {quantization}"
            )

        results[quantization] = {
            "model_size_mib": require_float(
                row,
                "model_size_mib",
                path,
            ),
            "prompt_tokens_per_second": require_float(
                row,
                "mean_prompt_tps",
                path,
            ),
            "generation_tokens_per_second": require_float(
                row,
                "mean_generation_tps",
                path,
            ),
        }

    missing = set(QUANTIZATION_ORDER) - set(results)

    if missing:
        raise ValueError(
            "Quantization summary is missing: "
            + ", ".join(sorted(missing))
        )

    return results


def load_quality_results(
    path: Path,
) -> dict[str, float]:
    rows = read_csv(path)
    results: dict[str, float] = {}

    for row in rows:
        quantization = row.get("quantization", "").strip()

        if not quantization:
            raise ValueError(
                f"Missing quantization name in {path}: {row}"
            )

        strict_value = row.get(
            "strict_percent",
            "",
        ).strip()

        if strict_value == "":
            continue

        strict_percent = float(strict_value)

        if not 0 <= strict_percent <= 100:
            raise ValueError(
                f"Invalid quality percentage: {strict_percent}"
            )

        results[quantization] = strict_percent

    return results


def build_dataset(
    quantization_results: dict[str, dict[str, float]],
    quality_results: dict[str, float],
    quantization_summary: Path,
    quality_summary: Path,
    quantization_run_id: str,
    quality_run_id: str,
) -> dict[str, Any]:
    candidates = []

    for quantization in QUANTIZATION_ORDER:
        measurements = quantization_results[quantization]

        candidates.append({
            "id": (
                "smollm2-1.7b-"
                + quantization_slug(quantization)
            ),
            "model_family": "SmolLM2-1.7B-Instruct",
            "quantization": quantization,
            "kleidiai": True,
            "model_size_mib": round(
                measurements["model_size_mib"],
                4,
            ),
            "prompt_tokens_per_second": round(
                measurements[
                    "prompt_tokens_per_second"
                ],
                4,
            ),
            "generation_tokens_per_second": round(
                measurements[
                    "generation_tokens_per_second"
                ],
                4,
            ),
            "strict_quality_percent": (
                quality_results.get(quantization)
            ),
        })

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "sources": {
            "quantization_workflow_run_id":
                quantization_run_id,
            "quality_workflow_run_id":
                quality_run_id,
            "quantization_summary":
                str(quantization_summary),
            "quality_summary":
                str(quality_summary),
        },
        "hardware": {
            "runner": "ubuntu-24.04-arm",
            "architecture": "aarch64",
            "cpu": "Arm Neoverse-N2",
            "cores": 4,
            "memory_gib": 15,
        },
        "workload": {
            "model_family": "SmolLM2-1.7B-Instruct",
            "prompt_tokens": 128,
            "generation_tokens": 64,
            "threads": 4,
            "benchmark_repetitions": 3,
        },
        "candidates": candidates,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MOSAIC candidates from benchmark "
            "and quality artifacts."
        )
    )

    parser.add_argument(
        "--quantization-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--quality-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--quantization-run-id",
        required=True,
    )

    parser.add_argument(
        "--quality-run-id",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    quantization_summary = find_unique_file(
        arguments.quantization_dir,
        "quantization-summary.csv",
    )

    quality_summary = find_unique_file(
        arguments.quality_dir,
        "quality-score-summary.csv",
    )

    quantization_results = load_quantization_results(
        quantization_summary
    )

    quality_results = load_quality_results(
        quality_summary
    )

    dataset = build_dataset(
        quantization_results=quantization_results,
        quality_results=quality_results,
        quantization_summary=quantization_summary,
        quality_summary=quality_summary,
        quantization_run_id=(
            arguments.quantization_run_id
        ),
        quality_run_id=arguments.quality_run_id,
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.output.write_text(
        json.dumps(
            dataset,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== GENERATED CANDIDATES =====")

    for candidate in dataset["candidates"]:
        print(
            f"{candidate['id']} | "
            f"size={candidate['model_size_mib']} MiB | "
            f"prompt={candidate['prompt_tokens_per_second']} | "
            f"generation="
            f"{candidate['generation_tokens_per_second']} | "
            f"quality="
            f"{candidate['strict_quality_percent']}"
        )

    print(f"Output: {arguments.output}")
    print("===== CANDIDATE GENERATION SUCCESS =====")


if __name__ == "__main__":
    main()
