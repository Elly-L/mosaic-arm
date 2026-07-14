from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OBJECTIVES = ("latency", "memory", "balanced")


def load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)

    candidates = dataset.get("candidates")

    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Dataset must contain at least one candidate.")

    return dataset


def validate_candidate(candidate: dict[str, Any]) -> None:
    required_fields = (
        "id",
        "quantization",
        "model_size_mib",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    )

    missing = [
        field
        for field in required_fields
        if field not in candidate
    ]

    if missing:
        raise ValueError(
            f"Candidate {candidate.get('id', '<unknown>')} "
            f"is missing fields: {', '.join(missing)}"
        )

    numeric_fields = (
        "model_size_mib",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    )

    for field in numeric_fields:
        value = candidate[field]

        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(
                f"Candidate {candidate['id']} has invalid {field}: {value}"
            )


def filter_candidates(
    candidates: list[dict[str, Any]],
    quality_floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for candidate in candidates:
        validate_candidate(candidate)

        quality = candidate.get("strict_quality_percent")

        if quality is None:
            rejected.append({
                "id": candidate["id"],
                "reason": "missing_quality_evidence",
            })
            continue

        if quality < quality_floor:
            rejected.append({
                "id": candidate["id"],
                "reason": "below_quality_floor",
                "quality_percent": quality,
                "required_percent": quality_floor,
            })
            continue

        eligible.append(candidate.copy())

    return eligible, rejected


def score_candidates(
    candidates: list[dict[str, Any]],
    objective: str,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    maximum_prompt_speed = max(
        candidate["prompt_tokens_per_second"]
        for candidate in candidates
    )

    maximum_generation_speed = max(
        candidate["generation_tokens_per_second"]
        for candidate in candidates
    )

    minimum_model_size = min(
        candidate["model_size_mib"]
        for candidate in candidates
    )

    scored: list[dict[str, Any]] = []

    for candidate in candidates:
        prompt_score = (
            candidate["prompt_tokens_per_second"]
            / maximum_prompt_speed
        )

        generation_score = (
            candidate["generation_tokens_per_second"]
            / maximum_generation_speed
        )

        memory_score = (
            minimum_model_size
            / candidate["model_size_mib"]
        )

        if objective == "latency":
            selection_score = (
                0.65 * prompt_score
                + 0.35 * generation_score
            )
        elif objective == "memory":
            selection_score = memory_score
        elif objective == "balanced":
            selection_score = (
                0.40 * prompt_score
                + 0.30 * generation_score
                + 0.30 * memory_score
            )
        else:
            raise ValueError(f"Unsupported objective: {objective}")

        scored_candidate = candidate.copy()

        scored_candidate["normalized_scores"] = {
            "prompt_speed": round(prompt_score, 6),
            "generation_speed": round(generation_score, 6),
            "memory_efficiency": round(memory_score, 6),
        }

        scored_candidate["selection_score"] = round(
            selection_score,
            6,
        )

        scored.append(scored_candidate)

    scored.sort(
        key=lambda candidate: (
            candidate["selection_score"],
            candidate["strict_quality_percent"],
            -candidate["model_size_mib"],
        ),
        reverse=True,
    )

    return scored


def select_configuration(
    dataset: dict[str, Any],
    objective: str,
    quality_floor: float,
) -> dict[str, Any]:
    eligible, rejected = filter_candidates(
        dataset["candidates"],
        quality_floor,
    )

    scored = score_candidates(eligible, objective)

    winner = scored[0] if scored else None

    return {
        "objective": objective,
        "quality_floor_percent": quality_floor,
        "hardware": dataset.get("hardware", {}),
        "workload": dataset.get("workload", {}),
        "selected_configuration": winner,
        "eligible_candidates": scored,
        "rejected_candidates": rejected,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best Arm64 inference configuration "
            "while enforcing a quality floor."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/candidates.json"),
    )

    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        required=True,
    )

    parser.add_argument(
        "--quality-floor",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if not 0 <= arguments.quality_floor <= 100:
        raise ValueError("Quality floor must be between 0 and 100.")

    dataset = load_dataset(arguments.data)

    result = select_configuration(
        dataset=dataset,
        objective=arguments.objective,
        quality_floor=arguments.quality_floor,
    )

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    selected = result["selected_configuration"]

    print("===== MOSAIC SELECTION =====")
    print(f"Objective: {arguments.objective}")
    print(f"Quality floor: {arguments.quality_floor}%")

    if selected is None:
        print("Selected configuration: NONE")
    else:
        print(f"Selected configuration: {selected['id']}")
        print(f"Quantization: {selected['quantization']}")
        print(f"Selection score: {selected['selection_score']}")

    print(f"Output: {arguments.output}")


if __name__ == "__main__":
    main()
