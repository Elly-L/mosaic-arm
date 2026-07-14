from __future__ import annotations

import argparse
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPORT_TITLE = "MOSAIC-Arm Optimization Proof Report"

REASON_LABELS = {
    "missing_quality_evidence": "Missing quality evidence",
    "below_quality_floor": "Below required quality floor",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return data


def load_decisions(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        raise FileNotFoundError(
            f"Decision directory not found: {directory}"
        )

    paths = sorted(directory.glob("*.json"))

    if not paths:
        raise FileNotFoundError(
            f"No decision JSON files found under {directory}"
        )

    decisions = []

    for path in paths:
        decision = load_json(path)
        decision["_filename"] = path.name
        decisions.append(decision)

    decisions.sort(
        key=lambda item: (
            float(item.get("quality_floor_percent", 0)),
            str(item.get("objective", "")),
        )
    )

    return decisions


def require_candidates(
    dataset: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = dataset.get("candidates")

    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "Candidate dataset must contain a non-empty candidates list."
        )

    required = (
        "id",
        "quantization",
        "model_size_mib",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
    )

    for candidate in candidates:
        missing = [
            field
            for field in required
            if field not in candidate
        ]

        if missing:
            raise ValueError(
                f"Candidate {candidate.get('id', '<unknown>')} "
                f"is missing: {', '.join(missing)}"
            )

    return candidates


def format_number(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "Not available"

    return f"{float(value):,.{decimals}f}"


def format_quality(value: Any) -> str:
    if value is None:
        return "Not evaluated"

    return f"{float(value):.2f}%"


def format_boolean(value: Any) -> str:
    return "Enabled" if bool(value) else "Disabled"


def percent_change(new_value: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("Cannot calculate change from zero baseline.")

    return ((new_value / baseline) - 1.0) * 100.0


def percentage_point_change(
    new_value: float,
    baseline: float,
) -> float:
    return new_value - baseline


def find_by_quantization(
    candidates: list[dict[str, Any]],
    quantization: str,
) -> dict[str, Any] | None:
    return next(
        (
            candidate
            for candidate in candidates
            if candidate["quantization"] == quantization
        ),
        None,
    )


def find_performance_dominators(
    candidates: list[dict[str, Any]],
) -> dict[str, list[str]]:
    dominated_by: dict[str, list[str]] = {}

    for candidate in candidates:
        dominators = []

        for other in candidates:
            if other["id"] == candidate["id"]:
                continue

            no_larger = (
                float(other["model_size_mib"])
                <= float(candidate["model_size_mib"])
            )

            no_slower_prompt = (
                float(other["prompt_tokens_per_second"])
                >= float(candidate["prompt_tokens_per_second"])
            )

            no_slower_generation = (
                float(other["generation_tokens_per_second"])
                >= float(candidate["generation_tokens_per_second"])
            )

            strictly_better = any(
                (
                    float(other["model_size_mib"])
                    < float(candidate["model_size_mib"]),
                    float(other["prompt_tokens_per_second"])
                    > float(candidate["prompt_tokens_per_second"]),
                    float(other["generation_tokens_per_second"])
                    > float(candidate["generation_tokens_per_second"]),
                )
            )

            if (
                no_larger
                and no_slower_prompt
                and no_slower_generation
                and strictly_better
            ):
                dominators.append(other["id"])

        if dominators:
            dominated_by[candidate["id"]] = dominators

    return dominated_by


def candidate_status(
    candidate: dict[str, Any],
    dominated_by: dict[str, list[str]],
) -> str:
    if candidate.get("strict_quality_percent") is None:
        return "Missing quality evidence"

    dominators = dominated_by.get(candidate["id"])

    if dominators:
        return (
            "Performance-dominated by "
            + ", ".join(dominators)
        )

    return "Eligible when quality floor permits"


def decision_rejections(
    decision: dict[str, Any],
) -> str:
    rejected = decision.get("rejected_candidates", [])

    if not rejected:
        return "None"

    descriptions = []

    for item in rejected:
        reason = REASON_LABELS.get(
            item.get("reason"),
            item.get("reason", "Unknown reason"),
        )

        if item.get("reason") == "below_quality_floor":
            description = (
                f"{item['id']}: {reason} "
                f"({item.get('quality_percent')}% < "
                f"{item.get('required_percent')}%)"
            )
        else:
            description = f"{item['id']}: {reason}"

        descriptions.append(description)

    return "; ".join(descriptions)


def build_summary(
    dataset: dict[str, Any],
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    q4 = find_by_quantization(candidates, "Q4_K_M")
    q8 = find_by_quantization(candidates, "Q8_0")

    comparison = None

    if q4 is not None and q8 is not None:
        q4_quality = q4.get("strict_quality_percent")
        q8_quality = q8.get("strict_quality_percent")

        comparison = {
            "baseline": q4["id"],
            "candidate": q8["id"],
            "model_size_change_percent": round(
                percent_change(
                    float(q8["model_size_mib"]),
                    float(q4["model_size_mib"]),
                ),
                2,
            ),
            "prompt_throughput_change_percent": round(
                percent_change(
                    float(q8["prompt_tokens_per_second"]),
                    float(q4["prompt_tokens_per_second"]),
                ),
                2,
            ),
            "generation_throughput_change_percent": round(
                percent_change(
                    float(
                        q8["generation_tokens_per_second"]
                    ),
                    float(
                        q4["generation_tokens_per_second"]
                    ),
                ),
                2,
            ),
            "quality_change_percentage_points": (
                round(
                    percentage_point_change(
                        float(q8_quality),
                        float(q4_quality),
                    ),
                    2,
                )
                if q4_quality is not None
                and q8_quality is not None
                else None
            ),
        }

    decision_summary = []

    for decision in decisions:
        selected = decision.get("selected_configuration")

        decision_summary.append({
            "source_file": decision["_filename"],
            "objective": decision.get("objective"),
            "quality_floor_percent": decision.get(
                "quality_floor_percent"
            ),
            "selected_id": (
                selected.get("id")
                if selected is not None
                else None
            ),
            "selected_quantization": (
                selected.get("quantization")
                if selected is not None
                else None
            ),
            "selection_score": (
                selected.get("selection_score")
                if selected is not None
                else None
            ),
            "rejected_candidates": decision.get(
                "rejected_candidates",
                [],
            ),
        })

    return {
        "report_title": REPORT_TITLE,
        "report_generated_at": datetime.now(UTC)
        .isoformat()
        .replace("+00:00", "Z"),
        "candidate_dataset_generated_at": dataset.get(
            "generated_at"
        ),
        "hardware": dataset.get("hardware", {}),
        "workload": dataset.get("workload", {}),
        "sources": dataset.get("sources", {}),
        "candidate_count": len(candidates),
        "q8_vs_q4": comparison,
        "decisions": decision_summary,
    }


def build_markdown(
    dataset: dict[str, Any],
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    hardware = dataset.get("hardware", {})
    workload = dataset.get("workload", {})
    sources = dataset.get("sources", {})

    dominated_by = find_performance_dominators(candidates)

    lines = [
        f"# {REPORT_TITLE}",
        "",
        (
            "This report was generated automatically from native "
            "Arm64 benchmark and quality-evaluation artifacts."
        ),
        "",
        "## Executive summary",
        "",
    ]

    comparison = summary.get("q8_vs_q4")

    if comparison is not None:
        lines.extend([
            (
                f"- Q8_0 used "
                f"{comparison['model_size_change_percent']:+.2f}% "
                "more model storage than Q4_K_M."
            ),
            (
                f"- Q8_0 changed prompt throughput by "
                f"{comparison['prompt_throughput_change_percent']:+.2f}% "
                "relative to Q4_K_M."
            ),
            (
                f"- Q8_0 changed generation throughput by "
                f"{comparison['generation_throughput_change_percent']:+.2f}% "
                "relative to Q4_K_M."
            ),
            (
                f"- Q8_0 changed strict quality by "
                f"{comparison['quality_change_percentage_points']:+.2f} "
                "percentage points relative to Q4_K_M."
            ),
        ])

    lines.extend([
        (
            "- MOSAIC applies the requested quality floor before "
            "ranking configurations for latency, memory, or a "
            "balanced objective."
        ),
        "",
        "## Test environment",
        "",
        "| Property | Value |",
        "|---|---|",
        (
            f"| Runner | "
            f"{hardware.get('runner', 'Not recorded')} |"
        ),
        (
            f"| Architecture | "
            f"{hardware.get('architecture', 'Not recorded')} |"
        ),
        (
            f"| CPU | "
            f"{hardware.get('cpu', 'Not recorded')} |"
        ),
        (
            f"| CPU cores | "
            f"{hardware.get('cores', 'Not recorded')} |"
        ),
        (
            f"| Memory | "
            f"{hardware.get('memory_gib', 'Not recorded')} GiB |"
        ),
        (
            f"| Model family | "
            f"{workload.get('model_family', 'Not recorded')} |"
        ),
        (
            f"| Prompt workload | "
            f"{workload.get('prompt_tokens', 'Not recorded')} tokens |"
        ),
        (
            f"| Generation workload | "
            f"{workload.get('generation_tokens', 'Not recorded')} tokens |"
        ),
        (
            f"| Inference threads | "
            f"{workload.get('threads', 'Not recorded')} |"
        ),
        "",
        "## Candidate evidence",
        "",
        (
            "| Candidate | Quantization | KleidiAI | Size (MiB) | "
            "Prompt tok/s | Generation tok/s | Strict quality | Status |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])

    for candidate in candidates:
        lines.append(
            "| "
            f"{candidate['id']} | "
            f"{candidate['quantization']} | "
            f"{format_boolean(candidate.get('kleidiai'))} | "
            f"{format_number(candidate['model_size_mib'])} | "
            f"{format_number(candidate['prompt_tokens_per_second'])} | "
            f"{format_number(candidate['generation_tokens_per_second'])} | "
            f"{format_quality(candidate.get('strict_quality_percent'))} | "
            f"{candidate_status(candidate, dominated_by)} |"
        )

    lines.extend([
        "",
        "## Automated decisions",
        "",
        (
            "| Objective | Quality floor | Selected configuration | "
            "Quantization | Selection score | Rejections |"
        ),
        "|---|---:|---|---:|---:|---|",
    ])

    for decision in decisions:
        selected = decision.get("selected_configuration")

        if selected is None:
            selected_id = "No eligible configuration"
            quantization = "—"
            score = "—"
        else:
            selected_id = selected["id"]
            quantization = selected["quantization"]
            score = format_number(
                selected.get("selection_score"),
                6,
            )

        lines.append(
            "| "
            f"{decision.get('objective', 'Unknown')} | "
            f"{format_number(decision.get('quality_floor_percent'))}% | "
            f"{selected_id} | "
            f"{quantization} | "
            f"{score} | "
            f"{decision_rejections(decision)} |"
        )

    lines.extend([
        "",
        "## Quality-floor behavior",
        "",
        (
            "MOSAIC first removes candidates that lack quality evidence "
            "or fall below the requested strict-quality threshold. "
            "It then normalizes prompt throughput, generation throughput, "
            "and memory efficiency across the remaining candidates."
        ),
        "",
        (
            "This prevents the fastest raw configuration from being "
            "selected when it violates the deployment's quality SLA."
        ),
        "",
        "## Key measured trade-off",
        "",
    ])

    if comparison is not None:
        lines.extend([
            "| Comparison | Result |",
            "|---|---:|",
            (
                "| Q8_0 model-size change vs Q4_K_M | "
                f"{comparison['model_size_change_percent']:+.2f}% |"
            ),
            (
                "| Q8_0 prompt-throughput change vs Q4_K_M | "
                f"{comparison['prompt_throughput_change_percent']:+.2f}% |"
            ),
            (
                "| Q8_0 generation-throughput change vs Q4_K_M | "
                f"{comparison['generation_throughput_change_percent']:+.2f}% |"
            ),
            (
                "| Q8_0 strict-quality change vs Q4_K_M | "
                f"{comparison['quality_change_percentage_points']:+.2f} "
                "percentage points |"
            ),
        ])
    else:
        lines.append(
            "Q4_K_M and Q8_0 were not both available for comparison."
        )

    lines.extend([
        "",
        "## Reproducibility evidence",
        "",
        (
            f"- Candidate dataset generated: "
            f"`{dataset.get('generated_at', 'Not recorded')}`"
        ),
        (
            f"- Quantization workflow run ID: "
            f"`{sources.get('quantization_workflow_run_id', 'Not recorded')}`"
        ),
        (
            f"- Quality workflow run ID: "
            f"`{sources.get('quality_workflow_run_id', 'Not recorded')}`"
        ),
        (
            f"- Quantization summary source: "
            f"`{sources.get('quantization_summary', 'Not recorded')}`"
        ),
        (
            f"- Quality summary source: "
            f"`{sources.get('quality_summary', 'Not recorded')}`"
        ),
        "",
        "## Interpretation limits",
        "",
        (
            "- Performance findings apply to the recorded Arm "
            "Neoverse-N2 runner, model family, quantizations, workload, "
            "thread count, and llama.cpp build."
        ),
        (
            "- The current quality score is based on eight deterministic "
            "evaluation cases and should be expanded before final claims."
        ),
        (
            "- A missing quality score is treated as missing evidence, "
            "not as a zero-quality result."
        ),
        (
            "- Throughput results do not alone measure production P95 "
            "latency, concurrency, energy use, or cost."
        ),
        "",
    ])

    return "\n".join(lines)


def html_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    header_html = "".join(
        f"<th>{html.escape(header)}</th>"
        for header in headers
    )

    row_html = []

    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(cell))}</td>"
            for cell in row
        )

        row_html.append(f"<tr>{cells}</tr>")

    return (
        "<div class=\"table-wrap\">"
        "<table>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table>"
        "</div>"
    )


def build_html(
    dataset: dict[str, Any],
    candidates: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    hardware = dataset.get("hardware", {})
    workload = dataset.get("workload", {})
    sources = dataset.get("sources", {})
    dominated_by = find_performance_dominators(candidates)
    comparison = summary.get("q8_vs_q4")

    candidate_rows = []

    for candidate in candidates:
        candidate_rows.append([
            candidate["id"],
            candidate["quantization"],
            format_boolean(candidate.get("kleidiai")),
            format_number(candidate["model_size_mib"]),
            format_number(
                candidate["prompt_tokens_per_second"]
            ),
            format_number(
                candidate["generation_tokens_per_second"]
            ),
            format_quality(
                candidate.get("strict_quality_percent")
            ),
            candidate_status(candidate, dominated_by),
        ])

    decision_rows = []

    for decision in decisions:
        selected = decision.get("selected_configuration")

        decision_rows.append([
            str(decision.get("objective", "Unknown")),
            (
                f"{format_number(decision.get('quality_floor_percent'))}%"
            ),
            (
                selected["id"]
                if selected is not None
                else "No eligible configuration"
            ),
            (
                selected["quantization"]
                if selected is not None
                else "—"
            ),
            (
                format_number(
                    selected.get("selection_score"),
                    6,
                )
                if selected is not None
                else "—"
            ),
            decision_rejections(decision),
        ])

    comparison_html = ""

    if comparison is not None:
        comparison_html = html_table(
            ["Comparison", "Result"],
            [
                [
                    "Q8_0 model-size change vs Q4_K_M",
                    (
                        f"{comparison['model_size_change_percent']:+.2f}%"
                    ),
                ],
                [
                    "Q8_0 prompt-throughput change vs Q4_K_M",
                    (
                        f"{comparison['prompt_throughput_change_percent']:+.2f}%"
                    ),
                ],
                [
                    "Q8_0 generation-throughput change vs Q4_K_M",
                    (
                        f"{comparison['generation_throughput_change_percent']:+.2f}%"
                    ),
                ],
                [
                    "Q8_0 strict-quality change vs Q4_K_M",
                    (
                        f"{comparison['quality_change_percentage_points']:+.2f} "
                        "percentage points"
                    ),
                ],
            ],
        )

    environment_table = html_table(
        ["Property", "Value"],
        [
            ["Runner", hardware.get("runner", "Not recorded")],
            [
                "Architecture",
                hardware.get("architecture", "Not recorded"),
            ],
            ["CPU", hardware.get("cpu", "Not recorded")],
            ["CPU cores", hardware.get("cores", "Not recorded")],
            [
                "Memory",
                f"{hardware.get('memory_gib', 'Not recorded')} GiB",
            ],
            [
                "Model family",
                workload.get("model_family", "Not recorded"),
            ],
            [
                "Prompt workload",
                f"{workload.get('prompt_tokens', 'Not recorded')} tokens",
            ],
            [
                "Generation workload",
                (
                    f"{workload.get('generation_tokens', 'Not recorded')} "
                    "tokens"
                ),
            ],
            [
                "Inference threads",
                workload.get("threads", "Not recorded"),
            ],
        ],
    )

    candidate_table = html_table(
        [
            "Candidate",
            "Quantization",
            "KleidiAI",
            "Size (MiB)",
            "Prompt tok/s",
            "Generation tok/s",
            "Strict quality",
            "Status",
        ],
        candidate_rows,
    )

    decision_table = html_table(
        [
            "Objective",
            "Quality floor",
            "Selected configuration",
            "Quantization",
            "Selection score",
            "Rejections",
        ],
        decision_rows,
    )

    generated_at = html.escape(
        summary["report_generated_at"]
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(REPORT_TITLE)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --background: #0b0f14;
      --panel: #121923;
      --panel-soft: #17212d;
      --border: #293747;
      --text: #e8edf2;
      --muted: #9eabb8;
      --accent: #8be28b;
      --warning: #f0c36a;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--background);
      color: var(--text);
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.6;
    }}

    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 48px 0 80px;
    }}

    header {{
      margin-bottom: 32px;
      padding: 32px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background:
        linear-gradient(
          135deg,
          rgba(139, 226, 139, 0.10),
          transparent 55%
        ),
        var(--panel);
    }}

    h1, h2 {{
      line-height: 1.2;
    }}

    h1 {{
      margin: 0 0 12px;
      font-size: clamp(2rem, 5vw, 3.4rem);
    }}

    h2 {{
      margin-top: 42px;
      color: var(--accent);
    }}

    .subtitle,
    .meta {{
      color: var(--muted);
    }}

    .callout {{
      padding: 18px 20px;
      border-left: 4px solid var(--accent);
      border-radius: 8px;
      background: var(--panel-soft);
    }}

    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
    }}

    th,
    td {{
      padding: 13px 14px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }}

    th {{
      background: var(--panel-soft);
      color: var(--accent);
      white-space: nowrap;
    }}

    tr:last-child td {{
      border-bottom: 0;
    }}

    code {{
      color: var(--warning);
    }}

    ul {{
      padding-left: 22px;
    }}

    footer {{
      margin-top: 48px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>{html.escape(REPORT_TITLE)}</h1>
    <p class="subtitle">
      Automatically generated from native Arm64 performance and
      quality-evaluation evidence.
    </p>
    <p class="meta">Report generated: {generated_at}</p>
  </header>

  <section>
    <h2>Executive summary</h2>
    <div class="callout">
      MOSAIC enforces the requested quality floor before ranking
      candidates for latency, memory efficiency, or a balanced objective.
      The fastest raw configuration is therefore rejected when it fails
      the deployment's quality SLA.
    </div>
  </section>

  <section>
    <h2>Test environment</h2>
    {environment_table}
  </section>

  <section>
    <h2>Candidate evidence</h2>
    {candidate_table}
  </section>

  <section>
    <h2>Automated decisions</h2>
    {decision_table}
  </section>

  <section>
    <h2>Key measured trade-off</h2>
    {comparison_html}
  </section>

  <section>
    <h2>Quality-floor behavior</h2>
    <p>
      Candidates with missing quality evidence or strict-quality scores
      below the requested threshold are excluded before objective scoring.
      Eligible candidates are ranked using normalized prompt throughput,
      generation throughput, and memory efficiency.
    </p>
  </section>

  <section>
    <h2>Reproducibility evidence</h2>
    <ul>
      <li>
        Candidate dataset generated:
        <code>{html.escape(str(dataset.get("generated_at", "Not recorded")))}</code>
      </li>
      <li>
        Quantization workflow run ID:
        <code>{html.escape(str(sources.get("quantization_workflow_run_id", "Not recorded")))}</code>
      </li>
      <li>
        Quality workflow run ID:
        <code>{html.escape(str(sources.get("quality_workflow_run_id", "Not recorded")))}</code>
      </li>
      <li>
        Quantization summary:
        <code>{html.escape(str(sources.get("quantization_summary", "Not recorded")))}</code>
      </li>
      <li>
        Quality summary:
        <code>{html.escape(str(sources.get("quality_summary", "Not recorded")))}</code>
      </li>
    </ul>
  </section>

  <section>
    <h2>Interpretation limits</h2>
    <ul>
      <li>
        Results apply to the recorded hardware, runtime build,
        model family, quantizations, and workload.
      </li>
      <li>
        The present quality evaluation contains eight deterministic cases.
      </li>
      <li>
        Missing quality evidence is not interpreted as zero quality.
      </li>
      <li>
        Throughput does not by itself represent production P95 latency,
        concurrency, energy consumption, or financial cost.
      </li>
    </ul>
  </section>

  <footer>
    Generated by MOSAIC-Arm.
  </footer>
</main>
</body>
</html>
"""


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a human-readable optimization proof report "
            "from MOSAIC candidates and decisions."
        )
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--decisions-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--markdown",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--html",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    dataset = load_json(arguments.candidates)
    candidates = require_candidates(dataset)
    decisions = load_decisions(arguments.decisions_dir)

    summary = build_summary(
        dataset=dataset,
        candidates=candidates,
        decisions=decisions,
    )

    markdown_report = build_markdown(
        dataset=dataset,
        candidates=candidates,
        decisions=decisions,
        summary=summary,
    )

    html_report = build_html(
        dataset=dataset,
        candidates=candidates,
        decisions=decisions,
        summary=summary,
    )

    for path in (
        arguments.markdown,
        arguments.html,
        arguments.summary_json,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    arguments.markdown.write_text(
        markdown_report,
        encoding="utf-8",
    )

    arguments.html.write_text(
        html_report,
        encoding="utf-8",
    )

    arguments.summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== OPTIMIZATION PROOF REPORT =====")
    print(f"Markdown: {arguments.markdown}")
    print(f"HTML: {arguments.html}")
    print(f"Summary JSON: {arguments.summary_json}")
    print(f"Candidates: {len(candidates)}")
    print(f"Decisions: {len(decisions)}")

    comparison = summary.get("q8_vs_q4")

    if comparison is not None:
        print(
            "Q8 vs Q4 prompt throughput: "
            f"{comparison['prompt_throughput_change_percent']:+.2f}%"
        )

        print(
            "Q8 vs Q4 generation throughput: "
            f"{comparison['generation_throughput_change_percent']:+.2f}%"
        )

        print(
            "Q8 vs Q4 strict quality: "
            f"{comparison['quality_change_percentage_points']:+.2f} pp"
        )

    print("===== REPORT GENERATION SUCCESS =====")


if __name__ == "__main__":
    main()
