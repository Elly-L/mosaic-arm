from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from mosaic import __version__
from mosaic.decision_engine import (
    load_dataset,
    select_configuration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

DASHBOARD_PATH = (
    REPOSITORY_ROOT
    / "mosaic"
    / "static"
    / "index.html"
)

class RecommendRequest(BaseModel):
    """Optimization request submitted by an API client."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "objective": "latency",
                    "quality_floor_percent": 50,
                },
                {
                    "objective": "memory",
                    "quality_floor_percent": 60,
                },
            ]
        },
    )

    objective: Literal[
        "latency",
        "memory",
        "balanced",
    ] = Field(
        description=(
            "Optimization objective used to rank eligible "
            "inference configurations."
        )
    )

    quality_floor_percent: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description=(
            "Minimum strict-quality score that a candidate "
            "must satisfy before optimization."
        ),
    )


app = FastAPI(
    title="MOSAIC-Arm API",
    description=(
        "Quality-aware inference configuration optimization "
        "for Arm64 cloud CPUs."
    ),
    version=__version__,
)


def resolve_environment_path(
    variable_name: str,
) -> Path | None:
    value = os.getenv(variable_name)

    if not value:
        return None

    return Path(value).expanduser().resolve()


def candidates_path() -> Path:
    configured = resolve_environment_path(
        "MOSAIC_CANDIDATES_PATH"
    )

    if configured is not None:
        return configured

    generated = (
        REPOSITORY_ROOT
        / "generated"
        / "candidates.json"
    )

    if generated.exists():
        return generated

    return (
        REPOSITORY_ROOT
        / "data"
        / "candidates.json"
    )


def report_path() -> Path:
    configured = resolve_environment_path(
        "MOSAIC_REPORT_HTML_PATH"
    )

    if configured is not None:
        return configured

    return (
        REPOSITORY_ROOT
        / "generated"
        / "report"
        / "optimization-proof-report.html"
    )


def load_current_dataset() -> dict[str, Any]:
    path = candidates_path()

    try:
        return load_dataset(path)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "candidate_dataset_unavailable",
                "message": str(error),
                "path": str(path),
            },
        ) from error


@app.get(
    "/",
    tags=["Dashboard"],
    summary="Open the MOSAIC dashboard",
    include_in_schema=False,
    response_class=FileResponse,
)
@app.get(
    "/dashboard",
    tags=["Dashboard"],
    summary="Open the MOSAIC dashboard",
    include_in_schema=False,
    response_class=FileResponse,
)
def dashboard() -> FileResponse:
    if not DASHBOARD_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "dashboard_not_found",
                "message": (
                    "The MOSAIC dashboard file "
                    "is unavailable."
                ),
                "path": str(DASHBOARD_PATH),
            },
        )

    return FileResponse(
        path=DASHBOARD_PATH,
        media_type="text/html",
    )

@app.get(
    "/health",
    tags=["System"],
    summary="Check API readiness",
    response_model=None,
)
def health() -> Any:
    candidate_file = candidates_path()
    proof_report = report_path()

    errors: list[str] = []
    candidate_count = 0

    try:
        dataset = load_dataset(candidate_file)
        candidate_count = len(
            dataset.get("candidates", [])
        )
    except Exception as error:
        errors.append(
            f"Candidate dataset: {error}"
        )

    if not proof_report.exists():
        errors.append(
            f"Proof report not found: {proof_report}"
        )

    payload = {
        "service": "mosaic-arm",
        "version": __version__,
        "status": "ok" if not errors else "degraded",
        "candidate_dataset": str(candidate_file),
        "candidate_count": candidate_count,
        "report_available": proof_report.exists(),
        "errors": errors,
    }

    if errors:
        return JSONResponse(
            status_code=503,
            content=payload,
        )

    return payload


@app.get(
    "/api/candidates",
    tags=["Optimization"],
    summary="List measured inference candidates",
)
def get_candidates() -> dict[str, Any]:
    dataset = load_current_dataset()

    return {
        "schema_version": dataset.get(
            "schema_version"
        ),
        "generated_at": dataset.get(
            "generated_at"
        ),
        "hardware": dataset.get(
            "hardware",
            {},
        ),
        "workload": dataset.get(
            "workload",
            {},
        ),
        "sources": dataset.get(
            "sources",
            {},
        ),
        "candidate_count": len(
            dataset["candidates"]
        ),
        "candidates": dataset["candidates"],
    }


@app.post(
    "/api/recommend",
    tags=["Optimization"],
    summary="Recommend an Arm64 inference configuration",
)
def recommend(
    request: RecommendRequest,
) -> dict[str, Any]:
    dataset = load_current_dataset()

    result = select_configuration(
        dataset=dataset,
        objective=request.objective,
        quality_floor=(
            request.quality_floor_percent
        ),
    )

    selected = result.get(
        "selected_configuration"
    )

    return {
        "status": (
            "selected"
            if selected is not None
            else "no_eligible_configuration"
        ),
        "objective": request.objective,
        "quality_floor_percent": (
            request.quality_floor_percent
        ),
        "selected_configuration": selected,
        "eligible_candidates": result[
            "eligible_candidates"
        ],
        "rejected_candidates": result[
            "rejected_candidates"
        ],
        "hardware": result["hardware"],
        "workload": result["workload"],
        "dataset_generated_at": dataset.get(
            "generated_at"
        ),
        "source_workflow_runs": {
            "quantization": dataset.get(
                "sources",
                {},
            ).get(
                "quantization_workflow_run_id"
            ),
            "quality": dataset.get(
                "sources",
                {},
            ).get(
                "quality_workflow_run_id"
            ),
        },
    }


@app.get(
    "/api/report",
    tags=["Evidence"],
    summary="View the Optimization Proof Report",
    response_class=FileResponse,
)
def get_report() -> FileResponse:
    path = report_path()

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "error": "report_not_found",
                "message": (
                    "The Optimization Proof Report "
                    "has not been generated."
                ),
                "path": str(path),
            },
        )

    return FileResponse(
        path=path,
        media_type="text/html",
    )
