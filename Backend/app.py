
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT_ROOT / "Artifacts"

app = FastAPI(
    title="MPLADS Guardian API",
    description="Backend API for MPLADS monitoring and risk analytics",
    version="1.0.0",
)

# Development configuration. Restrict origins before production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODULE_FILES = {
    "module1": ARTIFACTS / "module1" / "module1_scores.csv",
    "module2": ARTIFACTS / "module2" / "module2_scores.csv",
    "module3": ARTIFACTS / "module3" / "duplicate_scores.csv",
}


def load_module(module: str) -> pd.DataFrame:
    path = MODULE_FILES.get(module)

    if path is None:
        raise HTTPException(status_code=404, detail="Unknown module")

    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Output file not found for {module}: {path.name}",
        )

    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read {module} output: {exc}",
        )


def clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def records_from_frame(frame: pd.DataFrame):
    return [
        {key: clean_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


@app.get("/")
def root():
    return {
        "name": "MPLADS Guardian API",
        "status": "running",
        "modules": list(MODULE_FILES.keys()),
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "outputs": {
            module: {
                "exists": path.exists(),
                "path": str(path),
            }
            for module, path in MODULE_FILES.items()
        },
    }


@app.get("/api/overview")
def overview():
    result = {}

    for module, path in MODULE_FILES.items():
        if not path.exists():
            result[module] = {
                "available": False,
                "message": f"{path.name} not found",
            }
            continue

        frame = load_module(module)
        result[module] = {
            "available": True,
            "total_records": int(len(frame)),
            "columns": list(frame.columns),
        }

        # Include result-category counts when the file has a result column.
        if "result" in frame.columns:
            result[module]["result_counts"] = {
                str(k): int(v)
                for k, v in frame["result"].value_counts(
                    dropna=False
                ).items()
            }

    return result


@app.get("/api/modules/{module}/records")
def module_records(
    module: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    frame = load_module(module)
    page = frame.iloc[offset:offset + limit]

    return {
        "module": module,
        "total": int(len(frame)),
        "limit": limit,
        "offset": offset,
        "records": records_from_frame(page),
    }


@app.get("/api/modules/{module}/top")
def module_top(
    module: str,
    limit: int = Query(default=20, ge=1, le=100),
):
    frame = load_module(module)

    score_column = next(
        (
            column for column in (
                "duplicate_score",
                "fused_score",
                "time_risk_score",
                "risk_score",
                "score",
            )
            if column in frame.columns
        ),
        None,
    )

    if score_column is None:
        return {
            "module": module,
            "message": "No recognized score column found",
            "columns": list(frame.columns),
            "records": [],
        }

    top = frame.sort_values(score_column, ascending=False).head(limit)

    return {
        "module": module,
        "score_column": score_column,
        "total": int(len(frame)),
        "records": records_from_frame(top),
    }
