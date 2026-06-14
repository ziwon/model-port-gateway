from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from model_port.registry.store import (
    JsonModelRegistry,
    ModelRegistration as StoreRegistration,
    SqliteModelRegistry,
    open_model_registry,
)

app = FastAPI(title="model-port API", version="0.1.0")


class ModelSubmission(BaseModel):
    vendor: str
    name: str
    version: str
    task: str
    artifact_uri: str


class ModelRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    model_name: str
    version: str
    manifest_path: str
    stage: Literal["candidate"] = "candidate"


class PromotionRequest(BaseModel):
    target_stage: Literal["staging", "production"] = Field(...)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/models/submit")
def submit_model(req: ModelSubmission) -> dict[str, str]:
    # MVP placeholder: validate, persist, and trigger pipeline later.
    return {"status": "accepted", "model": f"{req.vendor}/{req.name}:{req.version}"}


@app.post("/models/register")
def register_model(req: ModelRegistration) -> dict[str, Any]:
    try:
        record = _registry().register(
            StoreRegistration(
                vendor=req.vendor,
                model_name=req.model_name,
                version=req.version,
                manifest_path=req.manifest_path,
                stage=req.stage,
            )
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "registered", "model": record}


@app.get("/models")
def list_models() -> dict[str, list[dict[str, Any]]]:
    return {"models": _registry().list()}


@app.get("/models/{vendor}/{model_name}/{version}")
def get_model(vendor: str, model_name: str, version: str) -> dict[str, Any]:
    record = _registry().get_by_parts(vendor, model_name, version)
    if not record:
        raise HTTPException(status_code=404, detail=f"model not found: {vendor}.{model_name}.{version}")
    return record


@app.post("/models/{model_id}/promote")
def promote_model(model_id: str, req: PromotionRequest) -> dict[str, Any]:
    result = _registry().promote(model_id, req.target_stage)
    if not result:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id}")
    return result


def _registry() -> JsonModelRegistry | SqliteModelRegistry:
    try:
        return open_model_registry()
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
