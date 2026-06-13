# Architecture

model-port is organized as a local-first ModelOps gateway.

It separates:

- experiment tracking from promotion control
- model evaluation from rollout decision
- vendor submission from production readiness
- cloud-simulation validation from strict edge-target validation

## Lifecycle

```mermaid
flowchart LR
  A["Vendor Model Submission"] --> B["Fine-tuning Pipeline<br/>LoRA / QLoRA"]
  B --> C["W&B Tracking<br/>runs / tables / artifacts"]
  C --> D["Evaluation + Drift Report"]
  D --> E["Model Manifest"]
  E --> F["model-port API<br/>register / promote"]
  F --> G{"Quality Gate"}
  G -->|passed| H["Promote to Staging"]
  G -->|failed| I["Block Promotion<br/>keep rejection metadata"]
  H --> J["Canary Rollout<br/>future"]
  J --> K["Device Telemetry<br/>future"]
  K --> G
```

## Detailed View

```mermaid
flowchart LR
  subgraph Vendor["Model Vendors / Internal AI Teams"]
    VM["Vendor Model<br/>Base VLM / CV Model"]
    DS["Vendor Dataset<br/>Caption / VQA / Edge Data"]
    MF["Submission Manifest<br/>vendor, model, version, task"]
  end

  subgraph Dev["Developer Workstation<br/>RTX 5080 16GB"]
    REPO["GitHub Repo<br/>model-port"]
    JUST["Justfile / CLI<br/>local-mlops flow"]
    COMPOSE["Docker Compose<br/>api + trainer + wandb"]
  end

  subgraph Trainer["Trainer Service"]
    PREP["Prepare Dataset<br/>JSONL + Images"]
    FT["LoRA / QLoRA Fine-tuning<br/>SmolVLM2 500M"]
    EVAL["Evaluation Pipeline<br/>base vs candidate"]
    DRIFT["Drift Report<br/>caption length, keywords, image stats"]
  end

  subgraph WB["Weights & Biases Local"]
    RUNS["Experiment Runs<br/>loss, lr, GPU memory"]
    TABLES["Eval Tables<br/>image, GT, base pred, candidate pred"]
    ART["Artifacts<br/>dataset + model + eval report"]
    REG["W&B Model Registry<br/>candidate / passed / rejected"]
  end

  subgraph Gateway["model-port API Gateway"]
    API["FastAPI<br/>/models/register<br/>/models/{id}/promote"]
    STORE["Local Registry Store<br/>models.json MVP"]
    GATE["Promotion Gate<br/>quality policy check"]
    MANIFEST["Model Manifest<br/>evaluation + deployment metadata"]
  end

  subgraph Policy["Quality Gate Profiles"]
    CLOUD["cloud-sim<br/>p95 latency <= 3000ms"]
    EDGE["edge-target<br/>p95 latency <= 100ms<br/>model size <= 500MB"]
  end

  subgraph Rollout["Future Rollout Layer"]
    RC["Rollout Controller"]
    CANARY["Canary Device Group"]
    STABLE["Stable Device Group"]
    TEL["Telemetry<br/>latency, FPS, errors"]
  end

  VM --> MF
  DS --> PREP
  MF --> REPO
  REPO --> JUST
  JUST --> COMPOSE
  COMPOSE --> PREP
  PREP --> FT
  FT --> RUNS
  FT --> EVAL
  EVAL --> TABLES
  EVAL --> DRIFT
  EVAL --> MANIFEST
  DRIFT --> MANIFEST
  MANIFEST --> ART
  ART --> REG
  MANIFEST --> API
  API --> STORE
  API --> GATE
  CLOUD --> GATE
  EDGE --> GATE
  GATE -->|passed| RC
  GATE -->|failed| REG
  RC --> CANARY
  RC --> STABLE
  CANARY --> TEL
  STABLE --> TEL
  TEL --> GATE
```

## Governance

Vendors cannot self-declare a model as passed. Promotion eligibility is derived
from the evaluated manifest, specifically `evaluation.passed`.

Failed candidates remain in the registry with rejection metadata. This preserves
vendor lineage, evaluation evidence, and promotion history for auditability.
