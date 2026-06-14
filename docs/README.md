# Architecture

model-port is organized as a local-first ModelOps gateway.

It separates:

- experiment tracking from promotion control
- model evaluation from rollout decision
- vendor submission from production readiness
- cloud-simulation validation from strict edge-target validation

![model-port architecture](assets/model-port-architecture.svg)

## Lifecycle

The lifecycle starts with a submitted model and ends with a governed promotion
decision. Each step produces an artifact that the next step can verify: training
produces a candidate, evaluation produces a report, the manifest captures the
candidate state, and the API applies the quality gate before rollout.

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

The detailed view expands the local MVP into the components that own each part
of the model lifecycle. The current implementation runs on Docker Compose, but
the boundaries are intentionally close to the future k3s or Kubernetes shape:
trainer, tracking, registry gateway, policy, and rollout remain separate
concerns.

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
    REG["W&B Model Registry<br/>candidate / staging / production<br/>rejected-latency / rejected-quality"]
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

The vendor boundary represents external vendors or internal model teams. They
provide a model, dataset reference, and submission manifest, but they do not
control promotion status.

The developer workstation boundary is the MVP runtime. `Justfile` commands and
Docker Compose wire together the API, trainer, and W&B services so the full
pipeline can be exercised on one machine before moving to k3s.

The trainer service owns data preparation, LoRA or QLoRA fine-tuning, and
evaluation execution. It produces artifacts and reports, but avoids making
rollout decisions directly.

W&B is the default experiment and artifact system. It stores training runs,
evaluation tables, model artifacts, lifecycle aliases, and rejection metadata so
both blocked and promoted candidates remain auditable. The MVP uses aliases such
as `candidate`, `staging`, `production`, `rejected-latency`, and
`rejected-quality`.

The model-port API gateway owns registration and promotion control. It reads
the evaluated manifest, persists a local registry record, and blocks promotion
when the quality gate fails.

Quality gate profiles separate development validation from strict edge targets.
For example, `cloud-sim` can validate that the pipeline and model behavior are
reasonable, while `edge-target` can block candidates that are too slow or too
large for deployment.

The rollout layer is future-facing in the MVP. It sketches where canary rollout,
stable rollout, runtime telemetry, and feedback into later quality decisions
will live.

## Runtime Sequence

The runtime sequence shows the promotion path as a control loop, not just a
training job. Training and evaluation produce evidence, while the API and
quality gate decide whether the model can move forward.

```mermaid
sequenceDiagram
  participant Vendor
  participant Trainer
  participant WNB as W&B
  participant Eval as Evaluation
  participant API as model-port API
  participant Gate as Quality Gate
  participant Rollout as Rollout Controller

  Vendor->>Trainer: Submit model + dataset + manifest
  Trainer->>Trainer: Fine-tune v1 -> v2 with LoRA
  Trainer->>WNB: Log run metrics and artifacts
  Trainer->>Eval: Run base vs candidate inference
  Eval->>WNB: Log eval table and drift metrics
  Eval->>API: Register model manifest
  API->>Gate: Check quality profile
  Gate-->>API: passed or blocked

  alt Quality gate passed
    API->>Rollout: Promote candidate to staging
    Rollout->>Rollout: Canary rollout
  else Quality gate failed
    API-->>Vendor: Block promotion with reject reason
    API->>WNB: Keep rejected candidate metadata
  end
```

1. Vendor submission provides the base model, dataset reference, and manifest
   metadata. The manifest identifies the vendor, model name, version, task, and
   runtime contract.
2. The trainer owns fine-tuning and experiment logging. It writes run metrics,
   model artifacts, and lineage to W&B, but it does not decide production
   readiness.
3. Evaluation compares the base model and candidate model on the same dataset.
   It records latency, failure rate, drift metrics, and sample predictions in
   W&B tables.
4. The evaluated manifest becomes the handoff contract to the model-port API.
   Promotion eligibility is derived from the manifest evaluation section.
5. The quality gate applies a named profile such as `cloud-sim` or
   `edge-target`. A passing result can move to staging; a failing result blocks
   promotion and preserves rejection metadata.
6. The rollout controller is intentionally future-facing in the MVP. It
   represents the next layer for canary rollout, runtime telemetry, and feedback
   into later quality gate decisions.

## Governance

Vendors cannot self-declare a model as passed. Promotion eligibility is derived
from the evaluated manifest, specifically `evaluation.passed`.

Failed candidates remain in the registry with rejection metadata. This preserves
vendor lineage, evaluation evidence, and promotion history for auditability.
