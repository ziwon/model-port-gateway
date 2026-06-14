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
%%{init: {"theme": "base", "themeVariables": {"background": "#171717", "primaryColor": "#232323", "primaryTextColor": "#f5f5f5", "primaryBorderColor": "#d0d0d0", "lineColor": "#cfcfcf", "fontFamily": "Inter, Arial, sans-serif"}}}%%
flowchart TD
  A["Vendor Model Submission<br/>model + dataset + manifest"]
  B["Fine-tuning Pipeline<br/>LoRA / QLoRA"]
  C["W&B Tracking<br/>runs / tables / artifacts"]
  D["Evaluation + Drift Report<br/>latency + quality evidence"]
  E["Model Manifest<br/>candidate state + policy metadata"]
  F["model-port API<br/>register / promote"]
  G{"Quality Gate<br/>cloud-sim or edge-target"}
  H["Promote to Staging"]
  I["Block Promotion<br/>keep rejection metadata"]
  J["Canary Rollout<br/>future"]
  K["Device Telemetry<br/>future"]

  A --> B
  B --> C
  C --> D
  D --> E
  E --> F
  F --> G
  G -->|passed| H
  G -->|failed| I
  H --> J
  J --> K
  K -. "runtime feedback" .-> G

  classDef vendor fill:#232323,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef training fill:#1b070a,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef tracking fill:#62164d,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef evidence fill:#52676b,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef gateway fill:#5a3520,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef passed fill:#173f32,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef blocked fill:#4a1114,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  class A vendor;
  class B training;
  class C tracking;
  class D,E evidence;
  class F,G gateway;
  class H,J,K passed;
  class I blocked;
```

## Detailed View

The detailed view is split into smaller diagrams so each boundary remains
readable in GitHub. The same ownership model applies across all views: training
produces evidence, W&B stores experiment and artifact lineage, and the gateway
owns promotion control.

### Local Execution

The local execution view shows what runs on a single developer machine. Vendor
inputs enter through a manifest and dataset reference, then the local Compose
stack wires the trainer and W&B services together.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#171717", "primaryColor": "#232323", "primaryTextColor": "#f5f5f5", "primaryBorderColor": "#d0d0d0", "lineColor": "#cfcfcf", "fontFamily": "Inter, Arial, sans-serif"}}}%%
flowchart TD
  subgraph Vendor["Model Vendors / Internal AI Teams"]
    VM["Vendor Model<br/>Base VLM / CV Model"]
    DS["Vendor Dataset<br/>Caption / VQA / Edge Data"]
    MF["Submission Manifest<br/>vendor, model, version, task"]
  end

  subgraph Dev["Developer Workstation<br/>RTX 5080 16GB"]
    REPO["GitHub Repo<br/>model-port-gateway"]
    JUST["Justfile / CLI<br/>local-mlops flow"]
    COMPOSE["Docker Compose<br/>api + trainer + wandb"]
  end

  subgraph Trainer["Trainer Service"]
    PREP["Prepare Dataset<br/>JSONL + Images"]
    FT["LoRA / QLoRA Fine-tuning<br/>SmolVLM2 500M"]
    EVAL["Evaluation Pipeline<br/>base vs candidate"]
  end

  subgraph WB["Weights & Biases Local"]
    RUNS["Experiment Runs<br/>loss, lr, GPU memory"]
    TABLES["Eval Tables<br/>image, GT, base pred, candidate pred"]
    ART["Artifacts<br/>dataset + model + eval report"]
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
  EVAL --> ART

  classDef vendor fill:#232323,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef workstation fill:#1b070a,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef trainer fill:#62164d,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef wandb fill:#52676b,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  class VM,DS,MF vendor;
  class REPO,JUST,COMPOSE workstation;
  class PREP,FT,EVAL trainer;
  class RUNS,TABLES,ART wandb;
  style Vendor fill:#171717,stroke:#5a5a5a,color:#f5f5f5
  style Dev fill:#171717,stroke:#5a5a5a,color:#f5f5f5
  style Trainer fill:#171717,stroke:#5a5a5a,color:#f5f5f5
  style WB fill:#171717,stroke:#5a5a5a,color:#f5f5f5
```

Vendor and internal model teams provide inputs, but they do not control model
status. The trainer owns data preparation, fine-tuning, and evaluation
execution, while W&B records run metrics, prediction tables, and artifacts.

### Promotion Control

The promotion control view is the core gateway path. Evaluation output becomes
a manifest, the API persists a registry record, and quality profiles decide
whether the candidate can move forward.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#171717", "primaryColor": "#232323", "primaryTextColor": "#f5f5f5", "primaryBorderColor": "#d0d0d0", "lineColor": "#cfcfcf", "fontFamily": "Inter, Arial, sans-serif"}}}%%
flowchart TD
  REPORT["Evaluation Report<br/>latency, failure rate, drift, accuracy"]
  MANIFEST["Model Manifest<br/>evaluation + deployment metadata"]

  subgraph Gateway["model-port API Gateway"]
    API["FastAPI<br/>/models/register<br/>/models/{id}/promote"]
    STORE["Local Registry Store<br/>models.json"]
    GATE["Promotion Gate<br/>quality policy check"]
  end

  subgraph Policy["Quality Gate Profiles"]
    CLOUD["cloud-sim<br/>p95 latency <= 3000ms"]
    EDGE["edge-target<br/>p95 latency <= 100ms<br/>model size <= 500MB"]
  end

  subgraph Registry["W&B Registry + Artifact Aliases"]
    CANDIDATE["candidate"]
    STAGING["staging"]
    PRODUCTION["production"]
    REJECTED["rejected-latency<br/>rejected-quality"]
  end

  REPORT --> MANIFEST
  MANIFEST --> API
  API --> STORE
  API --> GATE
  CLOUD --> GATE
  EDGE --> GATE
  GATE -->|passed| STAGING
  GATE -->|failed| REJECTED
  CANDIDATE --> GATE
  STAGING --> PRODUCTION

  classDef evidence fill:#52676b,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef gateway fill:#5a3520,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef policy fill:#62164d,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef accepted fill:#173f32,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef rejected fill:#4a1114,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  class REPORT,MANIFEST evidence;
  class API,STORE,GATE gateway;
  class CLOUD,EDGE policy;
  class CANDIDATE,STAGING,PRODUCTION accepted;
  class REJECTED rejected;
  style Gateway fill:#171717,stroke:#5a5a5a,color:#f5f5f5
  style Policy fill:#171717,stroke:#5a5a5a,color:#f5f5f5
  style Registry fill:#171717,stroke:#5a5a5a,color:#f5f5f5
```

The API is the promotion authority. It reads the evaluated manifest and blocks
promotion when the selected quality profile fails. W&B aliases keep lifecycle
state visible even for rejected candidates.

### Rollout Feedback

The rollout feedback view is future-facing. It shows where staging promotion
would hand off to canary rollout, how stable rollout follows, and how runtime
telemetry feeds later quality decisions.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#171717", "primaryColor": "#232323", "primaryTextColor": "#f5f5f5", "primaryBorderColor": "#d0d0d0", "lineColor": "#cfcfcf", "fontFamily": "Inter, Arial, sans-serif"}}}%%
flowchart TD
  GATE["Quality Gate<br/>passed candidate"]

  subgraph Rollout["Future Rollout Layer"]
    RC["Rollout Controller"]
    CANARY["Canary Device Group"]
    STABLE["Stable Device Group"]
    TEL["Telemetry<br/>latency, FPS, errors, drift"]
  end

  GATE --> RC
  RC --> CANARY
  RC --> STABLE
  CANARY --> TEL
  STABLE --> TEL
  TEL -. "runtime feedback" .-> GATE

  classDef gate fill:#5a3520,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef rollout fill:#173f32,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  classDef telemetry fill:#52676b,stroke:#d0d0d0,color:#f5f5f5,stroke-width:2px;
  class GATE gate;
  class RC,CANARY,STABLE rollout;
  class TEL telemetry;
  style Rollout fill:#171717,stroke:#5a5a5a,color:#f5f5f5
```

This layer is not required for the current local runtime, but the boundary is
kept explicit so the Compose implementation can evolve toward k3s or Kubernetes
without mixing training, registry, and rollout responsibilities.

## Runtime Sequence

The runtime sequence shows the promotion path as a control loop, not just a
training job. Training and evaluation produce evidence, while the API and
quality gate decide whether the model can move forward.

```mermaid
%%{init: {"theme": "base", "themeVariables": {"background": "#171717", "primaryColor": "#232323", "primaryTextColor": "#f5f5f5", "primaryBorderColor": "#d0d0d0", "lineColor": "#cfcfcf", "actorBkg": "#232323", "actorTextColor": "#f5f5f5", "actorBorder": "#d0d0d0", "actorLineColor": "#d0d0d0", "activationBkgColor": "#52676b", "activationBorderColor": "#d0d0d0", "sequenceNumberColor": "#171717", "sequenceNumberBackground": "#d0d0d0", "signalColor": "#d0d0d0", "signalTextColor": "#f5f5f5", "labelBoxBkgColor": "#1f1f1f", "labelBoxBorderColor": "#d0d0d0", "labelTextColor": "#f5f5f5", "loopTextColor": "#f5f5f5", "noteBkgColor": "#62164d", "noteBorderColor": "#d0d0d0", "noteTextColor": "#f5f5f5", "fontFamily": "Inter, Arial, sans-serif"}}}%%
sequenceDiagram
  autonumber
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
6. The rollout controller is intentionally future-facing in the local reference
   runtime. It
   represents the next layer for canary rollout, runtime telemetry, and feedback
   into later quality gate decisions.

## Governance

Vendors cannot self-declare a model as passed. Promotion eligibility is derived
from the evaluated manifest, specifically `evaluation.passed`.

Stage transitions are constrained to `candidate -> staging -> production`.
Candidates cannot skip directly to production, and production is treated as a
terminal stage. Rollback should be represented as a new candidate or a separate
future rollback workflow, not as an unrestricted stage mutation.

Failed candidates remain in the registry with rejection metadata. This preserves
vendor lineage, evaluation evidence, and promotion history for auditability.
