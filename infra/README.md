# Infrastructure

The MVP infrastructure target is single-machine Docker Compose.

For now, the source of truth is the repository-root `compose.yaml` plus the
recipes in `Justfile`. The `infra/` tree is reserved for deployment-specific
material that should not complicate the local Compose loop.

## Layout

- `../compose.yaml`: active MVP Compose stack for API, trainer, W&B, volumes, and GPU wiring.
- `compose/`: reserved for future Compose overlays if the local stack grows beyond one file.
- `k8s/`: reserved for the later k3s implementation.

## Promotion Rule

Do not mirror the full Compose stack into k3s manifests yet. The next k3s phase
should start only after the Compose path proves the full contract:

```text
dataset -> train -> W&B run -> eval report -> manifest -> API register -> promote/block
```

When that contract is stable, add k3s manifests from the API outward:

1. API Deployment/Service.
2. Shared artifact storage strategy.
3. W&B or external tracking configuration.
4. Trainer Job/CronJob with GPU scheduling.
5. Registry/promote workflow.
