# AGENTS.md

## What This Repo Is

`model-port` is a small Python 3.11 ModelOps scaffold for vendor model intake, fine-tuning, evaluation, registry promotion, and rollout simulation. See [README.md](README.md) for the project overview and architecture.

## Working Rules

- Prefer the existing YAML/config flow in [configs/](configs) and the loaders in [model_port/common/config.py](model_port/common/config.py).
- Keep changes small and consistent with the current Typer CLI pattern in [model_port/pipelines/](model_port/pipelines) and [model_port/registry/](model_port/registry).
- Treat dry-run paths as first-class; several commands are intentionally scaffolding and should fail clearly when real training or registration is not implemented.
- Do not duplicate documentation that already exists in [README.md](README.md); link to it instead.

## Commands

- Install dev dependencies: `pip install -e .[train,dev,api]`
- Run checks: `just test` or `ruff check . && python -m pytest -q`
- Start the API: `just api`
- Exercise scaffolding end to end: `just dryrun`

## Codebase Conventions

- Keep YAML manifests and train configs aligned with the required fields in [tests/test_manifest.py](tests/test_manifest.py).
- Use `load_yaml()` / `load_train_config()` instead of open-coded parsing.
- Preserve the current line-length convention from [pyproject.toml](pyproject.toml) when editing Python files.
- Prefer `rich.print` and `typer.BadParameter` for CLI validation and user-facing errors in registry and pipeline entrypoints.

## Useful Files

- [Justfile](Justfile)
- [README.md](README.md)
- [pyproject.toml](pyproject.toml)
- [configs/model_manifest.example.yaml](configs/model_manifest.example.yaml)
- [configs/train.smolvlm.yaml](configs/train.smolvlm.yaml)
- [tests/test_manifest.py](tests/test_manifest.py)
