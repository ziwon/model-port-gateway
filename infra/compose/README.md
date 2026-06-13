# Compose

The active MVP Compose stack lives at the repository root:

```bash
docker compose -f ../../compose.yaml up --build
```

Keep it there while the project is MVP-focused because Docker Compose tools and
developer muscle memory expect a root `compose.yaml`.

Use this directory later only for optional overlays, for example:

- `compose.gpu.yaml`
- `compose.observability.yaml`
- `compose.prod-sim.yaml`

Until those overlays exist, avoid duplicating the root Compose file here.
