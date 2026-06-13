# k3s

k3s support is intentionally deferred until the Compose MVP is stable.

Do not add Kubernetes manifests as placeholders that are not runnable. When this
directory becomes active, prefer small, tested manifests organized by component:

```text
k8s/
  api/
  trainer/
  registry/
  observability/
```

The first k3s milestone should be API-only:

```text
FastAPI image -> Deployment -> Service -> health check
```

After that, add trainer jobs, GPU scheduling, storage, and rollout policy.
