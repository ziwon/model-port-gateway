set dotenv-load := true

install:
    pip install -e '.[train,dev,api]'

test:
    ruff check .
    python -m pytest -q

dryrun:
    python scripts/prepare_sample_dataset.py --output data/sample_captions.jsonl --num-samples 32
    python -m model_port.registry.vendor_submit --manifest configs/model_manifest.example.yaml --dry-run
    python -m model_port.pipelines.finetune --config configs/train.smolvlm.yaml --dry-run
    python -m model_port.pipelines.evaluate --config configs/train.smolvlm.yaml --dry-run
    python -m model_port.registry.wandb_register --manifest configs/model_manifest.example.yaml --dry-run

api:
    uvicorn model_port.api.main:app --reload --host 0.0.0.0 --port 8080

compose-up:
    docker compose up --build

compose-down:
    docker compose down

compose-logs:
    docker compose logs -f

trainer-shell:
    docker compose up -d --build wandb trainer
    docker compose exec trainer bash

trainer-gpu:
    docker compose up -d --build trainer
    docker compose exec trainer nvidia-smi

trainer-deps:
    docker compose up -d --build trainer
    docker compose exec trainer python -c "import torchvision, num2words; print('trainer deps ok')"

prepare-data:
    docker compose up -d --build trainer
    docker compose exec trainer python scripts/prepare_sample_dataset.py --output data/sample_captions.jsonl --num-samples 32

train:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.pipelines.finetune --config configs/train.smolvlm.yaml

evaluate:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.pipelines.evaluate --config configs/train.smolvlm.yaml --model-dir artifacts/models/smolvlm2-caption-lora --dataset data/sample_captions.jsonl --output artifacts/eval/eval_report.json

evaluate-edge:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.pipelines.evaluate --config configs/train.smolvlm.yaml --model-dir artifacts/models/smolvlm2-caption-lora --dataset data/sample_captions.jsonl --output artifacts/eval/eval_report.edge-target.json --quality-profile edge-target

evaluate-v011:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.pipelines.evaluate --config configs/train.smolvlm.v0.1.1.yaml --model-dir artifacts/models/smolvlm2-caption-lora --dataset data/sample_captions.jsonl --output artifacts/eval/eval_report-v0.1.1.json --version 0.1.1

build-manifest:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml

build-manifest-v011:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report-v0.1.1.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.1.yaml

register:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report.json

register-v011:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-smart-captioner-0.1.1.yaml --eval-report artifacts/eval/eval_report-v0.1.1.json --model-dir artifacts/models/smolvlm2-caption-lora --aliases candidate,cloud-sim-passed,v0.1.1

api-register:
    docker compose up -d --build api
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/register -H 'Content-Type: application/json' -d '{"vendor":"vendor-demo","model_name":"smart-captioner","version":"0.1.0","manifest_path":"artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml","stage":"candidate","quality_gate_passed":false}'

api-register-v011:
    docker compose up -d --build api
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/register -H 'Content-Type: application/json' -d '{"vendor":"vendor-demo","model_name":"smart-captioner","version":"0.1.1","manifest_path":"artifacts/manifests/vendor-demo-smart-captioner-0.1.1.yaml","stage":"candidate","quality_gate_passed":true}'

promote-v011:
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/vendor-demo.smart-captioner.0.1.1/promote -H 'Content-Type: application/json' -d '{"target_stage":"staging"}'

local-mlops:
    docker compose up -d --build wandb api trainer
    docker compose exec trainer python scripts/prepare_sample_dataset.py --output data/sample_captions.jsonl --num-samples 32
    docker compose exec trainer python -m model_port.pipelines.finetune --config configs/train.smolvlm.yaml
    docker compose exec trainer python -m model_port.pipelines.evaluate --config configs/train.smolvlm.yaml --model-dir artifacts/models/smolvlm2-caption-lora --dataset data/sample_captions.jsonl --output artifacts/eval/eval_report.json
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report.json
