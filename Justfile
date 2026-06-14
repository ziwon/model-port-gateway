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
    uvicorn model_port.api.main:app --reload --host 0.0.0.0 --port ${MODEL_PORT_API_PORT:-18080}

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

prepare-scene-data:
    docker compose up -d --build trainer
    docker compose exec trainer python scripts/prepare_scene_dataset.py --output data/scene_classification.jsonl --num-samples 200

prepare-cifar10-data:
    docker compose up -d --build trainer
    docker compose exec trainer python scripts/prepare_cifar10_dataset.py --output data/cifar10_subset.jsonl --num-samples 500

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

train-v020:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.pipelines.train_classifier --config configs/train.edge_classifier.v0.2.0.yaml

train-v030:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.pipelines.train_classifier --config configs/train.edge_classifier.v0.3.0.yaml

evaluate-v020:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.pipelines.evaluate_classifier --config configs/train.edge_classifier.v0.2.0.yaml --model-dir artifacts/models/edge-scene-classifier-v0.2.0 --dataset data/scene_classification.jsonl --output artifacts/eval/eval_report-v0.2.0.json --version 0.2.0

evaluate-v030:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.pipelines.evaluate_classifier --config configs/train.edge_classifier.v0.3.0.yaml --model-dir artifacts/models/edge-object-classifier-v0.3.0 --dataset data/cifar10_subset.jsonl --output artifacts/eval/eval_report-v0.3.0.json --version 0.3.0

build-manifest:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml

build-manifest-v011:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report-v0.1.1.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.1.yaml

build-manifest-v020:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.edge_classifier.yaml --eval-report artifacts/eval/eval_report-v0.2.0.json --output artifacts/manifests/vendor-demo-edge-scene-classifier-0.2.0.yaml

build-manifest-v030:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.edge_classifier.v0.3.0.yaml --eval-report artifacts/eval/eval_report-v0.3.0.json --output artifacts/manifests/vendor-demo-edge-object-classifier-0.3.0.yaml

register:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml --eval-report artifacts/eval/eval_report.json --model-dir artifacts/models/smolvlm2-caption-lora

register-v011:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-smart-captioner-0.1.1.yaml --eval-report artifacts/eval/eval_report-v0.1.1.json --model-dir artifacts/models/smolvlm2-caption-lora --aliases candidate,cloud-sim-passed,v0.1.1

register-v020:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-edge-scene-classifier-0.2.0.yaml --eval-report artifacts/eval/eval_report-v0.2.0.json --model-dir artifacts/models/edge-scene-classifier-v0.2.0 --aliases candidate,edge-target-passed,v0.2.0

register-v030:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-edge-object-classifier-0.3.0.yaml --eval-report artifacts/eval/eval_report-v0.3.0.json --model-dir artifacts/models/edge-object-classifier-v0.3.0 --aliases candidate,edge-target-public-data,v0.3.0

api-register:
    docker compose up -d --build api
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/register -H 'Content-Type: application/json' -d '{"vendor":"vendor-demo","model_name":"smart-captioner","version":"0.1.0","manifest_path":"artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml","stage":"candidate"}'

api-register-v011:
    docker compose up -d --build api
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/register -H 'Content-Type: application/json' -d '{"vendor":"vendor-demo","model_name":"smart-captioner","version":"0.1.1","manifest_path":"artifacts/manifests/vendor-demo-smart-captioner-0.1.1.yaml","stage":"candidate"}'

api-register-v020:
    docker compose up -d --build api
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/register -H 'Content-Type: application/json' -d '{"vendor":"vendor-demo","model_name":"edge-scene-classifier","version":"0.2.0","manifest_path":"artifacts/manifests/vendor-demo-edge-scene-classifier-0.2.0.yaml","stage":"candidate"}'

api-register-v030:
    docker compose up -d --build api
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/register -H 'Content-Type: application/json' -d '{"vendor":"vendor-demo","model_name":"edge-object-classifier","version":"0.3.0","manifest_path":"artifacts/manifests/vendor-demo-edge-object-classifier-0.3.0.yaml","stage":"candidate"}'

promote-v011:
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/vendor-demo.smart-captioner.0.1.1/promote -H 'Content-Type: application/json' -d '{"target_stage":"staging"}'

promote-v020:
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/vendor-demo.edge-scene-classifier.0.2.0/promote -H 'Content-Type: application/json' -d '{"target_stage":"staging"}'

promote-v030:
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/vendor-demo.edge-object-classifier.0.3.0/promote -H 'Content-Type: application/json' -d '{"target_stage":"staging"}'

local-mlops:
    docker compose up -d --build wandb api trainer
    docker compose exec trainer python scripts/prepare_sample_dataset.py --output data/sample_captions.jsonl --num-samples 32
    docker compose exec trainer python -m model_port.pipelines.finetune --config configs/train.smolvlm.yaml
    docker compose exec trainer python -m model_port.pipelines.evaluate --config configs/train.smolvlm.yaml --model-dir artifacts/models/smolvlm2-caption-lora --dataset data/sample_captions.jsonl --output artifacts/eval/eval_report.json
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml --eval-report artifacts/eval/eval_report.json --model-dir artifacts/models/smolvlm2-caption-lora
