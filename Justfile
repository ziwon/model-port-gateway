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

trainer-cuda-gpu:
    docker compose --profile cuda up -d --build trainer-cuda
    docker compose exec trainer-cuda nvidia-smi

trainer-deps:
    docker compose up -d --build trainer
    docker compose exec trainer python -c "import torchvision, num2words; print('trainer deps ok')"

trainer-cuda-deps:
    docker compose --profile cuda up -d --build trainer-cuda
    docker compose exec trainer-cuda python -c "import onnxruntime as ort; print('onnxruntime providers:', ort.get_available_providers())"

prepare-data:
    docker compose up -d --build trainer
    docker compose exec trainer python scripts/prepare_sample_dataset.py --output data/sample_captions.jsonl --num-samples 32

prepare-scene-data:
    docker compose up -d --build trainer
    docker compose exec trainer python scripts/prepare_scene_dataset.py --output data/scene_classification.jsonl --num-samples 200

prepare-cifar10-data:
    docker compose up -d --build trainer
    docker compose exec trainer python scripts/prepare_cifar10_dataset.py --output data/cifar10_subset.jsonl --num-samples 500

prepare-rejected-v020-report:
    docker compose up -d --build trainer
    docker compose exec trainer python scripts/write_demo_eval_report.py --output artifacts/eval/eval_report-v0.2.0-rejected-quality.json --model-name edge-scene-classifier --version 0.2.0 --dataset scratch-edge-classification --accuracy 0.296 --p95-latency-ms 5.4685 --failure-rate 0.0 --drift-score 0.34 --model-size-mb 5.875 --min-accuracy 0.55 --max-p95-latency-ms 100 --max-failure-rate 0.01 --max-drift-score 0.3 --max-model-size-mb 100

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

export-onnx-v030:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.pipelines.export_classifier_onnx --config configs/train.edge_classifier.v0.3.0.yaml --model-dir artifacts/models/edge-object-classifier-v0.3.0 --output artifacts/onnx/edge-object-classifier-v0.3.0/model.onnx

evaluate-onnx-cpu-v030:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.pipelines.evaluate_classifier_onnx --config configs/train.edge_classifier.v0.3.0.yaml --onnx-path artifacts/onnx/edge-object-classifier-v0.3.0/model.onnx --dataset data/cifar10_subset.jsonl --output artifacts/eval/eval_report-v0.3.0-onnx-cpu.json --runtime cpu --version 0.3.0

evaluate-onnx-cuda-v030:
    docker compose --profile cuda up -d --build wandb trainer-cuda
    docker compose exec trainer-cuda python -m model_port.pipelines.evaluate_classifier_onnx --config configs/train.edge_classifier.v0.3.0.yaml --onnx-path artifacts/onnx/edge-object-classifier-v0.3.0/model.onnx --dataset data/cifar10_subset.jsonl --output artifacts/eval/eval_report-v0.3.0-onnx-cuda.json --runtime cuda --version 0.3.0

compare-runtime-v030:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.pipelines.compare_runtime_reports artifacts/eval/eval_report-v0.3.0.json artifacts/eval/eval_report-v0.3.0-onnx-cpu.json --output artifacts/eval/runtime_comparison-v0.3.0.md

compare-runtime-cuda-v030:
    docker compose --profile cuda up -d --build trainer-cuda
    docker compose exec trainer-cuda python -m model_port.pipelines.compare_runtime_reports artifacts/eval/eval_report-v0.3.0.json artifacts/eval/eval_report-v0.3.0-onnx-cpu.json artifacts/eval/eval_report-v0.3.0-onnx-cuda.json --output artifacts/eval/runtime_comparison-v0.3.0.md

build-manifest:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report.rejected-latency.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml

build-manifest-v011:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report-v0.1.1.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.1.yaml

build-manifest-v020:
    docker compose up -d --build trainer
    docker compose exec trainer python scripts/write_demo_eval_report.py --output artifacts/eval/eval_report-v0.2.0-rejected-quality.json --model-name edge-scene-classifier --version 0.2.0 --dataset scratch-edge-classification --accuracy 0.296 --p95-latency-ms 5.4685 --failure-rate 0.0 --drift-score 0.34 --model-size-mb 5.875 --min-accuracy 0.55 --max-p95-latency-ms 100 --max-failure-rate 0.01 --max-drift-score 0.3 --max-model-size-mb 100
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.edge_classifier.yaml --eval-report artifacts/eval/eval_report-v0.2.0-rejected-quality.json --output artifacts/manifests/vendor-demo-edge-scene-classifier-0.2.0.yaml

build-manifest-v030:
    docker compose up -d --build trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.edge_classifier.v0.3.0.yaml --eval-report artifacts/eval/eval_report-v0.3.0.json --output artifacts/manifests/vendor-demo-edge-object-classifier-0.3.0.yaml

register:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report.rejected-latency.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml --eval-report artifacts/eval/eval_report.rejected-latency.json --model-dir artifacts/models/smolvlm2-caption-lora --aliases candidate,rejected-latency,v0.1.0

register-v011:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-smart-captioner-0.1.1.yaml --eval-report artifacts/eval/eval_report-v0.1.1.json --model-dir artifacts/models/smolvlm2-caption-lora --aliases candidate,cloud-sim-passed,v0.1.1

register-v020:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python scripts/write_demo_eval_report.py --output artifacts/eval/eval_report-v0.2.0-rejected-quality.json --model-name edge-scene-classifier --version 0.2.0 --dataset scratch-edge-classification --accuracy 0.296 --p95-latency-ms 5.4685 --failure-rate 0.0 --drift-score 0.34 --model-size-mb 5.875 --min-accuracy 0.55 --max-p95-latency-ms 100 --max-failure-rate 0.01 --max-drift-score 0.3 --max-model-size-mb 100
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.edge_classifier.yaml --eval-report artifacts/eval/eval_report-v0.2.0-rejected-quality.json --output artifacts/manifests/vendor-demo-edge-scene-classifier-0.2.0.yaml
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-edge-scene-classifier-0.2.0.yaml --eval-report artifacts/eval/eval_report-v0.2.0-rejected-quality.json --model-dir artifacts/models/edge-scene-classifier-v0.2.0 --aliases candidate,rejected-quality,v0.2.0

register-v030:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-edge-object-classifier-0.3.0.yaml --eval-report artifacts/eval/eval_report-v0.3.0.json --model-dir artifacts/models/edge-object-classifier-v0.3.0 --aliases candidate,staging,v0.3.0

register-production-v030:
    docker compose up -d --build wandb trainer
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-edge-object-classifier-0.3.0.yaml --eval-report artifacts/eval/eval_report-v0.3.0.json --model-dir artifacts/models/edge-object-classifier-v0.3.0 --aliases staging,production,v0.3.0

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

promote-production-v030:
    curl -fsS -X POST http://127.0.0.1:${MODEL_PORT_API_PORT:-18080}/models/vendor-demo.edge-object-classifier.0.3.0/promote -H 'Content-Type: application/json' -d '{"target_stage":"production"}'

local-mlops:
    docker compose up -d --build wandb api trainer
    docker compose exec trainer python scripts/prepare_sample_dataset.py --output data/sample_captions.jsonl --num-samples 32
    docker compose exec trainer python -m model_port.pipelines.finetune --config configs/train.smolvlm.yaml
    docker compose exec trainer python -m model_port.pipelines.evaluate --config configs/train.smolvlm.yaml --model-dir artifacts/models/smolvlm2-caption-lora --dataset data/sample_captions.jsonl --output artifacts/eval/eval_report.json
    docker compose exec trainer python -m model_port.registry.build_manifest --base configs/model_manifest.example.yaml --eval-report artifacts/eval/eval_report.json --output artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml
    docker compose exec trainer python -m model_port.registry.wandb_register --manifest artifacts/manifests/vendor-demo-smart-captioner-0.1.0.yaml --eval-report artifacts/eval/eval_report.json --model-dir artifacts/models/smolvlm2-caption-lora
