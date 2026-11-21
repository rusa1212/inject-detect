"""인젝션 분류기 파인튜닝 스크립트 (S7-1).

    python -m training.train --config training/configs/mdeberta-v3-base.yaml
    python -m training.train --config training/configs/klue-roberta-base.yaml --set epochs=5

Colab 노트북(``training/train_colab.ipynb``)도 이 스크립트를 그대로 호출한다 (로직 단일화).

- 데이터: ``data/processed/{train,valid}.jsonl`` (``python -m data.build`` 로 생성)
- 매 에폭 끝에 valid 를 평가 프레임워크(``eval.metrics.compute_metrics``)로 채점하고,
  ``best_metric`` (기본 ``ko.f1``) 이 가장 좋은 체크포인트를 ``<run_dir>/best/`` 에 저장한다.
- 출력 (``training/outputs/<run_name>/``, 레포 미포함)::

    config.yaml     실제로 쓴 설정
    history.jsonl   에폭별 train loss·valid 지표
    summary.json    최적 에폭·valid 지표·환경 정보
    best/           최적 체크포인트 (save_pretrained 형식)

라벨: 0 = SAFE, 1 = INJECTION. 점수 = softmax 의 INJECTION 확률.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import yaml

from eval.dataset import Sample, load_split
from eval.metrics import EvalMetrics, compute_metrics
from training.config import ROOT, TrainConfig, load_config

__all__ = ["ID2LABEL", "class_weights", "load_samples", "metric_value", "train"]

ID2LABEL = {0: "SAFE", 1: "INJECTION"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
LOG_EVERY = 50  # 스텝


# ---------------------------------------------------------------- 데이터·지표 (torch 불필요)


def load_samples(cfg: TrainConfig, split: str) -> list[Sample]:
    """split 을 읽어 언어 필터·우회 변형 제외·개수 제한을 적용한다."""
    samples = load_split(split, cfg.resolve_path(cfg.data_dir))
    if cfg.langs:
        samples = [s for s in samples if s.lang in cfg.langs]
    if split == "train" and not cfg.train_evasion:
        samples = [s for s in samples if not s.evasion]
    limit = cfg.limit_train if split == "train" else cfg.limit_eval
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError(f"{split} 샘플이 없습니다 (data_dir={cfg.data_dir}, langs={cfg.langs})")
    return samples


def class_weights(labels: Sequence[int], spec: str | list[float] | None) -> list[float] | None:
    """``class_weight`` 설정 → [w0, w1]. ``balanced`` 는 n / (2 * count_c)."""
    if spec is None:
        return None
    if spec == "balanced":
        counts = Counter(labels)
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError(
                f"balanced 가중치를 계산하려면 두 라벨이 모두 필요합니다: {dict(counts)}"
            )
        n = len(labels)
        return [n / (2 * counts[0]), n / (2 * counts[1])]
    return [float(w) for w in spec]


def metric_value(metrics: EvalMetrics, key: str) -> float:
    """``"ko.f1"`` 같은 키로 EvalMetrics 에서 값을 꺼낸다."""
    lang, _, name = key.partition(".")
    if lang not in metrics.by_lang:
        raise KeyError(f"valid 에 '{lang}' 언어가 없습니다 (best_metric={key})")
    return float(getattr(metrics.by_lang[lang], name))


def _summary_metrics(metrics: EvalMetrics) -> dict:
    """로그용 핵심 지표: 언어별 P/R/F1, 정상·하드 네거티브 오탐률, 우회 탐지율."""
    d = metrics.to_dict()
    out: dict = {}
    for lang, m in d["by_lang"].items():
        out[lang] = {k: round(m[k], 4) for k in ("precision", "recall", "f1")}
        fp = d["false_positive"].get(lang, {})
        for kind, r in fp.items():
            out[lang][f"fpr_{kind}"] = None if r["value"] is None else round(r["value"], 4)
        ev = d["evasion"].get(lang, {}).get("all")
        if ev:
            out[lang]["evasion_recall"] = None if ev["value"] is None else round(ev["value"], 4)
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------- 학습 (torch 필요)


def predict_scores(model, tokenizer, texts: Sequence[str], cfg: TrainConfig, device) -> list[float]:
    import torch

    model.eval()
    out: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(texts), cfg.eval_batch_size):
            enc = tokenizer(
                list(texts[i : i + cfg.eval_batch_size]),
                truncation=True,
                max_length=cfg.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            probs = torch.softmax(model(**enc).logits.float(), dim=-1)[:, LABEL2ID["INJECTION"]]
            out.extend(probs.clamp(0.0, 1.0).tolist())
    return out


def train(cfg: TrainConfig) -> dict:
    """설정대로 학습하고 summary dict 를 돌려준다 (``<run_dir>/summary.json`` 에도 저장)."""
    try:
        import torch
        import transformers
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
    except ImportError as e:  # pragma: no cover - 환경 의존
        raise ImportError(
            "학습에는 torch·transformers 가 필요합니다: pip install -e .[train]"
        ) from e

    started = time.time()
    transformers.set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.fp16 and device.type == "cuda"

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    train_samples = load_samples(cfg, "train")
    valid_samples = load_samples(cfg, "valid")
    labels = [s.label for s in train_samples]
    weights = class_weights(labels, cfg.class_weight)
    print(
        f"[data] train {len(train_samples)} {dict(sorted(Counter(labels).items()))} | "
        f"valid {len(valid_samples)} | class_weight {weights} | device {device} amp {use_amp}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    # 가중치는 항상 fp32 로 둔다. mDeBERTa 처럼 fp16 으로 저장된 체크포인트를 transformers 5 는
    # 저장 dtype 그대로 읽어 CPU 에서도 순수 fp16 학습이 되고 loss 가 NaN 이 된다.
    # 혼합 정밀도는 CUDA 에서 autocast(fp16=true) 로만 적용한다.
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
        )
        .float()
        .to(device)
    )

    enc = tokenizer([s.text for s in train_samples], truncation=True, max_length=cfg.max_length)
    features = [
        {"input_ids": ids, "attention_mask": am, "labels": y}
        for ids, am, y in zip(enc["input_ids"], enc["attention_mask"], labels, strict=True)
    ]
    loader = DataLoader(
        features,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    no_decay = ("bias", "LayerNorm.weight", "layernorm", "layer_norm")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate)
    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=None if weights is None else torch.tensor(weights, device=device)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history_path = run_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    best: dict | None = None
    step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"loss 가 {loss.item()} 입니다 (epoch {epoch}, step {step}). "
                    "fp16 이 원인이면 --set fp16=false 로 다시 실행하세요"
                )
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            running += loss.item()
            n_batches += 1
            if i % cfg.grad_accum_steps == 0 or i == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % LOG_EVERY == 0:
                    print(
                        f"[train] epoch {epoch} step {step}/{total_steps} "
                        f"loss {running / n_batches:.4f}",
                        flush=True,
                    )
                if step >= total_steps:
                    break

        scores = predict_scores(model, tokenizer, [s.text for s in valid_samples], cfg, device)
        metrics = compute_metrics(valid_samples, scores, cfg.threshold)
        value = metric_value(metrics, cfg.best_metric)
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": round(running / max(n_batches, 1), 6),
            cfg.best_metric: round(value, 4),
            "valid": _summary_metrics(metrics),
        }
        improved = best is None or value > best["value"]
        if improved:
            best = {"epoch": epoch, "step": step, "value": value, "metrics": metrics}
            if cfg.save_model:
                model.save_pretrained(run_dir / "best")
                tokenizer.save_pretrained(run_dir / "best")
        row["best"] = improved
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[valid] epoch {epoch} loss {row['train_loss']:.4f} {cfg.best_metric} {value:.4f}"
            + (" *best" if improved else ""),
            flush=True,
        )
        if step >= total_steps:
            break

    assert best is not None
    summary = {
        "run_name": cfg.run_name,
        "model_name": cfg.model_name,
        "best_metric": cfg.best_metric,
        "best_value": round(best["value"], 4),
        "best_epoch": best["epoch"],
        "best_step": best["step"],
        "total_steps": step,
        "valid": best["metrics"].to_dict(),
        "data": {"train": len(train_samples), "valid": len(valid_samples), "class_weight": weights},
        "env": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "amp": use_amp,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "git_commit": _git_commit(),
        },
        "elapsed_sec": round(time.time() - started, 1),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[done] best epoch {best['epoch']} {cfg.best_metric} {best['value']:.4f} → {run_dir}",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m training.train", description="인젝션 분류기 파인튜닝"
    )
    p.add_argument("--config", help="YAML 설정 파일 (training/configs/*.yaml)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="설정 덮어쓰기")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(Path(args.config) if args.config else None, args.set)
    except (ValueError, FileNotFoundError) as e:
        print(f"설정 오류: {e}", file=sys.stderr)
        return 2
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
