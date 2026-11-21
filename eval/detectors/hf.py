"""공개 HuggingFace 분류 모델 Detector 어댑터 (S6-2).

등록된 탐지기:

- ``protectai-v2``: ``protectai/deberta-v3-base-prompt-injection-v2`` (Apache-2.0, 영어 학습)
- ``promptguard2-86m``: ``meta-llama/Llama-Prompt-Guard-2-86M`` (Llama 4 Community License, gated.
  HF 에서 라이선스 동의 + ``hf auth login`` 필요. 다국어 mDeBERTa 기반)
- ``promptguard2-22m``: ``meta-llama/Llama-Prompt-Guard-2-22M`` (gated, 영어 중심 경량)

점수 = softmax 확률 중 **공격 라벨**들의 합. 공격 라벨은 ``config.id2label`` 의 이름으로 찾는다
(대소문자 무시). 입력은 ``max_length`` 토큰에서 잘린다 (긴 입력 앞부분만 본다).

torch·transformers 는 탐지기를 만들 때(``HFClassifierDetector(...)``) import 한다.
레지스트리 등록만으로는 로딩되지 않으므로 ``pip install -e .[train]`` 없이도
``python -m eval.run --list`` 등은 동작한다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from eval.detector import register_detector

__all__ = ["HF_MODELS", "HFClassifierDetector", "resolve_attack_ids"]

# 이름 → (model_id, 공격 라벨 이름들)
HF_MODELS: dict[str, tuple[str, tuple[str, ...]]] = {
    "protectai-v2": ("protectai/deberta-v3-base-prompt-injection-v2", ("INJECTION",)),
    "promptguard2-86m": ("meta-llama/Llama-Prompt-Guard-2-86M", ("MALICIOUS", "LABEL_1")),
    "promptguard2-22m": ("meta-llama/Llama-Prompt-Guard-2-22M", ("MALICIOUS", "LABEL_1")),
}


def resolve_attack_ids(
    id2label: Mapping[int | str, str], attack_labels: Iterable[str]
) -> list[int]:
    """``id2label`` 에서 공격 라벨 이름(대소문자 무시)에 해당하는 클래스 id 목록.

    하나도 못 찾으면 ``ValueError`` (라벨 이름이 바뀐 모델을 조용히 잘못 채점하지 않도록).
    """
    wanted = {a.upper() for a in attack_labels}
    ids = sorted(int(i) for i, lab in id2label.items() if str(lab).upper() in wanted)
    if not ids:
        raise ValueError(
            f"공격 라벨 {sorted(wanted)} 을 id2label 에서 찾지 못했습니다: {dict(id2label)}"
        )
    if len(ids) == len(id2label):
        raise ValueError(f"모든 라벨이 공격 라벨로 지정되었습니다: {dict(id2label)}")
    return ids


class HFClassifierDetector:
    """``AutoModelForSequenceClassification`` 기반 탐지기 (CPU 추론)."""

    def __init__(
        self,
        model_id: str,
        attack_labels: Iterable[str],
        *,
        name: str | None = None,
        threshold: float = 0.5,
        max_length: int = 512,
        batch_size: int = 16,
        revision: str | None = None,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold 는 0~1 범위여야 합니다: {threshold}")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - 환경 의존
            raise ImportError(
                "공개 모델 탐지기에는 torch·transformers 가 필요합니다: pip install -e .[train]"
            ) from e

        self._torch = torch
        self.model_id = model_id
        self.name = name or model_id.rsplit("/", 1)[-1]
        self.threshold = float(threshold)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision)
        self.model.to(device).eval()
        self.attack_ids = resolve_attack_ids(self.model.config.id2label, attack_labels)
        # 실제 사용한 모델 커밋 (재현용): HF 캐시 스냅샷 디렉터리 이름이 커밋 해시다
        self.revision = _cached_commit(model_id, revision) or revision

    def predict(self, texts: Sequence[str]) -> list[float]:
        torch = self._torch
        out: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = list(texts[i : i + self.batch_size])
                enc = self.tokenizer(
                    batch,
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1)
                out.extend(probs[:, self.attack_ids].sum(dim=-1).clamp(0.0, 1.0).tolist())
        return out


def _cached_commit(model_id: str, revision: str | None) -> str | None:
    """HF 캐시에 받은 ``config.json`` 경로에서 커밋 해시를 읽는다. 못 찾으면 None."""
    from pathlib import Path

    from huggingface_hub import try_to_load_from_cache

    if Path(model_id).is_dir():  # 로컬 디렉터리 모델 (테스트·자체 체크포인트)
        return None
    path = try_to_load_from_cache(model_id, "config.json", revision=revision)
    return Path(path).parent.name if isinstance(path, str) else None


def _make_factory(name: str):
    model_id, labels = HF_MODELS[name]

    def factory(**kwargs) -> HFClassifierDetector:
        return HFClassifierDetector(model_id, labels, name=name, **kwargs)

    factory.__name__ = f"_{name.replace('-', '_')}"
    return factory


for _name in HF_MODELS:
    register_detector(_name)(_make_factory(_name))
