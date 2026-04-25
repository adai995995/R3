from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from datasets import load_dataset, Dataset


def _normalize_answer(x: str) -> str:
    return " ".join(str(x).strip().split()).lower()


@dataclass
class QaEnv:
    """Compatibility shim for `gem.envs.qa_env.QaEnv`.

    Provides dataset loading + a simple `check_correct` implementation.
    The ROLL wrapper may override reset/step, but expects attributes like:
    - dataset, idx, epoch
    - extractor(action) -> str|None
    """

    dataset_name: str = ""
    split: Optional[str] = None
    dataset: Optional[Dataset] = None
    question_key: str = "question"
    answer_key: str = "answer"
    seed: int = 0
    extract_boxed: bool = False
    load_from_cache_file: bool = True

    def __post_init__(self):
        if self.dataset is None:
            if not self.dataset_name:
                raise ValueError("QaEnv requires dataset_name or dataset")
            ds = load_dataset(self.dataset_name, split=self.split or "train")
            assert isinstance(ds, Dataset)
            self.dataset = ds
        self.idx = 0
        self.epoch = 0

        # Basic extractor: return the raw string (wrappers can override).
        self.extractor = lambda s: s

    def check_correct(self, model_answer: str, gold_answer: str) -> bool:
        return _normalize_answer(model_answer) == _normalize_answer(gold_answer)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.idx = random.randint(0, len(self.dataset) - 1)
        else:
            if self.idx == len(self.dataset):
                self.epoch += 1
                self.dataset = self.dataset.shuffle(seed=self.seed + self.epoch)
                self.idx = 0
        data = self.dataset[self.idx]
        self.first_obs = data[self.question_key]
        self.answer = data[self.answer_key]
        self.idx += 1
        return self.first_obs, {}


__all__ = ["QaEnv"]

