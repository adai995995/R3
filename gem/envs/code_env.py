from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

from datasets import load_dataset, Dataset

from gem.utils.sandbox import run_python


@dataclass
class CodeEnv:
    """Compatibility shim for `gem.envs.code_env.CodeEnv`.

    The ROLL wrapper (`roll.pipeline.agentic.env.gem.code_env.CodeEnv`) calls:
      - `self._check_correct(model_code)`
    and relies on dataset fields for `problem`/`tests` by default.
    """

    dataset_name: str = ""
    split: Optional[str] = None
    dataset: Optional[Dataset] = None
    question_key: str = "problem"
    test_key: str = "tests"
    seed: int = 0
    max_workers: int = 5
    max_tests: int = 12
    verbose: bool = False
    sandbox_type: str = "none"

    def __post_init__(self):
        if self.dataset is None:
            if not self.dataset_name:
                raise ValueError("CodeEnv requires dataset_name or dataset")
            ds = load_dataset(self.dataset_name, split=self.split or "train")
            assert isinstance(ds, Dataset)
            self.dataset = ds
        self.idx = 0
        self.epoch = 0

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
        self.tests = data[self.test_key]
        self.idx += 1
        return self.first_obs, {}

    def _check_correct(self, model_code: str) -> bool:
        # Very lightweight: treat tests as a list of python snippets that should exit 0.
        tests: Any = self.tests
        if isinstance(tests, str):
            tests_list: List[str] = [tests]
        elif isinstance(tests, list):
            tests_list = [t for t in tests if isinstance(t, str)]
        else:
            tests_list = []

        if self.max_tests and len(tests_list) > self.max_tests:
            tests_list = tests_list[: self.max_tests]

        for t in tests_list:
            code = f"{model_code}\n\n{t}\n"
            res: Dict[str, Any] = run_python(code, timeout_s=5.0)
            if res.get("exit_code", 1) != 0:
                return False
        return True


__all__ = ["CodeEnv"]

