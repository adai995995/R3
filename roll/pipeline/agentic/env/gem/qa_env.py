import random
import re
import string
from typing import Tuple, Any, SupportsFloat, Optional, Iterable, List

from datasets import Dataset
from gem.envs.qa_env import QaEnv as GEMQaEnv
from gem.core import Env
from gem.utils.constants import TERMINAL_STATE

_PLACEHOLDER_ANSWERS = {
    "<answer>",
    "<ans>",
    "<final>",
    "none",
    "n/a",
    "null",
}


def _normalize_answer(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    # remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # collapse whitespace
    s = " ".join(s.split())
    return s


def _tokenize(s: str) -> List[str]:
    s = _normalize_answer(s)
    return s.split() if s else []


def _f1_score(pred: str, gold: str) -> float:
    pt = _tokenize(pred)
    gt = _tokenize(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common = {}
    for t in pt:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gt:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    return 2 * precision * recall / (precision + recall)


def _best_match_metrics(pred: str, golds: Iterable[str]) -> dict[str, float]:
    pred_n = _normalize_answer(pred)
    best = {"em": 0.0, "contains": 0.0, "f1": 0.0}
    for g in golds:
        g_n = _normalize_answer(g)
        if not g_n:
            continue
        em = 1.0 if pred_n == g_n else 0.0
        contains = 1.0 if (g_n in pred_n or pred_n in g_n) else 0.0
        f1 = _f1_score(pred, g)
        if (em, contains, f1) > (best["em"], best["contains"], best["f1"]):
            best = {"em": em, "contains": contains, "f1": f1}
    return best


def _coerce_gold_answers(ans: Any) -> List[str]:
    if ans is None:
        return []
    if isinstance(ans, (list, tuple, set)):
        out = []
        for x in ans:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(ans).strip()
    if not s:
        return []
    # allow common alias separators (best-effort)
    if "||" in s:
        parts = [p.strip() for p in s.split("||")]
    elif "|" in s:
        parts = [p.strip() for p in s.split("|")]
    else:
        parts = [s]
    return [p for p in parts if p]


def _extract_final_answer(action: str) -> Optional[str]:
    if not isinstance(action, str) or not action.strip():
        return None
    # Prefer the last line that starts with 'FINAL:' (case-insensitive).
    lines = [ln.strip() for ln in action.splitlines() if ln.strip()]
    for ln in reversed(lines):
        lnl = ln.lower()
        # tolerate "Assistant: FINAL: ..."
        if lnl.startswith("assistant:"):
            ln = ln.split(":", 1)[1].strip()
            lnl = ln.lower()
        if lnl.startswith("final:"):
            ans = ln[len("final:"):].strip()
            if not ans:
                return None
            ans_n = _normalize_answer(ans)
            if not ans_n:
                return None
            if ans_n in _PLACEHOLDER_ANSWERS:
                return None
            if "<answer" in ans_n or "final:" in ans_n:
                return None
            return ans
    return None

class QaEnv(GEMQaEnv):
    def __init__(
        self,
        dataset_name: Optional[str] = "",
        split: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        question_key: str = "question",
        answer_key: str = "answer",
        seed: int = 0,
        extract_boxed: bool = False,
        load_from_cache_file: bool = True,  # False to force re-run the apply_prompt_func, useful when apply_prompt is changed
        **_,
    ):
        from datasets import tqdm
        tqdm.set_lock(tqdm.get_lock())
        super().__init__(dataset_name=dataset_name,
                         split=split,
                         dataset=dataset,
                         question_key=question_key,
                         answer_key=answer_key,
                         seed=seed,
                         extract_boxed=extract_boxed,
                         load_from_cache_file=load_from_cache_file, **_)
        # Make answer extraction robust to extra reasoning: require `FINAL: <answer>` in the output.
        self.extractor = _extract_final_answer

    def step(
        self, action: str
    ) -> Tuple[str, SupportsFloat, bool, bool, dict[str, Any]]:
        model_answer = self.extractor(action)
        action_is_valid = True
        if model_answer is None:
            reward = 0.0
            action_is_valid = False
        else:
            golds = _coerce_gold_answers(self.answer)
            m = _best_match_metrics(model_answer, golds)
            # Use F1 as a dense reward to "unlock" learning signal; success is relaxed.
            reward = float(m["f1"])
        metrics = {
            "action_is_valid": action_is_valid,
            # relaxed success: EM OR contains OR reasonably-high F1
            "success": (reward >= 0.5),
            "answer_em": 1.0 if action_is_valid and reward >= 1.0 else 0.0,
            "answer_f1": reward,
            "raw_reward": reward,
        }
        metrics_agg_mode = {
            "action_is_valid": "mean",
            "success": "last",
            "answer_em": "last",
            "answer_f1": "last",
            "raw_reward": "last",
        }
        info = {
            "metrics": metrics,
            "metrics_agg_mode": metrics_agg_mode
        }
        return TERMINAL_STATE, reward, True, True, info

    def reset(self, seed: Optional[None] = None) -> Tuple[str, dict[str, Any]]:
        """Sample a question from the dataset."""
        Env.reset(self, seed)
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