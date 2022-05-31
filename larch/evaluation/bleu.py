from typing import List

import evaluate

_metric = evaluate.load("bleu")


def calc_bleu(predictions: List[str], references: List[str]) -> dict:
    return _metric.compute(predictions=predictions, references=references)


def calc_bleus(predictions: List[str], references: List[str]) -> dict:
    return _metric.compute(predictions=predictions, references=references, smooth=True)
