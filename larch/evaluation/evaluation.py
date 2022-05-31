from typing import List, Dict

from .bleu import calc_bleu, calc_bleus
from .rouge import calc_rouge
from .postprocess import extract_raw_text


def calc_all_metrics(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    predictions_raw_text = [extract_raw_text(text, max_blocks=10) for text in predictions]
    references_raw_text = [extract_raw_text(text, max_blocks=10) for text in references]
    result = {
        'rouge': calc_rouge(predictions, references),
        'bleu': calc_bleu(predictions, references),
        'bleus': calc_bleus(predictions, references),
        'rouge_raw_text': calc_rouge(predictions_raw_text, references_raw_text),
        'bleu_raw_text': calc_bleu(predictions_raw_text, references_raw_text),
        'bleus_raw_text': calc_bleus(predictions_raw_text, references_raw_text)
    }
    return result
