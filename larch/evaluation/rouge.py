# coding=utf-8
# This file was adopted from https://github.com/huggingface/transformers/blob/v4.23.0/examples/pytorch/summarization/run_summarization.py
# See modifications that we made in git history. Otherwise, following copyrights apply:
#
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Dict

import evaluate
import nltk
from filelock import FileLock
from transformers.utils import is_offline_mode

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


_metric = evaluate.load("rouge")


def _postprocess_text(text: str) -> str:
    # rougeLSum expects newline after each sentence
    return "\n".join(nltk.sent_tokenize(text.strip()))


def calc_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    predictions = [_postprocess_text(text) for text in predictions]
    references = [_postprocess_text(text) for text in references]

    return _metric.compute(predictions=predictions, references=references, use_stemmer=True)
