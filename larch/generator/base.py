from typing import Tuple

from transformers import PreTrainedTokenizerBase


class BaseGenerator:

    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_context_length: int,
                 is_pretrained: bool):
        self._tokenizer = tokenizer
        self._max_context_length = max_context_length
        self._is_pretrained = is_pretrained

    def generate(self, context: str, prompt: str, max_length: int) -> Tuple[str, str]:
        raise NotImplementedError()

    def calculate_perplexity(self, context: str, content: str, num_content_tokens: int) -> float:
        raise NotImplementedError()

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    @property
    def is_pretrained(self) -> bool:
        return self._is_pretrained
