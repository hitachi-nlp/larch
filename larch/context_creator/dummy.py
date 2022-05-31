from typing import Optional

from transformers import PreTrainedTokenizerBase

from .model import Directory


def create_context(
        root_dir: Directory,
        *,
        use_prompt: bool,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        project_name: Optional[str]) -> str:
    # we don't actually use any of the input but we leave there for consistency
    assert (tokenizer is None) == (max_tokens is None)
    return ''
