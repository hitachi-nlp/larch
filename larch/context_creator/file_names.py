from typing import Optional

from transformers import PreTrainedTokenizerBase

from .common import deterministic_shuffle_files
from .common import extract_all_files, truncate_by_token_length
from .model import Directory


def create_context(
        root_dir: Directory,
        *,
        use_prompt: bool,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        project_name: Optional[str]) -> str:
    assert tokenizer is not None
    assert max_tokens is not None
    if use_prompt:
        if project_name is not None:
            project_str = f' called "{project_name}"'
        else:
            project_str = ''

        prompt_prefix = f'Here is a a Python project{project_str} with following files:\n\n===\n'
        prompt_suffix = '\n===\n\nWrite a detailed readme in markdown:\n\n===\n'
    else:
        prompt_prefix = ''
        prompt_suffix = ''
    num_prompt_tokens = (
        len(tokenizer.encode(
            prompt_prefix, return_tensors='pt', verbose=False,
            add_special_tokens=False)[0]) +
        len(tokenizer.encode(
            prompt_suffix, return_tensors='pt', verbose=False,
            add_special_tokens=False)[0]))
    assert num_prompt_tokens < max_tokens

    files = list(extract_all_files(root_dir))
    deterministic_shuffle_files(files)
    content = ' '.join([f.name for f in files])

    content = truncate_by_token_length(tokenizer, content, max_tokens - num_prompt_tokens)
    return prompt_prefix + content + prompt_suffix
