from typing import Optional

from transformers import PreTrainedTokenizerBase

from .common import extract_all_files, truncate_by_token_length, deterministic_shuffle_files
from .model import Directory


def create_context(
        root_dir: Directory,
        *,
        use_prompt: bool,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        project_name: Optional[str]) -> str:
    assert (tokenizer is None) == (max_tokens is None)
    if use_prompt:
        if project_name is not None:
            project_str = f' called "{project_name}"'
        else:
            project_str = ''

        prompt_prefix = f'Here are files from a Python project{project_str}:\n\n===\n'
        prompt_suffix = '\n===\n\nWrite a detailed readme in markdown:\n\n===\n'
    else:
        prompt_prefix = ''
        prompt_suffix = ''

    separator = '\n===\n'
    num_prompt_tokens = (
        len(tokenizer.encode(
            prompt_prefix, return_tensors='pt', verbose=False,
            add_special_tokens=False)[0]) +
        len(tokenizer.encode(
            prompt_suffix, return_tensors='pt', verbose=False,
            add_special_tokens=False)[0]))

    # Make it random, but, at the same time, deterministic
    files = list(extract_all_files(root_dir))
    deterministic_shuffle_files(files)

    context = ''
    for file in files:
        if file.excluded:
            continue
        context += file.content.strip()
        if tokenizer is not None:
            context_ = truncate_by_token_length(
                tokenizer, context, max_tokens - num_prompt_tokens)
            if len(context_) < (len(context)):
                return prompt_prefix + context_ + prompt_suffix
        context += separator
    # rstrip separator
    context = context[:-len(separator)]
    if tokenizer is not None:
        # max_tokens - 1 because we append \n
        context = truncate_by_token_length(
            tokenizer, context, max_tokens - num_prompt_tokens)

    return prompt_prefix + context + prompt_suffix
