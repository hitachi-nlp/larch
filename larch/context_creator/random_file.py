from typing import Optional

from transformers import PreTrainedTokenizerBase

from larch.entrypoint_extractor.features.common import match_python_file
from .common import (
    extract_all_files,
    truncate_by_token_length,
    deterministic_shuffle_files,
    cleanse_file
)
from .file_names import create_context as _create_context_using_file_names
from .model import Directory


def create_context(
        root_dir: Directory,
        *,
        use_prompt: bool,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        project_name: Optional[str]) -> str:
    assert (tokenizer is None) == (max_tokens is None)
    # Make it random, but, at the same time, deterministic
    files = list(extract_all_files(root_dir))
    deterministic_shuffle_files(files)
    non_excluded_files = [
        file for file in files
        if not file.excluded and match_python_file(file.name) is not None
    ]
    if len(non_excluded_files) == 0:
        print('No candidate (non-excluded Pythonfile) for random file is found. '
              'Falling back to .file_names.create_context.')
        return _create_context_using_file_names(
            root_dir,
            use_prompt=use_prompt,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            project_name=project_name
        )

    input_file = non_excluded_files[0]
    content = cleanse_file(input_file.content)

    if use_prompt:
        if project_name is not None:
            project_str = f' called "{project_name}"'
        else:
            project_str = ''

        prompt_prefix = f'Here is a file from a Python project{project_str}:\n\n===\n'
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
    content = truncate_by_token_length(tokenizer, content, max_tokens - num_prompt_tokens)
    return prompt_prefix + content + prompt_suffix
