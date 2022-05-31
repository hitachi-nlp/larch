from typing import Dict, Tuple, Optional

import xgboost as xgb
from transformers import PreTrainedTokenizerBase

from larch.entrypoint_extractor import load_single_data, predict
from .common import truncate_by_token_length, deterministic_shuffle_files, cleanse_file
from .model import Directory, File
from .random_file import create_context as _create_context_using_all_files

model = None

# This hardcoded hyperparameter defines the number of files added to the context
# set it to 0 and it will not add any file name
N_FILE_NAMES = 10


def init(model_path: str):
    global model
    model = xgb.XGBRanker()
    model.load_model(model_path)


def _sample_file_names(files: Dict[Tuple[str, ...], File]) -> str:
    if len(files) == 0:
        return ''
    files = list(files.values())
    deterministic_shuffle_files(files)
    return ' '.join((f.name for f in files[:N_FILE_NAMES]))


def create_context(
        root_dir: Directory,
        *,
        use_prompt: bool,
        tokenizer: PreTrainedTokenizerBase,
        max_tokens: int,
        project_name: Optional[str]) -> str:
    assert (tokenizer is None) == (max_tokens is None)
    global model
    assert model is not None
    path_to_features, files = load_single_data('repo', root_dir, None, None, None, timeout=60)
    if len(path_to_features) == 0:
        print('No candidate (python files) for entrypoint is found. '
              'Falling back to .random_file.create_context.')
        return _create_context_using_all_files(
            root_dir,
            use_prompt=use_prompt,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            project_name=project_name
        )
    best_path, _ = predict(model, path_to_features)
    input_file = files[best_path]
    content = cleanse_file(input_file.content)

    file_name_str = _sample_file_names(files)
    if use_prompt:
        if project_name is not None:
            project_str = f' called "{project_name}"'
        else:
            project_str = ''

        prompt_prefix = f'Here is the entrypoint of a Python project{project_str}:\n\n===\n'
        prompt_suffix = '\n===\n\nWrite a detailed readme in markdown:\n\n===\n'
        if len(file_name_str) > 0:
            prompt_suffix = (
                f'\n===\n\nThis program has following files:\n\n===\n{file_name_str}'
                + prompt_suffix)
    else:
        prompt_prefix = ''
        if N_FILE_NAMES > 0:
            prompt_suffix = f'\n===\n{file_name_str}'
        else:
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
