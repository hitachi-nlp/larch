import hashlib
import random
import re
from typing import Generator, List

import numpy as np
from transformers import PreTrainedTokenizerBase

from larch.entrypoint_extractor.features.imports import RE_IMPORT
from larch.utils import construct_word_matcher
from .model import Directory, File


# FIXME: This is redundant to larch.entrypoint_extractor.features.common.aggregate_files
def extract_all_files(root_dir: Directory) -> Generator[File, None, None]:
    for child in root_dir.children:
        if isinstance(child, Directory):
            yield from extract_all_files(child)
        else:
            yield child


def truncate_by_token_length(tokenizer: PreTrainedTokenizerBase, text: str, max_length: int):
    assert max_length > 0
    tokens = tokenizer(
        text, return_tensors='np', return_offsets_mapping=True, verbose=False)
    input_ids = tokens['input_ids'][0]
    if len(input_ids) <= max_length:
        return text
    offset_mapping = tokens['offset_mapping'][0]
    context_end = int(np.argmin(offset_mapping[:, 1] < len(text)))
    truncated_context = input_ids[:min(context_end, max_length)]
    return tokenizer.decode(truncated_context, skip_special_tokens=True)


def deterministic_shuffle_files(files: List[File]) -> None:
    # Make it random, but, at the same time, deterministic
    # note that this shuffles files in place
    files = sorted(files, key=lambda f: f.name)
    file_names = ''.join((f.name for f in files))
    hash_val = hashlib.sha256(file_names.encode('utf-8', errors='ignore')).digest()
    random.seed(hash_val)
    random.shuffle(files)
    return None



_HEADER_KEYWORDS = {
    'license',
    'copyright'
}
_contains_header_keywords = construct_word_matcher(_HEADER_KEYWORDS)


def _remove_header(content: str) -> str:
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        if len(lines[i].strip()) == 0 or lines[i].startswith('#'):
            i += 1
            continue
        break
    if i == 0:
        # no header
        return content
    header = '\n'.join(lines[:i])
    if _contains_header_keywords(header.lower()):
        return '\n'.join(lines[i:])
    else:
        return content


_RE_MULTIPLE_CR = re.compile('\n\n+')


def cleanse_file(content: str) -> str:
    content = _RE_MULTIPLE_CR.sub('\n\n', content.replace('\r', '\n'))
    # Remove the header
    content = _remove_header(content)

    # Remove all imports
    content = RE_IMPORT.sub('', content)

    content = _RE_MULTIPLE_CR.sub('\n\n', content)
    return content.strip()
