import re
from typing import Set, Callable


def construct_word_matcher(patterns: Set[str]) -> Callable[[str], bool]:
    pat = re.compile(rf'\b(?:{"|".join(patterns)})\b')

    def _func(content: str) -> bool:
        return pat.search(content) is not None

    return _func
