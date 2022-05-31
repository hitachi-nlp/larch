from typing import Dict, List, Optional, Union

from larch.context_creator.model import File
from larch.utils import construct_word_matcher
from .common import PathSpec, extract_python_files, FeatureMixin

_ARGUMENT_PARSER_KEYWORDS = {
    'ArgumentParser',
    'OptionParser',
    'import click',
    r'click\.command',
    'import docopt',
    'HfArgumentParser'
}
_contains_argument_parser = construct_word_matcher(_ARGUMENT_PARSER_KEYWORDS)

_MAIN_FUNC_KEYWORDS = {
    'def +main',
    '__main__'
}
_contains_main_func = construct_word_matcher(_MAIN_FUNC_KEYWORDS)


_WEB_FRAMEWORK_KEYWORDS = {
    'from +bottle',
    'import +bottle',
    r'Flask\('
}
_contains_web_framework = construct_word_matcher(_WEB_FRAMEWORK_KEYWORDS)


class ContentFeatures(FeatureMixin):
    _TOO_SHORT_THRESH = 200

    def __init__(
            self,
            contains_main_func: bool,
            contains_argument_parser: bool,
            contains_web_framework: bool,
            content_chars: int
    ):
        self._contains_main_func: bool = contains_main_func
        self._contains_argument_parser: bool = contains_argument_parser
        self._contains_web_framework: bool = contains_web_framework
        self._content_chars: int = content_chars
        self._too_short: bool = self._is_too_short(content_chars)

    @classmethod
    def _is_too_short(cls, content_chars: int) -> bool:
        return content_chars < cls._TOO_SHORT_THRESH

    def get_pseudo_label_array(self) -> List[Optional[bool]]:
        return [
            True if self._contains_main_func else None,
            True if self._contains_argument_parser else None,
            True if self._contains_web_framework else None,
            False if self._too_short else None
        ]

    @classmethod
    def get_pseudo_label_names(cls) -> List[str]:
        return [
            'contains_main_func',
            'contains_argument_parser',
            'contains_web_framework',
            'too_short'
        ]

    def get_feature_array(self) -> List[Union[float, bool, int, None]]:
        return [
            bool(self._contains_main_func),
            bool(self._contains_argument_parser),
            bool(self._contains_web_framework),
            float(self._content_chars)
        ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return [
            'contains_main_func',
            'contains_argument_parser',
            'contains_web_framework',
            'content_chars'
        ]


def extract_content_features(files: Dict[PathSpec, File]) -> Dict[PathSpec, ContentFeatures]:
    files = extract_python_files(files)
    features = dict()
    for path, f in files.items():
        features[path] = ContentFeatures(
            contains_main_func=_contains_main_func(f.content),
            contains_argument_parser=_contains_argument_parser(f.content),
            contains_web_framework=_contains_web_framework(f.content),
            content_chars=len(f.content)
        )
    return features
