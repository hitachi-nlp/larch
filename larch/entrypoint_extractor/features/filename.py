from typing import Dict, List, Optional, Union

from larch.context_creator.model import File
from .common import PathSpec, extract_python_files, FeatureMixin


_MAIN_ISH_NAMES = {
    'cli.py',
    '_cli.py',
    'app.py'
}


def _is_main_ish_name(filename: str) -> bool:
    return filename in _MAIN_ISH_NAMES


class FilenameFeatures(FeatureMixin):
    def __init__(
            self,
            has_main_in_filename: bool,
            is_main_ish_names: bool,
            is_module: bool,
            is_test_ish_name: bool,
            has_same_name_as_repo: bool,
            level: int
    ):
        self._has_main_in_filename: bool = has_main_in_filename
        self._is_main_ish_names: bool = is_main_ish_names
        self._is_module: bool = is_module
        self._is_test_ish_name: bool = is_test_ish_name
        self._has_same_name_as_repo: bool = has_same_name_as_repo
        self._level: int = level

    def get_pseudo_label_array(self) -> List[Optional[bool]]:
        return [
            True if self._has_main_in_filename else None,
            True if self._is_main_ish_names else None,
            False if self._is_module else None,
            False if self._is_test_ish_name else None,
            True if self._has_same_name_as_repo else None
        ]

    @classmethod
    def get_pseudo_label_names(cls) -> List[str]:
        return [
            'has_main_in_filename',
            'is_main_ish_names',
            'is_module',
            'is_test_ish_name',
            'has_same_name_as_repo'
        ]

    def get_feature_array(self) -> List[Union[float, bool, int, None]]:
        return [
            bool(self._has_main_in_filename),
            bool(self._is_main_ish_names),
            bool(self._is_module),
            float(self._level),
            bool(self._is_test_ish_name)
        ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return [
            'has_main_in_filename',
            'is_main_ish_names',
            'is_module',
            'level',
            'is_test_ish_name'
        ]


def extract_filename_features(
        files: Dict[PathSpec, File],
        repo_name: Optional[str]) -> Dict[PathSpec, FilenameFeatures]:
    files = extract_python_files(files)
    features = dict()
    for path, f in files.items():
        fn = f.name.lower()
        features[path] = FilenameFeatures(
            has_main_in_filename='main' in fn,
            is_main_ish_names=_is_main_ish_name(fn),
            is_module=fn == '__init__.py',
            is_test_ish_name=fn.startswith('test_'),
            has_same_name_as_repo=False if repo_name is None else fn.split('.')[0] == repo_name,
            level=len(path) - 1
        )
    return features
