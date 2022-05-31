import re
from typing import Tuple, Dict, List, Optional, Union

from larch.context_creator.model import Directory, File

PathSpec = Tuple[str, ...]

_RE_PYTHON_FILE = re.compile('(?P<base>[a-zA-Z_0-9]+)\\.py[cx]?$')


def match_python_file(filename: str) -> Optional[str]:
    m = _RE_PYTHON_FILE.match(filename)
    return None if m is None else m.group('base')


def extract_python_files(files: Dict[PathSpec, File]) -> Dict[PathSpec, File]:
    extracted_files = dict()
    for path, file in files.items():
        if file.excluded:
            continue
        base_name = match_python_file(file.name)
        if base_name is not None:
            extracted_files[path] = file
    return extracted_files


def _aggregate_files(dir_tree: Directory, parents: List[str]) -> Dict[PathSpec, File]:
    files: Dict[PathSpec, File] = dict()
    for c in dir_tree.children:
        if isinstance(c, Directory):
            _files = _aggregate_files(c, parents + [c.name])
            n_combined = len(files) + len(_files)
            files.update(_files)
            assert len(files) == n_combined
        elif isinstance(c, File):
            files[tuple(parents + [c.name])] = c
        else:
            assert not 'Should not reach here'
    return files


def aggregate_files(dir_tree: Directory) -> Dict[PathSpec, File]:
    return _aggregate_files(dir_tree, [])


class FeatureMixin:
    def get_pseudo_label_array(self) -> List[Optional[bool]]:
        raise NotImplementedError()

    @classmethod
    def get_pseudo_label_names(cls) -> List[str]:
        raise NotImplementedError()

    @classmethod
    def get_pseudo_label_dim(cls) -> int:
        return len(cls.get_pseudo_label_names())

    def get_feature_array(self) -> List[Union[float, bool, int, None]]:
        raise NotImplementedError()

    @classmethod
    def get_feature_names(cls) -> List[str]:
        raise NotImplementedError()

    @classmethod
    def get_feature_dim(cls) -> int:
        return len(cls.get_feature_names())
