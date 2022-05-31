import re
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional, Union

from larch.context_creator.model import File
from .common import PathSpec, extract_python_files, FeatureMixin

_RE_CLASS = re.compile(
    '(?:(?:\n *)|(?:^ *))class +(?P<class>[^ (:]+) *(?:\\((?P<base>[^(:]*)\\) *)?:',
    flags=re.DOTALL
)
_RE_BASE_CLASS_SUB = re.compile(
    '[\r\n\\\\]', flags=re.DOTALL
)

ClassInheritance = Tuple[str, Set[str]]


def _extract_class(text: str) -> List[ClassInheritance]:
    # you can have parentheses or comma within base classes (like using a
    # factory of a class), but we ignore those rare cases
    classes = []
    for m in _RE_CLASS.finditer(text):
        class_name = m.group('class')
        assert class_name is not None
        base_class_str = m.group('base')
        if base_class_str is not None:
            base_classes = {
                c.strip()
                for c in _RE_BASE_CLASS_SUB.sub(' ', base_class_str).split(',')
                if len(c.strip()) > 0
            }
        else:
            base_classes = set()
        classes.append((class_name, base_classes))
    return classes


def _count_inheritance(classes_per_file: Dict[PathSpec, List[ClassInheritance]]) -> Dict[PathSpec, int]:
    counts = {path: 0 for path in classes_per_file.keys()}
    # we ignore import relationships and allow multiple paths for a single class name
    class_to_paths = defaultdict(set)
    for path, classes in classes_per_file.items():
        for cls_name, _ in classes:
            class_to_paths[cls_name].add(path)
    class_to_paths = dict(class_to_paths)
    for path, classes in classes_per_file.items():
        for _, base_classes in classes:
            for cls_name in base_classes:
                for path in class_to_paths.get(cls_name, []):
                    counts[path] += 1
    return counts


class InheritanceFeatures(FeatureMixin):
    _INHERITANCE_THRESH = 3

    def __init__(
            self,
            num_inherited: int
    ):
        self._num_inherited: bool = num_inherited

    def get_pseudo_label_array(self) -> List[Optional[bool]]:
        return [
            True if self._num_inherited >= self._INHERITANCE_THRESH else None
        ]

    @classmethod
    def get_pseudo_label_names(cls) -> List[str]:
        return [
            'inherited_many_times'
        ]

    def get_feature_array(self) -> List[Union[float, bool, int, None]]:
        return [
            float(self._num_inherited)
        ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return [
            'num_inherited'
        ]


def extract_inheritance_features(files: Dict[PathSpec, File]) -> Dict[PathSpec, InheritanceFeatures]:
    files = extract_python_files(files)
    files = {path: f.content for path, f in files.items()}
    classes_per_file = {
        path: _extract_class(content)
        for path, content in files.items()
    }
    num_inherited_per_file = _count_inheritance(classes_per_file)
    # FIXME: Stop using random threshold
    return {
        path: InheritanceFeatures(num_inherited)
        for path, num_inherited in num_inherited_per_file.items()
    }
