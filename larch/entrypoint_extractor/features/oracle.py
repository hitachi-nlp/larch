import itertools
import re
from enum import Enum
from typing import Dict, List, Optional, Set, Union

import yaml
import yaml.parser
import yaml.scanner

from larch.context_creator.model import File
from .common import PathSpec, extract_python_files, FeatureMixin, match_python_file
from .imports import identify_imports, extract_import


class MentionFeature(Enum):
    FILE_NOT_AVAILABLE = -1
    FILE_AVAILABLE_BUT_NOT_MENTIONED = 0
    FILE_AVAILABLE_AND_IS_MENTIONED = 1

    def to_pseudo_label(self) -> Optional[bool]:
        if self == MentionFeature.FILE_NOT_AVAILABLE:
            return None
        elif self == MentionFeature.FILE_AVAILABLE_BUT_NOT_MENTIONED:
            return False
        elif self == MentionFeature.FILE_AVAILABLE_AND_IS_MENTIONED:
            return True
        else:
            assert not 'Should not reach here'


class OracleFeatures(FeatureMixin):
    def __init__(
            self,
            is_setup_entrypoint: MentionFeature,
            is_mentioned_in_readme: MentionFeature,
    ):
        self._is_setup_entrypoint: MentionFeature = is_setup_entrypoint
        self._is_mentioned_in_readme: MentionFeature = is_mentioned_in_readme

    def get_pseudo_label_array(self) -> List[Optional[bool]]:
        return [
            self._is_setup_entrypoint.to_pseudo_label(),
            self._is_mentioned_in_readme.to_pseudo_label()
        ]

    @classmethod
    def get_pseudo_label_names(cls) -> List[str]:
        return [
            'is_setup_entrypoint',
            'is_mentioned_in_readme'
        ]

    def get_feature_array(self) -> List[Union[float, bool, int, None]]:
        return []

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return []


# We cannot correctly parse python files with regex due to Chomsky Hierarchy,
# so we put a strong prior on the structure of the extraction target and assume
# that entry_points argument doesn't include a curly bracket
_RE_ENTRYPOINT = re.compile(
    'entry_points[ \n\t\r\\\\]*=[ \n\t\r\\\\]*(?P<entrypoint>\\{[^\\}]*\\})',
    flags=re.DOTALL
)


def _extract_entrypoint_from_setuppy(content: str) -> Set[PathSpec]:
    entrypoints_modules = set()
    for m in _RE_ENTRYPOINT.finditer(content):
        ep = m.group('entrypoint')
        assert ep is not None
        # Parsing with yaml allows python dictionary formatting which does not
        # conform to the strict JSON syntax
        try:
            ep_dict = yaml.safe_load(ep)
        except (yaml.parser.ParserError, yaml.scanner.ScannerError):
            continue
        for members in ep_dict.values():
            if not isinstance(members, list):
                continue
            for member in members:
                if not isinstance(member, str):
                    continue
                member = member.strip().split('=')
                if len(member) != 2:
                    continue
                imported = member[1].strip().split(':')
                if len(imported) != 2:
                    continue
                entrypoints_modules.add(
                    tuple((mod.strip() for mod in imported[0].split('.'))))
    return entrypoints_modules


def _aggregate_modules(files: Dict[PathSpec, File]) -> Dict[PathSpec, PathSpec]:
    path_to_module = dict()
    for path, file in files.items():
        if file.excluded:
            continue
        base_name = match_python_file(file.name)
        if base_name is not None:
            path_to_module[path] = tuple(itertools.chain(path[:-1], [base_name]))
    return path_to_module


def _extract_setup(
        files: Dict[PathSpec, File],
        path_to_module: Dict[PathSpec, PathSpec],
        setuppy: Optional[File]) -> Dict[PathSpec, MentionFeature]:
    if setuppy is None and ('setup.py', ) in files:
        setuppy = files[('setup.py', )]
    if setuppy is None:
        return {path: MentionFeature.FILE_NOT_AVAILABLE for path, file in files.items()}
    entrypoints_modules = _extract_entrypoint_from_setuppy(setuppy.content)
    if len(entrypoints_modules) == 0:
        return {path: MentionFeature.FILE_NOT_AVAILABLE for path, file in files.items()}
    setup_features = dict()
    for path in files.keys():
        if path in path_to_module and path_to_module[path] in entrypoints_modules:
            setup_features[path] = MentionFeature.FILE_AVAILABLE_AND_IS_MENTIONED
        else:
            setup_features[path] = MentionFeature.FILE_AVAILABLE_BUT_NOT_MENTIONED
    if any((f == MentionFeature.FILE_AVAILABLE_AND_IS_MENTIONED
            for f in setup_features.values())):
        return setup_features
    else:
        return {path: MentionFeature.FILE_NOT_AVAILABLE for path, file in files.items()}


_RE_MARKDOWN_CODEBLOCK = re.compile(
    '``` *(?P<type>[a-zA-Z0-9_-]+)? *\n(?P<content>.*?)```',
    flags=re.DOTALL
)


_RE_PYTHON_MODULE_EXEC = re.compile(
    'python[23]?[ \n\t\r\\\\]+-m[ \n\t\r\\\\]+(?P<module>[a-zA-Z0-9_.]+)[ \n\t\r\\\\]',
    flags=re.DOTALL
)


def _extract_python_module_exec(content: str) -> Set[PathSpec]:
    modules = set()
    for m in _RE_PYTHON_MODULE_EXEC.finditer(content):
        assert m.group('module') is not None
        modules.add(tuple(m.group('module').split('.')))
    return modules


def _extract_imports_from_markdown(content: str) -> Set[PathSpec]:
    imports = set()
    for m in _RE_MARKDOWN_CODEBLOCK.finditer(content):
        if m.group('type') is None or m.group('type') == 'python':
            imports |= extract_import(m.group('content'))
    return imports


def _extract_path_mentions(content: str, files: Dict[PathSpec, File]) -> Set[PathSpec]:
    path_mentions = set()
    for path in files.keys():
        if '/'.join(path) in content:
            path_mentions.add(path)
    return path_mentions


def _extract_imported_module_from_readme(
        files: Dict[PathSpec, File],
        readme: Optional[File],
        path_to_module: Dict[PathSpec, PathSpec]) -> Dict[PathSpec, MentionFeature]:
    """ Identify file mentions in readme. This function targets three types of
    mentions:
    1. Import statements in code blocks
    2. Module execution (such as `python -m my.module ...`) in whole text
    3. Path mentions (such as 'path/to/important_file.py')
    """
    if readme is None:
        return {path: MentionFeature.FILE_NOT_AVAILABLE for path, file in files.items()}
    imports = _extract_imports_from_markdown(readme.content)
    imports |= _extract_python_module_exec(readme.content)
    path_mentions = _extract_path_mentions(readme.content, files)
    if len(imports) == 0 and len(path_mentions) == 0:
        return {path: MentionFeature.FILE_NOT_AVAILABLE for path, file in files.items()}
    placeholder = ('!placeholder', )
    if len(imports) > 0:
        # create empty mapping from module
        module_to_imports = {
            mod: set() for mod in path_to_module.values()
        }
        assert placeholder not in module_to_imports
        module_to_imports[placeholder] = imports
        _, valid_imports = identify_imports(module_to_imports)
        # key is deleted if nothing matches
        if len(valid_imports) == 0:
            valid_imports[placeholder] = set()
    else:
        valid_imports = {placeholder: set()}
    if len(valid_imports[placeholder]) == 0 and len(path_mentions) == 0:
        return {path: MentionFeature.FILE_NOT_AVAILABLE for path, file in files.items()}

    readme_import_features = dict()
    for path in files.keys():
        if path in path_to_module and path_to_module[path] in valid_imports[placeholder]:
            readme_import_features[path] = MentionFeature.FILE_AVAILABLE_AND_IS_MENTIONED
        elif path in path_mentions:
            readme_import_features[path] = MentionFeature.FILE_AVAILABLE_AND_IS_MENTIONED
        else:
            readme_import_features[path] = MentionFeature.FILE_AVAILABLE_BUT_NOT_MENTIONED
    if any((f == MentionFeature.FILE_AVAILABLE_AND_IS_MENTIONED
            for f in readme_import_features.values())):
        return readme_import_features
    else:
        return {path: MentionFeature.FILE_NOT_AVAILABLE for path, file in files.items()}


def extract_oracle_features(files: Dict[PathSpec, File], readme: Optional[File], setuppy: Optional[File]) -> Dict[PathSpec, OracleFeatures]:
    files = extract_python_files(files)
    path_to_module = _aggregate_modules(files)
    setup_features = _extract_setup(files, path_to_module, setuppy)
    readme_import_features = _extract_imported_module_from_readme(
        files, readme, path_to_module)
    features = dict()
    for path, f in files.items():
        features[path] = OracleFeatures(
            is_setup_entrypoint=setup_features[path],
            is_mentioned_in_readme=readme_import_features[path],
        )
    return features
