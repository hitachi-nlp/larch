import itertools
import re
import signal
from collections import defaultdict
from typing import Any, Set
from typing import Optional, List, Dict, Tuple, Union

import networkx as nx

from larch.context_creator.model import File
from .common import PathSpec, match_python_file, FeatureMixin

_PAT_IMPORTS_PARENTHESIS = '(?:\\((?P<imports_parenthesis>[^\\)]+)\\))'
_PAT_IMPORTS = '(?P<imports>(?:[ ,._a-zA-Z0-9\\*]|(?:\\\\(?:(?:\r\n)|\r|\n)))+)'
_PAT_START = '(?:(?:\n *)|(?:^ *))'
_PAT_FROM = 'from *(?P<from>[ ._a-zA-Z0-9]+)'
RE_IMPORT = re.compile(
    f'{_PAT_START}(?:{_PAT_FROM} +)?c?import +(?:{_PAT_IMPORTS_PARENTHESIS}|{_PAT_IMPORTS})',
    flags=re.DOTALL
)
_RE_CR = re.compile('[\n\r\\\\]+')

ModuleSpec = Tuple[str, ...]


def extract_import(text: str) -> Set[ModuleSpec]:
    modules_all = set()
    for m in RE_IMPORT.finditer(text):
        if m.group('imports') is not None:
            modules = [
                mod.strip()
                for mod in _RE_CR.sub(' ', m.group('imports')).split(',')
                if len(mod.strip()) > 0]
        elif m.group('imports_parenthesis') is not None:
            modules = [
                mod.strip()
                for mod in _RE_CR.sub(' ', m.group('imports_parenthesis')).split(',')
                if len(mod.strip()) > 0]
        else:
            assert not 'Should not reach here'
        if m.group('from') is not None:
            if m.group('from') == '.':
                from_mod = ['']
            else:
                from_mod = m.group('from').split('.')
        else:
            from_mod = []
        for mod in modules:
            modules_all.add(tuple(itertools.chain(from_mod, mod.split('.'))))
    return modules_all


def _is_imported(importer: PathSpec, imp: PathSpec, importee: PathSpec) -> bool:
    assert isinstance(importer, tuple)
    assert isinstance(imp, tuple)
    assert isinstance(importee, tuple)
    if importee[-1] == '__init__':
        importee = importee[:-1]

    if len(importee) == 0:
        return imp == ('.', )

    if imp[0] == '':
        # relative import
        n_up = sum((i == '' for i in imp))
        if set(imp[:n_up]) != {''}:
            # it is malformed; it has relative import in the middle
            return False

        assert n_up > 0
        truncated_importer = importer[:-n_up]
        if truncated_importer == importee[:len(truncated_importer)]:
            # It shares root directory
            truncated_importee = importee[len(truncated_importer):]
            if imp[n_up:len(truncated_importee) + n_up] == truncated_importee:
                return True
        return False
    else:
        # try absolute import
        # we cut import off by len(importee) to avoid looking into the file
        # like "from my_module import MyClass"
        if imp[:len(importee)] == importee:
            return True
    # Try relative import even when "." is not there as Python 2.7 allowed
    # implicit relative import
    truncated_importer = importer[:-1]
    if truncated_importer == importee[:len(truncated_importer)]:
        # It shares root directory
        truncated_importee = importee[len(truncated_importer):]
        if imp[:len(truncated_importee)] == truncated_importee:
            return True
    return False


def _identify_best_import_cand(
        importer: PathSpec,
        imp: PathSpec,
        importee_candidates: List[PathSpec]) -> Optional[PathSpec]:
    importee_candidates = sorted(importee_candidates, key=lambda importee: (-len(importee), importee[-1] == '__init__'))
    for importee in importee_candidates:
        if _is_imported(importer, imp, importee):
            return importee
    return None


def identify_imports(path_to_imports: Dict[ModuleSpec, Set[ModuleSpec]]) -> Tuple[ModuleSpec, Dict[ModuleSpec, Set[ModuleSpec]]]:
    # FIXME: This operation costs O(len(path_to_imports)^3)
    if len(path_to_imports) == 0:
        return tuple(), dict()
    valid_imports_per_root = dict()
    roots = {
        path[:i]
        for path in path_to_imports.keys()
        for i in range(len(path))
    }
    for root in roots:
        valid_path_to_imports = list({
            path[len(root):]
            for path in path_to_imports.keys()
            if path[:len(root)] == root and len(path[len(root):]) > 0
        })
        # Heuristic to reduce computation
        if len(path_to_imports) > 500 and len(valid_path_to_imports) < len(path_to_imports) / 2:
            continue
        valid_imports = defaultdict(set)
        for path, imports in path_to_imports.items():
            for imp in imports:
                importee_rel = _identify_best_import_cand(
                    path[len(root):], imp, valid_path_to_imports)
                importee_abs = _identify_best_import_cand(
                    path, tuple(itertools.chain(root, imp)), valid_path_to_imports)
                if importee_abs is not None and (importee_rel is None or len(importee_abs) >= len(importee_rel)):
                    valid_imports[path].add(
                        tuple(itertools.chain(root, importee_abs)))
                elif importee_rel is not None and (importee_abs is None or len(importee_rel) > len(importee_abs)):
                    valid_imports[path].add(
                        tuple(itertools.chain(root, importee_rel)))
        valid_imports_per_root[root] = dict(valid_imports)
    # Find root with maximum number of imports
    root, valid_imports = max(
        valid_imports_per_root.items(),
        key=lambda root_imports: (sum(map(len, root_imports[1].values())), -len(root_imports[0]))
    )

    return root, valid_imports


class ImportFeatures(FeatureMixin):
    def __init__(
            self,
            rank: int,
            num_importers: int,
            num_importees: int,
            max_rank: int
    ):
        self._rank: int = rank
        self._num_importers: int = num_importers
        self._num_importees: int = num_importees
        self._max_rank: int = max_rank

    def get_pseudo_label_array(self) -> List[Optional[bool]]:
        return [
            True if self._rank == 0 else None,
            False if self._rank == self._max_rank else None
        ]

    @classmethod
    def get_pseudo_label_names(cls) -> List[str]:
        return [
            'top_rank',
            'bottom_rank'
        ]

    def get_feature_array(self) -> List[Union[float, bool, int, None]]:
        return [
            float(self._rank),
            float(self._num_importers),
            float(self._num_importees),
            bool(self._rank == self._max_rank)
        ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return [
            'rank',
            'num_importers',
            'num_importees',
            'bottom_rank'
        ]


def _remove_bidirectional_edges(imports: Dict[ModuleSpec, Set[ModuleSpec]]) -> Dict[ModuleSpec, Set[ModuleSpec]]:
    # We will keep imports from higher hierarchy
    # We keep both if they are of same hierarchy
    # We don't care if they share the same ancestors or not to aggresively
    # prune bidirectional edges
    new_imports = defaultdict(set)
    for importer, importees in imports.items():
        for importee in importees:
            # whether there are bidirectional edges
            if importee in imports and importer in imports[importee]:
                if len(importer) <= len(importee):
                    new_imports[importer].add(importee)
                else:
                    # one edge was pruned
                    pass
            else:
                new_imports[importer].add(importee)
    return dict(new_imports)


def _calc_node_ranks(graph: nx.DiGraph) -> Dict[Any, int]:
    # Rank is defined as the maximum import steps from top files, where top
    # files are files that aren't imported from other files
    # This has a little odd behavior when there is a loop in the directed graph
    # but it should still work
    if len(graph.nodes) == 0:
        # empty graph, return an empty dict
        return dict()
    distances = dict(nx.all_pairs_shortest_path_length(graph))
    ranks = {
        node: max((d.get(node, 0) for d in distances.values()))
        for node in graph.nodes
    }
    # make sure that thare is no jump between ranks
    i = 0
    while i <= max(ranks.values()):
        if i not in set(ranks.values()):
            ranks = {node: rank - 1 if rank > i else rank
                     for node, rank in ranks.items()}
        else:
            i += 1
    return ranks


def _analyze_import_graph(imports: Dict[ModuleSpec, Set[ModuleSpec]]) -> Dict[ModuleSpec, ImportFeatures]:
    import_graph = nx.DiGraph()
    nodes = list(itertools.chain(imports.keys(), *imports.values()))
    import_graph.add_nodes_from(nodes)
    edges = [
        (importer, importee)
        for importer, importees in imports.items()
        for importee in importees]
    import_graph.add_edges_from(edges)
    ranks = _calc_node_ranks(import_graph)

    import_features = dict()
    for node in nodes:
        rank = ranks[node]
        num_importers = sum(1 for _ in import_graph.predecessors(node))
        num_importees = sum(1 for _ in import_graph.successors(node))
        import_features[node] = ImportFeatures(
            rank=rank,
            num_importers=num_importers,
            num_importees=num_importees,
            max_rank=max(ranks.values())
        )
    return import_features


def _aggregate_imports(
        files: Dict[PathSpec, File]) -> Tuple[Dict[PathSpec, ModuleSpec], Dict[PathSpec, Set[ModuleSpec]]]:
    path_to_module = dict()
    module_to_imports = dict()
    for path, file in files.items():
        if file.excluded:
            continue
        base_name = match_python_file(file.name)
        if base_name is not None:
            module_path = tuple(itertools.chain(path[:-1], [base_name]))
            path_to_module[path] = module_path
            module_to_imports[module_path] = extract_import(file.content)
    return path_to_module, module_to_imports


def _extract_import_features(files: Dict[PathSpec, File]) -> Dict[PathSpec, ImportFeatures]:
    path_to_module, module_to_imports = _aggregate_imports(files)
    _, module_to_imported_module = identify_imports(module_to_imports)
    module_to_imported_module = _remove_bidirectional_edges(module_to_imported_module)
    module_to_features = _analyze_import_graph(module_to_imported_module)
    features = dict()
    for path in files.keys():
        if path in path_to_module and path_to_module[path] in module_to_features:
            features[path] = module_to_features[path_to_module[path]]
    return features


class TimeoutException(Exception):
    pass


def extract_import_features(
        files: Dict[PathSpec, File],
        repo_name: Optional[str] = None,
        timeout: Optional[int] = None) -> Dict[PathSpec, ImportFeatures]:
    # FIXME: we employ timeout as this is poorly implemented and costs O(len(files)^3)
    if timeout is None:
        return _extract_import_features(files)
    else:
        assert repo_name is not None

        def handler(signum, frame):
            raise TimeoutException()

        signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        try:
            features = _extract_import_features(files)
            signal.alarm(0)
        except TimeoutException:
            print('Timeout:', repo_name)
            return dict()
        return features
