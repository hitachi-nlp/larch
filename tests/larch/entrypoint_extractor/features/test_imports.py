import networkx as nx
import pytest
from larch.context_creator.model import Directory
from larch.entrypoint_extractor.features.common import aggregate_files
from larch.entrypoint_extractor.features.imports import extract_import, \
    identify_imports, _is_imported, _identify_best_import_cand, _calc_node_ranks, \
    _aggregate_imports


def test_extract_import():
    input_text = "import foo"
    output = extract_import(input_text)
    assert output == {('foo', )}

    input_text = """import foo
import foo_bar, baz"""
    output = extract_import(input_text)
    assert output == {('foo', ), ('foo_bar', ), ('baz', )}

    input_text = "cimport foo"
    output = extract_import(input_text)
    assert output == {('foo', )}

    input_text = "    import   foo"
    output = extract_import(input_text)
    assert output == {('foo', )}

    input_text = "_import = foo"
    output = extract_import(input_text)
    assert output == set()

    input_text = """import bar123, \\
    buzz,\\
    b"""
    output = extract_import(input_text)
    assert output == {('bar123', ), ('buzz', ), ('b', )}

    input_text = "import (_bar, baz)"
    output = extract_import(input_text)
    assert output == {('_bar', ), ('baz', )}

    input_text = "from . import abc"
    output = extract_import(input_text)
    assert output == {('', 'abc')}

    input_text = """from foo import (bar123,
    buzz,
    b
)"""
    output = extract_import(input_text)
    assert output == {('foo', 'bar123'), ('foo', 'buzz'), ('foo', 'b')}

    input_text = "from .my.library import bcd"
    output = extract_import(input_text)
    assert output == {('', 'my', 'library', 'bcd')}

    input_text = "import *"
    output = extract_import(input_text)
    assert output == {('*', )}

    input_text = "# import foo"
    output = extract_import(input_text)
    assert output == set()


@pytest.mark.dependency(
    depends=["tests/larch/entrypoint_extractor/features/test_common.py::test_aggregate_files"],
    scope='session')
def test__aggregate_imports():
    dir_tree = Directory.parse_obj({
        "name": "repo",
        "children": [
            {
                'name': 'main.py',
                'excluded': False,
                'content': 'from foo import baz'
            },
            {
                'name': 'foo',
                'children': [
                    {
                        'name': '__init__.py',
                        'excluded': False,
                        'content': 'from . import baz'
                    },
                    {
                        'name': 'baz.py',
                        'excluded': False,
                        'content': ''
                    },
                    {
                        'name': 'bar.py',
                        'excluded': False,
                        'content': 'import foo.baz'
                    }
                ]
            }
        ]
    })
    files = aggregate_files(dir_tree)
    path_to_module, module_to_imports = _aggregate_imports(files)
    expected = {
        ('main.py', ): ('main', ),
        ('foo', '__init__.py'): ('foo', '__init__'),
        ('foo', 'baz.py'): ('foo', 'baz'),
        ('foo', 'bar.py'): ('foo', 'bar')
    }
    assert path_to_module == expected
    expected = {
        ('main', ): {('foo', 'baz')},
        ('foo', '__init__'): {('', 'baz')},
        ('foo', 'baz'): set(),
        ('foo', 'bar'): {('foo', 'baz')}
    }
    assert module_to_imports == expected

    # An example with multiple imports and non-existent imports
    content = """import re
import foo.baz
import foo.baz"""
    dir_tree = Directory.parse_obj({
        "name": "repo",
        "children": [
            {
                'name': 'main.py',
                'excluded': False,
                'content': content
            },
            {
                'name': 'foo',
                'children': [
                    {
                        'name': 'baz.py',
                        'excluded': False,
                        'content': ''
                    }
                ]
            }
        ]
    })
    files = aggregate_files(dir_tree)
    path_to_module, module_to_imports = _aggregate_imports(files)
    expected = {
        ('main.py', ): ('main', ),
        ('foo', 'baz.py'): ('foo', 'baz'),
    }
    assert path_to_module == expected
    expected = {
        ('main', ): {('re', ), ('foo', 'baz')},
        ('foo', 'baz'): set()
    }
    assert module_to_imports == expected


def test__is_imported():
    # Simple absolute import
    assert _is_imported(('foo', 'bar'), ('foo', 'baz'), ('foo', 'baz')) is True

    # Simple absolute import of parent directory
    assert _is_imported(('foo', 'bar'), ('foo', 'baz'), ('foo', '__init__')) is True

    # Failed absolute import
    assert _is_imported(('main', ), ('foo', 'baz'), ('foo', 'abc')) is False

    # Absolute import with class
    assert _is_imported(('main', ), ('foo', 'baz', 'MyClass'), ('foo', 'baz')) is True

    # Relative import
    assert _is_imported(('foo', 'bar'), ('', 'baz', 'MyClass'), ('foo', 'baz')) is True

    # Failed relative import
    assert _is_imported(('foo', 'bar'), ('', 'abc'), ('foo', 'baz')) is False

    # Relative import from parent directory
    assert _is_imported(('foo', 'bar', 'baz'), ('', '', 'baz'), ('foo', 'baz')) is True

    # Implicit relative import
    assert _is_imported(('foo', 'bar'), ('baz', ), ('foo', 'baz')) is True

    # Edge case, having only __init__ in importee requires exception handling
    assert _is_imported(('foo', ), ('bar', 'baz'), ('__init__', )) is False
    assert _is_imported(('foo', ), ('.', 'baz'), ('__init__', )) is False
    assert _is_imported(('foo', ), ('.', ), ('__init__', )) is True


def test__identify_best_import_cand():
    # Simple example
    result = _identify_best_import_cand(
        ('foo', 'bar'),
        ('foo', 'baz'),
        [('foo', 'baz', 'bar'), ('foo', 'baz')])
    assert result == ('foo', 'baz')

    # Example with a false positive
    result = _identify_best_import_cand(
        ('foo', 'bar'),
        ('foo', 'baz', 'bar'),
        [('foo', 'baz', 'bar'), ('foo', 'baz')])
    assert result == ('foo', 'baz', 'bar')

    # Example with __init__
    result = _identify_best_import_cand(
        ('foo', 'bar'),
        ('foo', 'baz'),
        [('foo', '__init__'), ('foo', 'baz')])
    assert result == ('foo', 'baz')

    # Example with no match
    result = _identify_best_import_cand(
        ('main', ),
        ('foo', 'bar'),
        [('foo', 'abc'), ('foo', 'baz')])
    assert result is None

    # Example with class name and relative import
    result = _identify_best_import_cand(
        ('foo', 'bar'),
        ('', 'baz', 'MyClass'),
        [('foo', '__init__'), ('foo', 'baz')])
    assert result == ('foo', 'baz')


def test_identify_imports():
    path_to_import = {
        ('main',): {('foo', 'baz')},
        ('foo', '__init__'): {('', 'baz')},
        ('foo', 'baz'): set(),
        ('foo', 'bar'): {('foo', 'baz')}
    }
    root, mappings = identify_imports(path_to_import)
    assert root == tuple()
    expected = {
        ('main',): {('foo', 'baz')},
        ('foo', '__init__'): {('foo', 'baz')},
        ('foo', 'bar'): {('foo', 'baz')}
    }
    assert mappings == expected

    # test root identification
    path_to_import = {
        ('main',): {('foo', )},
        ('src', 'foo', '__init__'): {('', 'baz')},
        ('src', 'foo', 'baz'): set(),
        ('src', 'foo', 'bar'): {('foo', 'baz')}
    }
    root, mappings = identify_imports(path_to_import)
    assert root == ('src', )
    expected = {
        ('main',): {('src', 'foo', '__init__')},
        ('src', 'foo', '__init__'): {('src', 'foo', 'baz')},
        ('src', 'foo', 'bar'): {('src', 'foo', 'baz')}
    }
    assert mappings == expected


def test__calc_node_ranks():
    # A simple graph without a loop
    graph = nx.DiGraph()
    graph.add_nodes_from('abcdefgh')
    graph.add_edges_from(['ab', 'ac', 'ad', 'bg', 'cg', 'cd', 'df', 'ef', 'gh'])
    ranks = _calc_node_ranks(graph)
    expected = {
        'a': 0,
        'b': 1,
        'c': 1,
        'd': 1,
        'e': 0,
        'f': 2,
        'g': 2,
        'h': 3
    }
    assert ranks == expected

    # A graph with a loop and an isolated node
    graph = nx.DiGraph()
    graph.add_nodes_from('abcdefg')
    graph.add_edges_from(['ab', 'bc', 'cd', 'db', 'be', 'df'])
    ranks = _calc_node_ranks(graph)
    expected = {
        'a': 0,
        'b': 1,
        'c': 1,
        'd': 2,
        'e': 2,
        'f': 3,
        'g': 0
    }
    assert ranks == expected
