import pytest

from larch.context_creator.model import Directory
from larch.entrypoint_extractor.features.common import match_python_file, aggregate_files


@pytest.mark.dependency()
def test_aggregate_files():
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
    extracted_files = sorted((
        ('main.py', ),
        ('foo', '__init__.py'),
        ('foo', 'baz.py'),
        ('foo', 'bar.py')
    ))
    assert sorted(files.keys()) == extracted_files


def test_match_python_file():
    output = match_python_file('foo.py')
    assert output is not None and output == 'foo'

    output = match_python_file('foo_bar.pyc')
    assert output is not None and output == 'foo_bar'

    output = match_python_file('foo.py.txt')
    assert output is None
