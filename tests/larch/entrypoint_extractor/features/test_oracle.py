from larch.entrypoint_extractor.features.oracle import _extract_entrypoint_from_setuppy, _extract_python_module_exec


def test__extract_entrypoint_from_setuppy():
    input = """
    setup(
        entry_points={
            'console_scripts':   [
                'larch=larch.cli:cli',
                'larch-server = larch.server:cli',
            ],
        },
        install_requires=requirements,
    )
    """
    output = _extract_entrypoint_from_setuppy(input)
    expected = [('larch', 'cli'), ('larch', 'server')]
    assert sorted(output) == sorted(expected)


def test__extract_python_module_exec():
    output = _extract_python_module_exec('python -m  my.module -m options')
    assert output == {('my', 'module')}
