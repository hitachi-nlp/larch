from larch.entrypoint_extractor.features.inheritance import _extract_class


def test__extract_class():
    input_text = "class Foo:"
    output = _extract_class(input_text)
    assert output == [('Foo', set())]

    input_text = "    class Foo(Bar):"
    output = _extract_class(input_text)
    assert output == [('Foo', {'Bar'})]

    input_text = """class Foo():
    pass

class Bar(
    Baz,
    Foo
):"""
    output = _extract_class(input_text)
    assert sorted(output) == sorted([('Foo', set()), ('Bar', {'Baz', 'Foo'})])
