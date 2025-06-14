import pytest

from time_series_analysis.utils import ensure_iterable


@pytest.mark.parametrize(
    "input",
    [1, [1, 2], (1,), range(1), iter([1, 2])],
    ids=["int", "list of int", "tuple of int", "range", "iterator"],
)
def test_ensure_diffs_as_iterable(input):
    assert isinstance(ensure_iterable(input), tuple)


@pytest.mark.parametrize(
    "input",
    [1.0, [1.0, 2.0], (1.0,), ["s", 1]],
    ids=["float", "list of floats", "tuple of int", "list of objects"],
)
def test_ensure_diffs_as_iterable_raises(input):
    with pytest.raises(TypeError):
        ensure_iterable(input)
