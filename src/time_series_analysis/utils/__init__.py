from collections.abc import Iterable


def ensure_iterable(arg: int | Iterable[int]) -> tuple[int, ...]:
    if isinstance(arg, int):
        return (arg,)
    if all(isinstance(i, int) for i in arg):
        return tuple(arg)

    raise TypeError("All values must be integers")
