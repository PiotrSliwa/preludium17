from typing import Sequence, Callable, Optional, TypeVar, List

T = TypeVar('T')
R = TypeVar('R')


def last_index(sequence: Sequence[T], match: Callable[[T], bool]) -> Optional[int]:
    for (i, e) in enumerate(reversed(sequence)):
        if match(e):
            return len(sequence) - i - 1
    return None


def indexes_of(sequence: Sequence[T], match: Callable[[T], bool]) -> List[int]:
    return [i for i, e in enumerate(sequence) if match(e)]