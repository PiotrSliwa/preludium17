from lists import last_index, indexes_of


def test_last_index():
    iter = ['a', 'b', 'c', 'b', 'a']
    assert last_index(iter, lambda x: x == 'b') == 3


def test_indexes_of_two():
    iter = ['a', 'b', 'c', 'b', 'a']
    assert indexes_of(iter, lambda x: x == 'b') == [1, 3]


def test_indexes_of_none():
    iter = ['a', 'b', 'c', 'b', 'a']
    assert indexes_of(iter, lambda x: x == 'nonexistent') == []
