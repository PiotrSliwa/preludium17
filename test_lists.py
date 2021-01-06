from lists import last_index


def test_last_index():
    iter = ['a', 'b', 'c', 'b', 'a']
    assert last_index(iter, lambda x: x == 'b') == 3