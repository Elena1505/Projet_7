from dashboard import id_to_index


def test_index_0():
    index = id_to_index("100002")
    assert index == 0


def test_index_1():
    index = id_to_index("100003")
    assert index == 1


def test_index_2():
    index = id_to_index("100004")
    assert index == 2


def test_index_3():
    index = id_to_index("100006")
    assert index == 3


def test_index_4():
    index = id_to_index("100007")
    assert index == 4




