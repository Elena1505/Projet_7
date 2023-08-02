from dashboard import validator_id


def test_if_the_validator_return_1_for_a_valid_id():
    valid_id = validator_id("100002")
    assert valid_id == 1


def test_if_the_validator_return_0_for_a_non_valid_id():
    non_valid_id = validator_id("0")
    assert non_valid_id == 0
