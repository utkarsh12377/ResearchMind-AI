from app.llm.json_output import clamp_confidence, load_json_object


def test_parses_a_bare_object() -> None:
    assert load_json_object('{"a": 1}') == {"a": 1}


def test_strips_a_json_code_fence() -> None:
    assert load_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_strips_an_unlabelled_code_fence() -> None:
    assert load_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_recovers_an_object_wrapped_in_prose() -> None:
    raw = 'Certainly. Here is the result:\n{"a": 1}\nLet me know if you need more.'

    assert load_json_object(raw) == {"a": 1}


def test_returns_none_when_there_is_no_object() -> None:
    assert load_json_object("I cannot answer that.") is None


def test_returns_none_for_empty_input() -> None:
    assert load_json_object("") is None
    assert load_json_object("   ") is None


def test_returns_none_for_a_top_level_array() -> None:
    """Every caller expects a mapping; a list would fail on the first .get()."""
    assert load_json_object("[1, 2, 3]") is None


def test_returns_none_for_malformed_json() -> None:
    assert load_json_object('{"a": ') is None


def test_clamp_confidence_bounds_the_range() -> None:
    assert clamp_confidence(2.5) == 1.0
    assert clamp_confidence(-1) == 0.0
    assert clamp_confidence(0.42) == 0.42


def test_clamp_confidence_falls_back_on_junk() -> None:
    assert clamp_confidence("high") == 0.5
    assert clamp_confidence(None, default=0.1) == 0.1
