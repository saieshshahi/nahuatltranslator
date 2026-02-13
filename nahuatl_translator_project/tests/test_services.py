from webapp.services import build_prompt, guess_repetition


def test_build_prompt_contains_labels():
    p = build_prompt("en", "nah", "Hello", "Unknown")
    assert "translate English to Nahuatl" in p
    assert "Hello" in p


def test_repetition_heuristic_flags_niman():
    txt = "niman " * 30
    assert guess_repetition(txt) is True
