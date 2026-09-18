from guard.steps.masking import supplemental_mask


def test_spelled_out_phone_masked():
    text = "my phone number is nine five nine five nine five nine five nine five okay"
    masked, entities = supplemental_mask(text)
    assert entities == {"PHONE_NUMBER": 1}
    assert "nine five" not in masked
    assert "[REDACTED]" in masked


def test_disclosed_name_masked():
    text = "What is the protocol for study 101? my name is alen"
    masked, entities = supplemental_mask(text)
    assert entities == {"PERSON": 1}
    assert "alen" not in masked
    assert "my name is [REDACTED]" in masked


def test_call_me_stopword_not_masked():
    masked, entities = supplemental_mask("call me later please")
    assert entities == {}
    assert masked == "call me later please"


def test_short_number_word_runs_not_masked():
    masked, entities = supplemental_mask("i have two three four options")
    assert entities == {}
    assert masked == "i have two three four options"
