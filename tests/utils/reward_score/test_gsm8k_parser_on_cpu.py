from verl.utils.reward_score import gsm8k


def test_flexible_parser_strips_sentence_period_from_integer_answer():
    assert gsm8k.extract_solution("The answer is 308.", method="flexible") == "308"
    assert gsm8k.compute_score("The answer is 308.", "308", method="flexible") == 1.0


def test_flexible_parser_preserves_decimal_answers():
    assert gsm8k.extract_solution("The answer is 3.14", method="flexible") == "3.14"
    assert gsm8k.compute_score("The answer is 3.14", "3.14", method="flexible") == 1.0


def test_strict_parser_normalizes_trailing_period():
    assert gsm8k.extract_solution("#### 308.", method="strict") == "308"
    assert gsm8k.compute_score("#### 308.", "308", method="strict") == 1.0
