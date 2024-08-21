import math

import pytest

from llmcompiler.tools.math_tools import _evaluate_expression, MathEvaluationError


def test_basic_arithmetic():
    assert _evaluate_expression("2 + 2") == 4
    assert _evaluate_expression("10 - 5") == 5
    assert _evaluate_expression("3 * 4") == 12
    assert _evaluate_expression("15 / 3") == 5


def test_complex_expressions():
    assert _evaluate_expression("2 ** 3 + 4 * 5") == 28
    assert abs(_evaluate_expression("sin(pi/2)") - 1) < 1e-10
    assert _evaluate_expression("log(e)") == 1


def test_constants():
    assert abs(_evaluate_expression("pi") - math.pi) < 1e-10
    assert abs(_evaluate_expression("e") - math.e) < 1e-10


def test_large_numbers():
    assert _evaluate_expression("1e100 * 1e100") == 1e200


def test_small_numbers():
    assert abs(_evaluate_expression("1e-100 / 1e100") - 1e-200) < 1e-210


def test_array_result():
    with pytest.raises(MathEvaluationError, match="NumExpr evaluation error"):
        _evaluate_expression("[1, 2, 3]")


def test_nan_result():
    with pytest.raises(MathEvaluationError, match="division by zero"):
        _evaluate_expression("0 / 0")


def test_infinity_result():
    with pytest.raises(MathEvaluationError, match="division by zero"):
        _evaluate_expression("1 / 0")


def test_undefined_variable():
    with pytest.raises(MathEvaluationError, match="Undefined variable or function"):
        _evaluate_expression("x + 5")


def test_syntax_error():
    with pytest.raises(MathEvaluationError, match="Syntax error"):
        _evaluate_expression("2 +* 3")


def test_type_error():
    with pytest.raises(MathEvaluationError, match="Type error"):
        _evaluate_expression("'2' + 2")


def test_invalid_expression():
    with pytest.raises(MathEvaluationError):
        _evaluate_expression("invalid")


if __name__ == '__main__':
    pytest.main([__file__])
