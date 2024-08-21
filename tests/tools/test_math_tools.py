import math

import pytest

from llmcompiler.tools.math_tools import _evaluate_expression, MathEvaluationError


def test_evaluate_expression():
    assert _evaluate_expression("2 + 2") == 4
    assert abs(_evaluate_expression("pi") - math.pi) < 1e-10
    assert _evaluate_expression("2 ** 3") == 8
    try:
        _evaluate_expression("invalid")
    except MathEvaluationError:
        pass
    else:
        raise AssertionError("MathEvaluationError not raised")


if __name__ == '__main__':
    pytest.main([__file__])
