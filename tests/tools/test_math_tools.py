import math
from unittest.mock import Mock, MagicMock

import pytest
from langchain_openai import ChatOpenAI

from llmcompiler.tools.math_tools import _evaluate_expression, MathEvaluationError, get_math_tool


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


def test_invalid_problem_input():
    mock_llm = Mock(spec=ChatOpenAI)
    math_tool = get_math_tool(mock_llm)
    assert "Error: 'problem' must be a non-empty string." in math_tool.func("", None)
    assert "Error: 'problem' must be a non-empty string." in math_tool.func("   ", None)
    assert "Error: 'problem' must be a non-empty string." in math_tool.func(123, None)


def test_invalid_context_input():
    mock_llm = Mock(spec=ChatOpenAI)
    math_tool = get_math_tool(mock_llm)
    assert "Error: 'context' must be a list of strings or None." in math_tool.func("2+2", "not a list")
    assert "Error: All items in 'context' must be strings." in math_tool.func("2+2", [1, 2, 3])
    assert "Error: All items in 'context' must be strings." in math_tool.func("2+2", ["valid", 123])


def test_llm_integration():
    mock_llm = Mock(spec=ChatOpenAI)

    # Create a mock for the structured output
    mock_structured_output = MagicMock()
    mock_structured_output.code = "2 + 2"
    mock_structured_output.reasoning = "Simple addition"

    # Set up the chain of mocks
    mock_llm.with_structured_output.return_value.return_value = mock_structured_output

    math_tool = get_math_tool(mock_llm)

    assert math_tool.func("2 + 2") == "4"

    # Verify that the methods were called in the correct order
    mock_llm.with_structured_output.assert_called_once()
    mock_llm.with_structured_output.return_value.assert_called_once()


def test_llm_error_handling():
    mock_llm = Mock(spec=ChatOpenAI)

    # Set up the chain to raise an exception
    mock_llm.with_structured_output.return_value.side_effect = Exception("LLM error")

    math_tool = get_math_tool(mock_llm)
    assert "An unexpected error occurred: LLM error" in math_tool.func("2 + 2")

    # Verify that the method was called
    mock_llm.with_structured_output.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__])
