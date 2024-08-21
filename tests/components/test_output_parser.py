import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from llmcompiler.components.output_parser import LLMCompilerPlanParser, _parse_llm_compiler_action_args


# Mock tools for testing
class MockTool1(BaseTool):
    def _run(self, arg1: str, arg2: int):
        # Dummy implementation
        return f"Running {self.name} with arg1: {arg1}, arg2: {arg2}"


class MockTool2(BaseTool):
    def _run(self, arg1: str):
        # Dummy implementation
        return f"Running {self.name} with arg1: {arg1}"


mock_tools = [
    MockTool1(name="tool1", description="Tool 1", tool_args={"arg1": str, "arg2": int}),
    MockTool2(name="tool2", description="Tool 2", tool_args={"arg1": str}),
]


@pytest.fixture
def parser():
    return LLMCompilerPlanParser(tools=mock_tools)


def test_string_input(parser):
    input_str = "Thought: Let's use tool1\n1. tool1(arg1='test', arg2=42)\n2. join()"
    tasks = list(parser._transform([input_str]))
    assert len(tasks) == 2
    assert tasks[0]['tool'].name == 'tool1'
    assert tasks[0]['args'] == {'arg1': 'test', 'arg2': 42}
    assert tasks[0]['thought'] == "Let's use tool1"


def test_human_message_input(parser):
    input_msg = HumanMessage(content="Thought: Use tool2\n1. tool2(arg1='hello')\n2. join()")
    tasks = list(parser._transform([input_msg]))
    assert len(tasks) == 2
    assert tasks[0]['tool'].name == 'tool2'
    assert tasks[0]['args'] == {'arg1': 'hello'}
    assert tasks[0]['thought'] == "Use tool2"


def test_ai_message_input(parser):
    input_msg = AIMessage(content="Thought: Combine results\n1. join()")
    tasks = list(parser._transform([input_msg]))
    assert len(tasks) == 1
    assert tasks[0]['tool'] == 'join'
    assert tasks[0]['args'] == ()
    assert tasks[0]['thought'] == "Combine results"


def test_mixed_input(parser):
    inputs = [
        "Thought: Start with tool1\n",
        HumanMessage(content="1. tool1(arg1='start', arg2=10)\n"),
        AIMessage(content="Thought: Now use tool2\n1. tool2(arg1='middle')\n"),
        "Thought: Finally, join\n1. join()"
    ]
    tasks = list(parser._transform(inputs))
    assert len(tasks) == 3
    assert [task['tool'].name if isinstance(task['tool'], BaseTool) else task['tool'] for task in tasks] == ['tool1',
                                                                                                             'tool2',
                                                                                                             'join']


def test_invalid_input(parser):
    with pytest.raises(ValueError):
        list(parser._transform([42]))  # Invalid input type


def test_no_match(parser):
    input_str = "This is just some random text without any task."
    tasks = list(parser._transform([input_str]))
    assert len(tasks) == 0


class MockTool3(BaseTool):
    def _run(self, arg1: dict, arg2: list):
        # Dummy implementation
        return f"Running {self.name} with arg1: {arg1}, arg2: {arg2}"


class MockTool4(BaseTool):
    def _run(self, arg1: list):
        # Dummy implementation
        return f"Running {self.name} with arg1: {arg1}"


class MockTool5(BaseTool):
    def _run(self, arg1: str, arg2: dict):
        # Dummy implementation
        return f"Running {self.name} with arg1: {arg1}, arg2: {arg2}"


def test_parse_complex_args():
    tool = MockTool3(name="complex_tool", description="Complex Tool")
    args = "arg1={'key1': 'value1', 'key2': [1, 2, 3]}, arg2=[{'nested': 'dict'}, (1, 2, 3)]"
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": {'key1': 'value1', 'key2': [1, 2, 3]},
        "arg2": [{'nested': 'dict'}, (1, 2, 3)]
    }


def test_parse_nested_structures():
    tool = MockTool4(name="nested_tool", description="Nested Tool", tool_args={"arg1": list})
    args = "arg1=[1, [2, 3], {'key': 'value'}]"
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": [1, [2, 3], {'key': 'value'}]
    }


def test_parse_mixed_args():
    tool = MockTool5(name="mixed_tool", description="Mixed Tool", tool_args={"arg1": str, "arg2": dict})
    args = "arg1='simple string', arg2={'nested': [1, 2, 3]}"
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": 'simple string',
        "arg2": {'nested': [1, 2, 3]}
    }


def test_parser_initialization_valid():
    parser = LLMCompilerPlanParser(tools=mock_tools)
    assert parser.tools == mock_tools


def test_parser_initialization_invalid_type():
    with pytest.raises(ValueError, match="The 'tools' parameter must be a list."):
        LLMCompilerPlanParser(tools="not a list")


def test_parser_initialization_empty_list():
    with pytest.raises(ValueError, match="The 'tools' list cannot be empty."):
        LLMCompilerPlanParser(tools=[])


def test_parser_initialization_invalid_items():
    invalid_tools = [mock_tools[0], "not a tool", mock_tools[1]]
    with pytest.raises(ValueError, match="All items in the 'tools' list must be instances of BaseTool."):
        LLMCompilerPlanParser(tools=invalid_tools)
