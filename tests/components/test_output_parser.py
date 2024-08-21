import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from llmcompiler.components.output_parser import LLMCompilerPlanParser


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
