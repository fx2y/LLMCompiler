import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import BaseTool

from llmcompiler.components.output_parser import LLMCompilerPlanParser, _parse_llm_compiler_action_args, \
    InvalidToolError, ArgumentParsingError, instantiate_task


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
    assert tasks[0]['args'] == {}
    assert tasks[0]['thought'] == None


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


def test_invalid_tool_error():
    with pytest.raises(InvalidToolError, match="Tool invalid_tool not found."):
        instantiate_task(mock_tools, 1, "invalid_tool", "arg1='test'")


def test_argument_parsing_error():
    with pytest.raises(ArgumentParsingError, match="Error parsing arguments"):
        _parse_llm_compiler_action_args("arg1='test', arg2='not_an_int'", mock_tools[0])


def test_parse_args_with_tool_args():
    class ToolWithArgs(BaseTool):
        name = "tool_with_args"
        description = "A tool with specific args"
        args = {"arg1": str, "arg2": int, "arg3": list}

        def _run(self, arg1: str, arg2: int, arg3: list):
            return f"Running with {arg1}, {arg2}, {arg3}"

    tool = ToolWithArgs()
    args = "arg1='test', arg2=42, arg3=[1, 2, 3]"
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": "test",
        "arg2": 42,
        "arg3": [1, 2, 3]
    }


def test_parse_args_ignore_extra():
    class ToolWithLimitedArgs(BaseTool):
        name = "limited_args_tool"
        description = "A tool with limited args"
        args = {"arg1": str, "arg2": int}

        def _run(self, arg1: str, arg2: int):
            return f"Running with {arg1}, {arg2}"

    tool = ToolWithLimitedArgs()
    args = "arg1='test', arg2=42, extra_arg='ignored'"
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": "test",
        "arg2": 42
    }
    assert "extra_arg" not in parsed_args


def test_parse_args_missing_arg():
    class ToolWithRequiredArgs(BaseTool):
        name = "required_args_tool"
        description = "A tool with required args"
        args = {"arg1": str, "arg2": int}

        def _run(self, arg1: str, arg2: int):
            return f"Running with {arg1}, {arg2}"

    tool = ToolWithRequiredArgs()
    args = "arg1='test'"
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": "test"
    }
    assert "arg2" not in parsed_args


def test_parse_args_out_of_order():
    class ToolWithOrderedArgs(BaseTool):
        name = "ordered_args_tool"
        description = "A tool with ordered args"
        args = {"arg1": str, "arg2": int, "arg3": list}

        def _run(self, arg1: str, arg2: int, arg3: list):
            return f"Running with {arg1}, {arg2}, {arg3}"

    tool = ToolWithOrderedArgs()
    args = "arg2=42, arg1='test', arg3=[1, 2, 3]"
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": "test",
        "arg2": 42,
        "arg3": [1, 2, 3]
    }


def test_parse_complex_nested_args():
    class ComplexTool(BaseTool):
        name = "complex_tool"
        description = "A tool with complex nested args"
        args = {"arg1": dict, "arg2": list, "arg3": str}

        def _run(self, arg1: dict, arg2: list, arg3: str):
            return f"Running with {arg1}, {arg2}, {arg3}"

    tool = ComplexTool()
    args = "arg1={'key1': {'nested': [1, 2, 3]}, 'key2': 'value'}, arg2=[1, (2, 3), {'a': 'b'}], arg3='test, with, commas'"
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": {'key1': {'nested': [1, 2, 3]}, 'key2': 'value'},
        "arg2": [1, (2, 3), {'a': 'b'}],
        "arg3": 'test, with, commas'
    }


def test_parse_args_with_strings_containing_special_chars():
    class StringTool(BaseTool):
        name = "string_tool"
        description = "A tool with string args that may contain special characters"
        args = {"arg1": str, "arg2": str}

        def _run(self, arg1: str, arg2: str):
            return f"Running with {arg1}, {arg2}"

    tool = StringTool()
    args = "arg1='test, with, commas', arg2=\"test with \\\"quotes\\\" and {braces}\""
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": 'test, with, commas',
        "arg2": 'test with "quotes" and {braces}'
    }


def test_parse_args_with_mixed_types():
    class MixedTool(BaseTool):
        name = "mixed_tool"
        description = "A tool with mixed arg types"
        args = {"arg1": int, "arg2": list, "arg3": dict, "arg4": str}

        def _run(self, arg1: int, arg2: list, arg3: dict, arg4: str):
            return f"Running with {arg1}, {arg2}, {arg3}, {arg4}"

    tool = MixedTool()
    args = "arg1=42, arg2=[1, 'two', 3.0], arg3={'key': 'value', 'nested': [1, 2]}, arg4='simple string'"
    parsed_args = _parse_llm_compiler_action_args(args, tool)

    assert parsed_args == {
        "arg1": 42,
        "arg2": [1, 'two', 3.0],
        "arg3": {'key': 'value', 'nested': [1, 2]},
        "arg4": 'simple string'
    }


def test_argument_type_validation():
    class TypedTool(BaseTool):
        name = "typed_tool"
        description = "A tool with typed args"

        def _run(self, arg1: str, arg2: int, arg3: float, arg4: bool, arg5: list, arg6: dict):
            return f"Running with {arg1}, {arg2}, {arg3}, {arg4}, {arg5}, {arg6}"

    tool = TypedTool()

    # Test valid types
    args = "arg1='test', arg2=42, arg3=3.14, arg4=True, arg5=[1, 2, 3], arg6={'key': 'value'}"
    parsed_args = _parse_llm_compiler_action_args(args, tool)
    assert parsed_args == {
        "arg1": "test",
        "arg2": 42,
        "arg3": 3.14,
        "arg4": True,
        "arg5": [1, 2, 3],
        "arg6": {"key": "value"}
    }

    # Test invalid types
    with pytest.raises(ArgumentParsingError, match="Expected type int for value: 'not_an_int'"):
        _parse_llm_compiler_action_args("arg1='test', arg2='not_an_int'", tool)

    with pytest.raises(ArgumentParsingError, match="Expected type float for value: 'not_a_float'"):
        _parse_llm_compiler_action_args("arg3='not_a_float'", tool)

    # with pytest.raises(ArgumentParsingError, match="Expected type boolean for value: 'not_a_bool'"):
    #     _parse_llm_compiler_action_args("arg4='not_a_bool'", tool)


def test_type_conversion():
    class ConversionTool(BaseTool):
        name = "conversion_tool"
        description = "A tool to test type conversion"

        def _run(self, arg1: int, arg2: float, arg3: bool):
            return f"Running with {arg1}, {arg2}, {arg3}"

    tool = ConversionTool()

    args = "arg1='42', arg2='3.14', arg3='True'"
    parsed_args = _parse_llm_compiler_action_args(args, tool)
    assert parsed_args == {
        "arg1": 42,
        "arg2": 3.14,
        "arg3": True
    }

    # Test invalid conversion
    with pytest.raises(ArgumentParsingError, match="Expected type int for value: 'not_convertible'"):
        _parse_llm_compiler_action_args("arg1='not_convertible'", tool)


def test_parse_complex_list_and_dict():
    class ComplexTool(BaseTool):
        name = "complex_tool"
        description = "A tool with complex list and dict args"

        def _run(self, arg1: list, arg2: dict):
            return f"Running with {arg1}, {arg2}"

    tool = ComplexTool()

    args = "arg1=[1, 'two', [3, 4], {'nested': 'dict'}], arg2={'key1': [1, 2, 3], 'key2': {'nested': 'value'}}"
    parsed_args = _parse_llm_compiler_action_args(args, tool)
    assert parsed_args == {
        "arg1": [1, 'two', [3, 4], {'nested': 'dict'}],
        "arg2": {'key1': [1, 2, 3], 'key2': {'nested': 'value'}}
    }


def test_parse_string_representations():
    class StringRepTool(BaseTool):
        name = "string_rep_tool"
        description = "A tool with string representations of list and dict"

        def _run(self, arg1: list, arg2: dict):
            return f"Running with {arg1}, {arg2}"

    tool = StringRepTool()

    args = "arg1='[1, \"two\", [3, 4]]', arg2='{\"key\": \"value\", \"nested\": [1, 2, 3]}'"
    parsed_args = _parse_llm_compiler_action_args(args, tool)
    assert parsed_args == {
        "arg1": [1, "two", [3, 4]],
        "arg2": {"key": "value", "nested": [1, 2, 3]}
    }


def test_invalid_list_and_dict():
    class InvalidTool(BaseTool):
        name = "invalid_tool"
        description = "A tool to test invalid list and dict inputs"

        def _run(self, arg1: list, arg2: dict):
            return f"Running with {arg1}, {arg2}"

    tool = InvalidTool()

    with pytest.raises(ArgumentParsingError, match="Expected type list for value: 'not_a_list'"):
        _parse_llm_compiler_action_args("arg1='not_a_list'", tool)

    with pytest.raises(ArgumentParsingError, match="Expected type dict for value: 'not_a_dict'"):
        _parse_llm_compiler_action_args("arg2='not_a_dict'", tool)
