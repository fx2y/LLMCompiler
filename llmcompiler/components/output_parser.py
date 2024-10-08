import ast
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from typing_extensions import TypedDict


# Add these new custom exceptions
class InvalidToolError(OutputParserException):
    """Raised when an invalid tool is encountered."""


class ArgumentParsingError(OutputParserException):
    """Raised when there's an error parsing arguments."""


THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
ID_PATTERN = r"\$\{?(\d+)\}?"
END_OF_PLAN = ""


def _parse_llm_compiler_action_args(args: str, tool: Union[str, BaseTool]) -> dict[str, Any]:
    """Parse arguments from a string, handling complex structures and validating types."""
    if args == "" or isinstance(tool, str):
        return {}

    def parse_value(value: str, expected_type: type) -> Any:
        value = value.strip()
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            parsed_value = value

        if not isinstance(parsed_value, expected_type):
            try:
                # Attempt type conversion for basic types
                if expected_type == int:
                    parsed_value = int(parsed_value)
                elif expected_type == float:
                    parsed_value = float(parsed_value)
                elif expected_type == bool:
                    parsed_value = bool(parsed_value)
                elif expected_type == str:
                    parsed_value = str(parsed_value)
                elif expected_type == list:
                    if isinstance(parsed_value, str):
                        parsed_value = ast.literal_eval(parsed_value)
                    if not isinstance(parsed_value, list):
                        raise ValueError
                elif expected_type == dict:
                    if isinstance(parsed_value, str):
                        parsed_value = ast.literal_eval(parsed_value)
                    if not isinstance(parsed_value, dict):
                        raise ValueError
                else:
                    raise ValueError
            except ValueError:
                raise ArgumentParsingError(f"Expected type {expected_type.__name__} for value: {value}")

        return parsed_value

    extracted_args = {}
    remaining_args = args

    try:
        for key, schema in tool.args.items():
            if f"{key}=" in remaining_args:
                start_idx = remaining_args.index(f"{key}=") + len(key) + 1
                end_idx = start_idx
                nesting_level = 0
                in_string = False
                string_char = None

                for i, char in enumerate(remaining_args[start_idx:], start=start_idx):
                    if char in "\"'" and not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char and in_string:
                        in_string = False
                    elif char in "([{" and not in_string:
                        nesting_level += 1
                    elif char in ")]}" and not in_string:
                        nesting_level -= 1
                    elif char == "," and nesting_level == 0 and not in_string:
                        end_idx = i
                        break
                else:
                    end_idx = len(remaining_args)

                value = remaining_args[start_idx:end_idx].strip()
                expected_type = _get_expected_type(schema)
                extracted_args[key] = parse_value(value, expected_type)
                remaining_args = remaining_args[end_idx:].lstrip(", ")

    except ArgumentParsingError as e:
        raise e
    except Exception as e:
        raise OutputParserException(f"Error parsing arguments: {str(e)}") from e

    return extracted_args


def _get_expected_type(schema: Dict[str, Any]) -> type:
    type_str = schema.get('type', 'string')
    if type_str == 'string':
        return str
    elif type_str == 'integer':
        return int
    elif type_str == 'number':
        return float
    elif type_str == 'boolean':
        return bool
    elif type_str == 'array':
        return list
    elif type_str == 'object':
        return dict
    else:
        return str  # Default to string for unknown types


def default_dependency_rule(idx, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


def _get_dependencies_from_graph(
        idx: int, tool_name: str, args: Dict[str, Any]
) -> dict[str, list[str]]:
    """Get dependencies from a graph."""
    if tool_name == "join":
        return list(range(1, idx))
    return [i for i in range(1, idx) if default_dependency_rule(i, str(args))]


class Task(TypedDict):
    idx: int
    tool: BaseTool
    args: list
    dependencies: Dict[str, list]
    thought: Optional[str]


def instantiate_task(
        tools: Sequence[BaseTool],
        idx: int,
        tool_name: str,
        args: Union[str, Any],
        thought: Optional[str] = None,
) -> Task:
    if tool_name == "join":
        tool = "join"
    else:
        try:
            tool = tools[[tool.name for tool in tools].index(tool_name)]
        except ValueError as e:
            raise InvalidToolError(f"Tool {tool_name} not found.") from e
    tool_args = _parse_llm_compiler_action_args(args, tool)
    dependencies = _get_dependencies_from_graph(idx, tool_name, tool_args)
    return Task(
        idx=idx,
        tool=tool,
        args=tool_args,
        dependencies=dependencies,
        thought=thought,
    )


class LLMCompilerPlanParser(BaseTransformOutputParser[dict], extra="allow"):
    """Planning output parser."""

    tools: List[BaseTool]

    def __init__(self, tools: List[BaseTool]):
        self._validate_tools(tools)
        super().__init__(tools=tools)
        self.tools = tools

    @staticmethod
    def _validate_tools(tools: List[BaseTool]) -> None:
        if not isinstance(tools, list):
            raise ValueError("The 'tools' parameter must be a list.")
        if not tools:
            raise ValueError("The 'tools' list cannot be empty.")
        if not all(isinstance(tool, BaseTool) for tool in tools):
            raise ValueError("All items in the 'tools' list must be instances of BaseTool.")

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Task]:
        buffer = []
        current_thought = None

        try:
            for chunk in input:
                text = self._extract_text(chunk)
                for task, new_thought in self.ingest_token(text, buffer, current_thought):
                    if task:
                        yield task
                    current_thought = new_thought

            # Process any remaining content in the buffer
            if buffer:
                task, _ = self._parse_task("".join(buffer), current_thought)
                if task:
                    yield task
        except Exception as e:
            raise OutputParserException(f"Error in _transform: {str(e)}") from e

    def _extract_text(self, chunk: Union[str, BaseMessage]) -> str:
        if isinstance(chunk, str):
            return chunk
        elif isinstance(chunk, (HumanMessage, AIMessage)):
            return str(chunk.content)
        elif isinstance(chunk, BaseMessage):
            # Handle other message types (e.g., SystemMessage, ToolMessage)
            return str(chunk.content)
        else:
            raise ValueError(f"Unsupported input type: {type(chunk)}")

    def parse(self, text: str) -> List[Task]:
        try:
            return list(self._transform([text]))
        except Exception as e:
            raise OutputParserException(f"Error parsing input: {str(e)}") from e

    def stream(
            self,
            input: str | BaseMessage,
            config: RunnableConfig | None = None,
            **kwargs: Any | None,
    ) -> Iterator[Task]:
        yield from self.transform([input], config, **kwargs)

    def ingest_token(
            self, token: str, buffer: List[str], thought: Optional[str]
    ) -> Iterator[Tuple[Optional[Task], str]]:
        try:
            buffer.append(token)
            if "\n" in token:
                buffer_ = "".join(buffer).split("\n")
                suffix = buffer_[-1]
                for line in buffer_[:-1]:
                    task, thought = self._parse_task(line, thought)
                    if task:
                        yield task, thought
                buffer.clear()
                buffer.append(suffix)
        except Exception as e:
            raise OutputParserException(f"Error ingesting token: {str(e)}") from e

    def _parse_task(self, line: str, thought: Optional[str] = None):
        task = None
        try:
            if match := re.match(THOUGHT_PATTERN, line):
                # Optionally, action can be preceded by a thought
                thought = match.group(1)
            elif match := re.match(ACTION_PATTERN, line):
                # if action is parsed, return the task, and clear the buffer
                idx, tool_name, args, _ = match.groups()
                idx = int(idx)
                task = instantiate_task(
                    tools=self.tools,
                    idx=idx,
                    tool_name=tool_name,
                    args=args,
                    thought=thought,
                )
                thought = None
            # Else it is just dropped
        except Exception as e:
            raise OutputParserException(f"Error parsing task: {str(e)}") from e
        return task, thought
