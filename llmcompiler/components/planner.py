import json
from typing import Sequence, Dict, List, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableSerializable
from langchain_core.tools import BaseTool

from llmcompiler.components.output_parser import LLMCompilerPlanParser, Task


def create_planner(llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate) -> \
        RunnableSerializable[list, dict]:
    def get_tool_descriptions() -> str:
        """Get a string of all registered tool names, descriptions, and schemas."""
        descriptions = []
        for i, tool in enumerate(tools):
            schema = tool.args if tool and hasattr(tool, 'args') else {}
            description = f"{i + 1}. {tool.name}: {tool.description}\n\tSchema: {json.dumps(schema)}\n"
            descriptions.append(description)
        return "\n".join(descriptions)

    tool_descriptions = get_tool_descriptions()

    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    replanner_prompt = base_prompt.partial(
        replan="""
         - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results (given as Observation) of each plan and a general thought (given as Thought) about the executed results. You MUST use these information to create the next plan under "Current Plan".
         - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.
         - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.
         - You must continue the task index from the end of the previous one. Do not repeat task indices.
        """,
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list) -> bool:
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list) -> Dict[str, List[BaseMessage]]:
        return {"messages": state}

    def wrap_and_get_last_index(state: list) -> Dict[str, List[BaseMessage]]:
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, ToolMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": state}

    return RunnableBranch(
        (should_replan, wrap_and_get_last_index | replanner_prompt),
        wrap_messages | planner_prompt,
    ) | llm | LLMCompilerPlanParser(tools=tools)


def stream_plan(planner: RunnableSerializable[list, dict], messages: list[BaseMessage]) -> Iterator[Task]:
    return planner.stream(messages)