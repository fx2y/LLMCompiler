import itertools
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Sequence, Union

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBranch, chain as as_runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

from llmcompiler.components.output_parser import LLMCompilerPlanParser


class Task(TypedDict):
    idx: int
    tool: Union[BaseTool, str]
    args: Union[str, Dict[str, Any]]
    dependencies: List[int]


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


class FinalResponse(BaseModel):
    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


class State(TypedDict):
    messages: List[BaseMessage]


def create_planner(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        base_prompt: ChatPromptTemplate
) -> RunnableBranch:
    tool_descriptions = "\n".join(
        f"{i + 1}. {tool.description}\n" for i, tool in enumerate(tools)
    )

    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    replanner_prompt = base_prompt.partial(
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
               "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
               'You MUST use these information to create the next plan under "Current Plan".\n'
               ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
               " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
               " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
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
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": state}

    return RunnableBranch(
        (should_replan, wrap_and_get_last_index | replanner_prompt),
        wrap_messages | planner_prompt,
    ) | llm | LLMCompilerPlanParser(tools=tools)


def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results


def _execute_task(task: Task, observations: Dict[int, Any], config: Dict[str, Any]) -> Any:
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use

    args = task["args"]
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            resolved_args = args
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
            f" Args could not be resolved. Error: {repr(e)}"
        )

    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}."
            f" Args resolved to {resolved_args}. Error: {repr(e)})"
        )


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]) -> Union[str, List[Any], Any]:
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
def schedule_task(task_inputs: Dict[str, Any], config: Dict[str, Any]) -> None:
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        import traceback
        observation = traceback.format_exception()
    observations[task["idx"]] = observation


def schedule_pending_task(
        task: Task,
        observations: Dict[int, Any],
        retry_after: float = 0.2
) -> None:
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    futures = []
    retry_after = 0.25

    with ThreadPoolExecutor() as executor:
        for task in tasks:
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if deps and (any([dep not in observations for dep in deps])):
                futures.append(
                    executor.submit(
                        schedule_pending_task,
                        task,
                        observations,
                        retry_after
                    )
                )
            else:
                schedule_task.invoke(dict(task=task, observations=observations))

        wait(futures)

    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name,
            content=str(obs),
            additional_kwargs={"idx": k, "args": task_args}
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages


@as_runnable
def plan_and_schedule(state: Dict[str, List[BaseMessage]]) -> Dict[str, List[BaseMessage]]:
    messages = state["messages"]
    tasks = planner.stream(messages)
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {"messages": messages, "tasks": tasks}
    )
    return {"messages": scheduled_tasks}


def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return response + [
            SystemMessage(
                content=f"Context from last attempt: {decision.action.feedback}"
            )
        ]
    else:
        return {"messages": response + [AIMessage(content=decision.action.response)]}


def select_recent_messages(state: Dict[str, List[BaseMessage]]) -> Dict[str, List[BaseMessage]]:
    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}


def should_continue(state: Dict[str, List[BaseMessage]]) -> Union[str, END]:
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage):
        return END
    return "plan_and_schedule"


def create_llm_compiler_agent(
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        planner_prompt: ChatPromptTemplate,
        joiner_prompt: ChatPromptTemplate
) -> StateGraph:
    planner = create_planner(llm, tools, planner_prompt)
    joiner = select_recent_messages | joiner_prompt | llm.with_structured_output(JoinOutputs) | _parse_joiner_output

    graph_builder = StateGraph(State)
    graph_builder.add_node("plan_and_schedule", plan_and_schedule)
    graph_builder.add_node("join", joiner)
    graph_builder.add_edge("plan_and_schedule", "join")
    graph_builder.add_conditional_edges(
        "join",
        should_continue,
        {
            "plan_and_schedule": "plan_and_schedule",
            END: END
        }
    )
    graph_builder.add_edge(START, "plan_and_schedule")

    return graph_builder.compile()


# Usage example:
if __name__ == "__main__":
    from langchain_community.tools.tavily_search import TavilySearchResults
    from tools.math_tools import get_math_tool

    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"

    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    calculate = get_math_tool(llm)
    search = TavilySearchResults(max_results=1)
    tools = [search, calculate]

    planner_prompt = hub.pull("wfh/llm-compiler")
    joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(examples="")

    agent = create_llm_compiler_agent(llm, tools, planner_prompt, joiner_prompt)

    question = "What's the GDP of New York?"
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    print(result[END][-1].content)
