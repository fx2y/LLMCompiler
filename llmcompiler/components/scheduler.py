import time
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, List, Iterable

from langchain_core.messages import AIMessage, ToolMessage, BaseMessage
from langchain_core.runnables import chain as as_runnable
from typing_extensions import TypedDict

from llmcompiler.components.output_parser import Task
from llmcompiler.components.utils import _resolve_arg, _get_observations


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


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
def schedule_tasks(scheduler_input: SchedulerInput) -> List[BaseMessage]:
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
    tool_messages = []
    for k, (name, task_args, obs) in new_observations.items():
        tool_call_id = str(uuid.uuid4())
        ai_message = AIMessage(content="", tool_calls=[
            {
                "name": name,
                "args": task_args,
                "id": tool_call_id,
                "type": "tool_call",
            }
        ])
        tool_message = ToolMessage(
            name=name,
            content=str(obs),
            tool_call_id=tool_call_id,
            additional_kwargs={"idx": k, "args": task_args}
        )
        tool_messages.extend([ai_message, tool_message])

    return tool_messages
