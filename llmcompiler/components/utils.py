import re
from typing import List, Dict, Any, Union

from langchain_core.messages import BaseMessage, FunctionMessage

ID_PATTERN = re.compile(r"\$\{?(\d+)\}?")


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]) -> Union[str, List[Any], Any]:
    def replace_match(match):
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    if isinstance(arg, str):
        return ID_PATTERN.sub(replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results
