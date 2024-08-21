from typing import Union, List, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.constants import END


class FinalResponse(BaseModel):
    """The final response/answer."""
    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed.")


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""
    thought: str = Field(description="The chain of thought reasoning for the selected action")
    action: Union[FinalResponse, Replan]


def create_joiner(llm: BaseChatModel, joiner_prompt: ChatPromptTemplate):
    def _parse_joiner_output(decision: JoinOutputs) -> Dict[str, List[SystemMessage | AIMessage]]:
        response = [AIMessage(content=f"Thought: {decision.thought}")]
        if isinstance(decision.action, Replan):
            return {"messages": response + [
                SystemMessage(content=f"Context from last attempt: {decision.action.feedback}")]}
        else:
            return {"messages": response + [AIMessage(content=decision.action.response)]}

    def select_recent_messages(state) -> dict:
        messages = state["messages"]
        selected = []
        for msg in messages[::-1]:
            selected.append(msg)
            if isinstance(msg, HumanMessage):
                break
        return {"messages": selected[::-1]}

    return select_recent_messages | joiner_prompt | llm.with_structured_output(JoinOutputs) | _parse_joiner_output


def should_continue(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage):
        return END
    return "plan_and_schedule"
