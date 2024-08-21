import itertools
from typing import List, Dict, Any, Iterator

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph

from llmcompiler.components.joiner import create_joiner, \
    should_continue
from llmcompiler.components.planner import create_planner, stream_plan
from llmcompiler.components.scheduler import schedule_tasks


class LLMCompiler:
    def __init__(
            self,
            llm: BaseChatModel = None,
            tools: List[BaseTool] = None,
            planner_prompt: ChatPromptTemplate = None,
            joiner_prompt: ChatPromptTemplate = None
    ):
        self.llm = llm
        self.tools = tools
        self.planner_prompt = planner_prompt
        self.joiner_prompt = joiner_prompt

        self.planner = create_planner(self.llm, self.tools, self.planner_prompt)
        self.scheduler = schedule_tasks
        self.joiner = create_joiner(self.llm, self.joiner_prompt)

        self.graph = self._build_graph()

        self.langfuse_handler = CallbackHandler()

    def _build_graph(self) -> CompiledStateGraph:
        graph_builder = StateGraph(MessagesState)

        graph_builder.add_node("plan_and_schedule", self._plan_and_schedule)
        graph_builder.add_node("join", self.joiner)

        graph_builder.add_edge("plan_and_schedule", "join")
        graph_builder.add_conditional_edges(
            "join",
            should_continue,
            {
                "plan_and_schedule": "plan_and_schedule",
                END: END
            }
        )
        graph_builder.set_entry_point("plan_and_schedule")

        return graph_builder.compile()

    def _plan_and_schedule(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        tasks = stream_plan(self.planner, messages)
        try:
            tasks = itertools.chain([next(tasks)], tasks)
        except StopIteration:
            tasks = iter([])
        scheduled_tasks = self.scheduler.invoke({
            "messages": messages,
            "tasks": tasks
        })
        return {"messages": messages + scheduled_tasks}

    def run(self, input_message: str) -> str:
        final_state = self.graph.invoke({"messages": [HumanMessage(content=input_message)]},
                                        config={"callbacks": [self.langfuse_handler]})
        return final_state["messages"][-1].content

    def stream(self, input_message: str) -> Iterator[Dict[str, Any]]:
        return self.graph.stream({"messages": [HumanMessage(content=input_message)]},
                                 config={"callbacks": [self.langfuse_handler]})


if __name__ == '__main__':
    from llmcompiler.config import config as cfg
    from llmcompiler.tools.math_tools import get_math_tool

    llm = ChatOpenAI(
        model_name=cfg.LLM_MODEL_NAME,
        openai_api_base=cfg.OPENAI_API_BASE,
        openai_api_key=cfg.OPENAI_API_KEY,
        temperature=0.2
    )
    calculate = get_math_tool(llm)
    search = DuckDuckGoSearchResults(max_results=1)
    tools = [search, calculate]

    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a user query, create a plan to solve it with the utmost parallelizability. Each plan should comprise an action from the following {num_tools} types:
        {tool_descriptions}
        {num_tools}. join(): Collects and combines results from prior actions.

         - An LLM agent is called upon invoking join() to either finalize the user query or wait until the plans are executed.
         - join should always be the last action in the plan, and will be called in two scenarios:
           (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.
           (b) if the answer cannot be determined in the planning phase before you execute the plans. Guidelines:
         - Each action described above contains input/output types and description.
            - You must strictly adhere to the input and output types for each action.
            - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
         - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.
         - Each action MUST have a unique ID, which is strictly increasing.
         - Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
         - Always call join as the last action in the plan. Say '<END_OF_PLAN>' after you call join
         - Ensure the plan maximizes parallelizability.
         - Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.
         - Never introduce new actions other than the ones provided."""),
        ("placeholder", "{messages}"),
        ("system", """Remember, ONLY respond with the task list in the correct format! E.g.:
        idx. tool(arg_name=args)"""),
    ])

    joiner_prompt = ChatPromptTemplate.from_messages([
        ("system", """Solve a question answering task. Here are some guidelines:
         - In the Assistant Scratchpad, you will be given results of a plan you have executed to answer the user's question.
         - Thought needs to reason about the question based on the Observations in 1-2 sentences.
         - Ignore irrelevant action results.
         - If the required information is present, give a concise but complete and helpful answer to the user's question.
         - If you are unable to give a satisfactory finishing answer, replan to get the required information. Respond in the following format:

        Thought: <reason about the task results and whether you have sufficient information to answer the question>
        Action: <action to take>
        Available actions:
         (1) Finish(the final answer to return to the user): returns the answer and finishes the task.
         (2) Replan(the reasoning and other information that will help you plan again. Can be a line of any length): instructs why we must replan"""),
        ("placeholder", "{messages}"),
        ("system", """Using the above previous actions, decide whether to replan or finish. If all the required information is present. You may finish. If you have made many attempts to find the information without success, admit so and respond with whatever information you have gathered so the user can work well with you.

        {examples}"""),
    ]).partial(examples="")

    compiler = LLMCompiler(llm, tools, planner_prompt, joiner_prompt)

    # Simple question
    result = compiler.run("What's the GDP of New York?")
    print(result)

    # Multi-hop question
    for step in compiler.stream("What's the oldest parrot alive, and how much longer is that than the average?"):
        print(step)
        print("---")

    # Multi-step math
    result = compiler.run("What's ((3*(4+5)/0.5)+3245) + 8? What's 32/4.23? What's the sum of those two values?")
    print(result)

    for step in compiler.stream("What's the temperature in SF raised to the 3rd power?"):
        print(step)
        print("---")
