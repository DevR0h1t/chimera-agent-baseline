"""LangGraph ReAct agent graph.

Implements the Thought -> Action -> Observation loop as a cyclic
StateGraph with two nodes:

* **agent** -- calls the LLM (with tools bound) to decide the next action
* **tools** -- executes whichever tool the LLM selected

A conditional edge after the *agent* node routes to *tools* on a tool call,
or to END when the LLM produces a final text response.
"""

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

log = logging.getLogger(__name__)


def create_graph(
    tools: list[BaseTool],
    model: BaseChatModel,
    system_prompt: str,
    step_timeout: int = 120,
) -> StateGraph:
    """Build and compile the ReAct agent graph."""
    model_with_tools = model.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    graph = builder.compile()
    graph.step_timeout = step_timeout
    log.info("Compiled ReAct graph with %d tools", len(tools))
    return graph
