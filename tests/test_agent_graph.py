"""Smoke tests for the ReAct + form-fill LangGraph.

We don't actually call a real LLM — instead we use ``FakeListChatModel``
to script the agent's responses, including a tool call, a final
assistant message, and a JSON form-fill response that the
``PydanticOutputParser`` then validates.
"""

from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from chimera_agent_baseline.agent.form_fill import (
    _called_tools_from_messages,
    _final_assistant_text,
)
from chimera_agent_baseline.agent.graph import create_graph
from chimera_agent_baseline.output.schema import Weight, eligible_variables


@tool
def get_mri_report(case_id: str) -> str:
    """Returns a stub MRI report."""
    return '{"pirads": "5", "psa_density": 0.68, "cspca_pred": 0.79}'


@tool
def get_family_history(case_id: str) -> str:
    """Returns family history."""
    return '{"family_history": "Yes"}'


def test_extracts_called_tools_from_messages():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "get_mri_report", "args": {"case_id": "X"}}]),
        ToolMessage(content="ok", tool_call_id="1", name="get_mri_report"),
        AIMessage(content="done"),
    ]
    assert _called_tools_from_messages(msgs) == {"get_mri_report"}


def test_final_assistant_text_skips_tool_call_messages():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "get_mri_report", "args": {}}]),
        ToolMessage(content="ok", tool_call_id="1", name="get_mri_report"),
        AIMessage(content="My final answer"),
    ]
    assert _final_assistant_text(msgs) == "My final answer"


def test_eligibility_after_tool_call():
    elig = eligible_variables(1, called_tools={"get_mri_report"})
    assert "pirads" in elig
    assert "family_history" not in elig


class _StubModel(FakeMessagesListChatModel):
    """Test double for ChatVLLM:

    * scripts the agent's responses via ``responses`` (FakeMessagesListChatModel)
    * stubs ``bind_tools`` (default raises NotImplementedError) to return self

    The form_fill node now invokes the model directly and parses the
    response as JSON, so the third scripted response should be a valid
    JSON-payload AIMessage rather than a structured-output adapter result.
    """

    def bind_tools(self, tools, **kwargs):
        return self


def test_graph_runs_through_form_fill():
    # Scripted messages: one tool call, one final answer, one form-fill JSON.
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call-1", "name": "get_mri_report", "args": {"case_id": "PT-T1"}}],
    )
    final_msg = AIMessage(content="Recommend biopsy: PI-RADS 5, high PSAD.")
    form_payload = {
        "case_id": "PT-T1",
        "task": 1,
        "biopsy_decision": True,
        "confidence": "clear",
        "variable_weights": {
            "psa": "decisive",
            "age": "noted",
            "dre": "not_used",
            "comorbidity": "not_used",
            "prior_biopsy": "not_used",
            "pirads": "decisive",
            "psa_density": "decisive",
            "prostate_volume": "noted",
            "cspca": "important",
        },
        "reasoning": "PI-RADS 5 plus PSAD 0.68 is decisive — biopsy required.",
    }
    form_fill_msg = AIMessage(content=json.dumps(form_payload))
    model = _StubModel(responses=[tool_call_msg, final_msg, form_fill_msg])

    graph = create_graph(
        tools=[get_mri_report, get_family_history],
        model=model,
        system_prompt="(test)",
    )
    out = graph.invoke(
        {
            "messages": [HumanMessage(content="Should this patient be biopsied?")],
            "case_id": "PT-T1",
            "task": 1,
        },
        {"recursion_limit": 10},
    )

    assert "structured_response" in out
    sr = out["structured_response"]
    # Form-fill stamped task + decision are present.
    assert sr["task"] == 1
    assert sr["biopsy_decision"] is True
    # Family-history weight padded to "not_used" because get_family_history
    # was not actually called by the agent (it is gated on that tool).
    assert sr["variable_weights"]["family_history"] == Weight.NOT_USED.value
    # Pirads kept as decisive.
    assert sr["variable_weights"]["pirads"] == Weight.DECISIVE.value
