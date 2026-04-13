"""vLLM offline model backend.

Wraps :class:`vllm.LLM` as a LangChain :class:`BaseChatModel` so it can be
used directly with ``bind_tools()`` and the ReAct graph — no HTTP server
needed.  Tool-call parsing uses vLLM's built-in Gemma 4 parser.

This is the recommended backend for the Grand Challenge container where
everything runs in a single process.

Requires::

    pip install vllm
"""

import json
import logging
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, model_validator

log = logging.getLogger(__name__)


class ChatVLLM(BaseChatModel):
    """LangChain chat model backed by vLLM offline inference.

    Supports ``bind_tools()`` and structured ``tool_calls`` in responses,
    which is required by the LangGraph ReAct loop.

    Usage::

        from vllm import LLM, SamplingParams
        from chimera_agent_baseline.models.vllm_offline import ChatVLLM

        llm = LLM(model="./model", dtype="auto")
        model = ChatVLLM(
            llm=llm,
            sampling_params=SamplingParams(temperature=1.0, max_tokens=4096),
        )
        model_with_tools = model.bind_tools(tools)
        response = model_with_tools.invoke(messages)
    """

    llm: Any = Field(exclude=True)
    """A :class:`vllm.LLM` instance."""

    sampling_params: Any = Field(exclude=True)
    """A :class:`vllm.SamplingParams` instance."""

    tokenizer: Any = Field(default=None, exclude=True)
    """Tokenizer for decoding with special tokens.  Auto-resolved from *llm*."""

    tool_parser: str = Field(default="gemma4")
    """Name of the vLLM tool-call parser (gemma4, hermes, llama, …)."""

    bound_tools: list[dict] = Field(default_factory=list)
    """OpenAI-format tool schemas, set via :meth:`bind_tools`."""

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _resolve_tokenizer(self) -> "ChatVLLM":
        if self.tokenizer is None:
            self.tokenizer = self.llm.get_tokenizer()
        return self

    # -- LangChain interface ---------------------------------------------------

    @property
    def _llm_type(self) -> str:
        return "vllm-offline"

    def bind_tools(self, tools: list, **kwargs: Any) -> "ChatVLLM":
        """Return a copy of this model with tools bound."""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.model_copy(update={"bound_tools": formatted, "tool_parser": self.tool_parser})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        msg_dicts = _to_openai_messages(messages)
        tools = self.bound_tools or None

        outputs = self.llm.chat(
            messages=msg_dicts,
            sampling_params=self.sampling_params,
            tools=tools,
        )

        output = outputs[0].outputs[0]

        # Decode WITH special tokens so the tool-call parser can see markers
        full_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=False)
        clean_text = output.text  # readable version without special tokens

        # Parse tool calls using the configured parser
        tool_calls = _parse_tool_calls(full_text, self.tool_parser)

        if tool_calls:
            lc_tool_calls = [
                {
                    "name": tc["name"],
                    "args": tc["arguments"],
                    "id": str(uuid4()),
                    "type": "tool_call",
                }
                for tc in tool_calls
            ]
            # Preserve any reasoning text the model produced before the
            # tool calls (e.g. "I need to check the Gleason grading...").
            # Gemma 4 E2B doesn't produce this, but larger models might.
            reasoning = _extract_reasoning_prefix(clean_text, tool_calls)
            message = AIMessage(content=reasoning, tool_calls=lc_tool_calls)
        else:
            message = AIMessage(content=clean_text)

        return ChatResult(generations=[ChatGeneration(message=message)])


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _to_openai_messages(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to OpenAI-format dicts for vLLM chat."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry: dict = {"role": "assistant"}
            if msg.tool_calls:
                entry["content"] = None
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"]) if isinstance(tc["args"], dict) else tc["args"],
                        },
                    }
                    for tc in msg.tool_calls
                ]
            else:
                entry["content"] = msg.content
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                }
            )
        else:
            result.append({"role": "user", "content": str(msg.content)})
    return result


def _extract_reasoning_prefix(clean_text: str, tool_calls: list[dict]) -> str:
    """Extract any reasoning text that appeared before the first tool call.

    Models like Llama 3 or larger Gemma variants may produce text like
    "I need to check the clinical data first" before their tool calls.
    Gemma 4 E2B typically doesn't, but preserving this enables richer
    reasoning traces for participants using bigger models.
    """
    if not clean_text:
        return ""

    # Find where the first tool call starts in the clean text
    first_tool = tool_calls[0]["name"] if tool_calls else ""
    # Look for "call:<name>" pattern (Gemma) or the function name
    for marker in [f"call:{first_tool}", first_tool]:
        idx = clean_text.find(marker)
        if idx > 0:
            return clean_text[:idx].strip()

    return ""


# ---------------------------------------------------------------------------
# Tool-call parsing — dispatches to the right parser based on model format
# ---------------------------------------------------------------------------

# Map parser name → vLLM utils module that has a parse_tool_calls() function.
# Not all parsers have an offline utils module; those fall through to the
# generic JSON parser.
_VLLM_UTILS_MODULES: dict[str, str] = {
    "gemma4": "vllm.tool_parsers.gemma4_utils",
}


def _parse_tool_calls(text: str, parser: str = "gemma4") -> list[dict]:
    """Extract tool calls from decoded model output.

    Tries (in order):
    1. A vLLM offline utils module for the parser (e.g. ``gemma4_utils``)
    2. A generic JSON-based parser that handles the most common formats
       (Hermes, Llama, Qwen, Mistral — anything using JSON tool calls)
    """
    # Try vLLM's model-specific offline parser
    module_path = _VLLM_UTILS_MODULES.get(parser)
    if module_path:
        try:
            import importlib

            mod = importlib.import_module(module_path)
            return mod.parse_tool_calls(text, strict=False)
        except (ImportError, AttributeError):
            log.debug("vLLM parser '%s' unavailable, falling back to generic JSON parser", parser)

    # Generic JSON-based parser — covers Hermes, Llama, Qwen, Mistral, etc.
    results = _parse_json_tool_calls(text)
    if not results and text.strip():
        log.debug("No tool calls parsed from output: %.200s", text)
    return results


def _parse_json_tool_calls(text: str) -> list[dict]:
    """Parse tool calls from models that use JSON inside XML-like tags.

    Handles common patterns::

        <tool_call>{"name": "func", "arguments": {"key": "val"}}</tool_call>
        <|python_tag|>{"name": "func", "parameters": {"key": "val"}}
        {"name": "func", "arguments": {"key": "val"}}

    Also handles Gemma 4's ``call:name{args}`` as a fallback.
    """
    import re

    results = []

    # Pattern 1: JSON inside <tool_call> tags (Hermes, Qwen, many others)
    for match in re.finditer(r"</?tool_call>?\s*(\{.*?\})\s*(?:</tool_call>)?", text, re.DOTALL):
        results.extend(_try_parse_json_tool_call(match.group(1)))

    if results:
        return results

    # Pattern 2: raw JSON objects with "name" and "arguments"/"parameters"
    for match in re.finditer(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:', text, re.DOTALL):
        # Find the full JSON object
        start = match.start()
        depth, i = 0, start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    results.extend(_try_parse_json_tool_call(text[start : i + 1]))
                    break
            i += 1

    if results:
        return results

    # Pattern 3: Gemma 4 bare format — call:name{args}
    for match in re.finditer(r"(?:^|\s|>)call:(\w+)\{(.*?)\}", text, re.DOTALL):
        name, args_str = match.group(1), match.group(2)
        results.append({"name": name, "arguments": _parse_kv_args(args_str)})

    return results


def _try_parse_json_tool_call(text: str) -> list[dict]:
    """Try to parse a JSON string as a tool call."""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []

    if isinstance(obj, dict) and "name" in obj:
        args = obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                args = {}
        return [{"name": obj["name"], "arguments": args}]
    return []


def _parse_kv_args(args_str: str) -> dict:
    """Parse Gemma 4's ``key:<|"|>value<|"|>`` format into a dict."""
    if not args_str.strip():
        return {}

    cleaned = args_str.replace('<|"|>', '"')
    try:
        return json.loads("{" + cleaned + "}")
    except (json.JSONDecodeError, ValueError):
        pass

    import re

    result = {}
    for key, value in re.findall(r'(\w+):\s*"([^"]*)"', cleaned):
        result[key] = value
    if not result:
        for key, value in re.findall(r"(\w+):\s*([^,}]+)", args_str):
            result[key] = value.strip().strip('"').replace('<|"|>', "")
    return result
