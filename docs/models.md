# Model Configuration

## How it works

The agent uses [vLLM](https://docs.vllm.ai) for in-process inference. The
model is loaded directly into GPU memory — no HTTP server, no network access.
This is the same setup used in the Grand Challenge container.

The model must support **tool calling** — the ability to output structured
function calls that the ReAct loop can execute.

## Default model (Gemma 4 E2B-it)

The baseline ships with [Gemma 4 E2B-it](https://huggingface.co/google/gemma-4-E2B-it),
a 2.3B-parameter model with native function calling (~10GB in bf16).

```yaml
# configs/config.yaml
model:
  model_id: google/gemma-4-E2B-it
  tool_parser: gemma4
```

Download the weights:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it', local_dir='model/')"
```

When `model/` contains weights, they're loaded locally. When empty, vLLM
downloads from HuggingFace Hub using `model_id`.

## Swapping to a different model

Two things to change in `configs/config.yaml`:

```yaml
model:
  model_id: meta-llama/Llama-3.1-8B-Instruct
  tool_parser: llama
```

Then download the weights:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct', local_dir='model/')"
```

No code changes needed.

## Tool call parsers

Different models use different formats for function calling. The `tool_parser`
field tells the system how to parse the model's output.

| Parser | Models | Format |
|--------|--------|--------|
| `gemma4` | Gemma 4 | `call:func_name{key:value}` |
| `hermes` | Hermes, many fine-tunes | `<tool_call>{"name": ...}</tool_call>` |
| `llama` | Llama 3.x | JSON-based tool calls |
| `mistral` | Mistral, Mixtral | `[TOOL_CALLS]` + JSON |
| `pythonic` | Python-style calls | `func_name(arg=val)` |

List all parsers in your vLLM installation:

```bash
ls $(python -c "import vllm.tool_parsers; print(vllm.tool_parsers.__path__[0])")/*_tool_parser.py
```

The parser has a fallback chain: model-specific vLLM parser → generic JSON
parser → Gemma 4 bare format. For most models, setting the right
`tool_parser` value is sufficient.

## Experiment overlays

Test different models without editing the base config:

```yaml
# configs/experiment/llama8b.yaml
# @package _global_
model:
  model_id: meta-llama/Llama-3.1-8B-Instruct
  tool_parser: llama
generation:
  temperature: 0.6
```

```bash
make run RUN_ARGS="+experiment=llama8b"
```

## Choosing a model

Key considerations:

- **Tool calling support** — the model must output structured function calls.
  Most instruction-tuned models support this.
- **Size vs. GPU memory** — Grand Challenge offers T4 (16GB) or A10G (24GB).
  Gemma 4 E2B (2.3B, ~10GB bf16) fits comfortably. Larger models may need
  quantization.
- **vLLM support** — verify your model architecture is
  [supported by vLLM](https://docs.vllm.ai/en/latest/models/supported_models.html).
- **Tool parser** — check that a matching `tool_parser` exists. The generic
  JSON fallback handles most cases.
