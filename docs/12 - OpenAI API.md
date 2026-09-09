## OpenAI/Anthropic-compatible API

The main API for this project provides OpenAI and Anthropic compatibility, including Chat, Completions, Messages, and a local subset of the Responses API.

* It is 100% offline and private.
* It doesn't create any logs.
* It doesn't connect to OpenAI.
* It doesn't use the openai-python library.

### Starting the API

Add `--api` to your command-line flags.

* To create a public Cloudflare URL, add the `--public-api` flag.
* To listen on your local network, add the `--listen` flag.
* To change the port, which is 5000 by default, use `--api-port 1234` (change 1234 to your desired port number).
* To use SSL, add `--ssl-keyfile key.pem --ssl-certfile cert.pem`. ⚠️ **Note**: this doesn't work with `--public-api` since Cloudflare already uses HTTPS by default.
* To use an API key for authentication, add `--api-key yourkey`.

### WorkBuddy current-time context and prompt caching

For API chat requests in `instruct` mode (including Anthropic Messages), TextGen
moves exact `<current_time>...</current_time>` blocks from leading system/developer
context to a system message after the conversation, before any trailing assistant
prefill. The current time is preserved. Its changing value therefore no longer
invalidates the stable instructions and earlier history. This applies to plain-text
tag contents, including multiline timestamps; user messages, tool results, and
other tags such as `<time>` are unchanged. If WorkBuddy puts the block in a user
message, this workaround does not relocate it.

This behavior defaults to enabled. Send `"cache_friendly_current_time": false` to
disable it, particularly for chat templates that only support a leading system
message. It does not affect web chat or raw text completions. Restart TextGen after
updating the code; the first request warms the new prompt layout. Reuse still
depends on the backend retaining the matching prefix and on other prompt content
remaining unchanged. Moving the time each turn can still require reprocessing the
previous turn's tail; it does not guarantee a full-history cache hit.

### Examples

For the documentation with all the endpoints, parameters and their types, consult `http://127.0.0.1:5000/docs` or the [typing.py](https://github.com/oobabooga/textgen/blob/main/modules/api/typing.py) file.

The official examples in the [OpenAI documentation](https://platform.openai.com/docs/api-reference) should also work, and the same parameters apply (although the API here has more optional parameters).

#### Responses

`POST /v1/responses` uses the loaded model and the same generation backend as
Chat Completions. Existing Chat and Anthropic endpoints remain available. After
installing this code, restart TextGen normally to register the new routes.
The `model` field does not load or switch models; the response identifies the
model actually loaded. These routes use the existing `--api-key` authentication.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:5000/v1", api_key="yourkey")
response = client.responses.create(
    model="local",
    instructions="Answer briefly.",
    input="Explain what a KV cache does.",
    max_output_tokens=256,
)
print(response.output_text)

followup = client.responses.create(
    model="local",
    previous_response_id=response.id,
    input="How does it help the next turn?",
    max_output_tokens=256,
)
print(followup.output_text)
```

Input may be a string or an array of messages with `user`, `assistant`, `system`,
or `developer` roles. Content supports `input_text`, assistant `output_text`, and
user `input_image` with an HTTP(S) URL or image data URL. Image support depends on
the loaded model. Uploaded `file_id`, audio, and video inputs are not implemented.
Top-level `instructions` apply only to the current response and are not inherited
through `previous_response_id`; send them again when needed. Explicit system
messages in `input` remain in the stored history.

Streaming uses named Responses SSE events, including `response.created`,
`response.output_text.delta`, `response.output_item.done`, and a terminal
`response.completed`, `response.incomplete`, or `response.failed`. There is no
Chat-style `[DONE]` marker. Handle all three terminal states:

```python
events = client.responses.create(
    model="local", input="Hello", stream=True, store=False, max_output_tokens=128,
)
for event in events:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event.type == "response.incomplete":
        print("\nIncomplete:", event.response.incomplete_details.reason)
    elif event.type == "response.failed":
        print("\nFailed:", event.response.error.message)
```

The SDK's `responses.stream()` helper also works for completed responses. In
openai-python 3.10.0, `get_final_response()` requires `response.completed`; read
the `response.incomplete` or `response.failed` event directly in those cases.
Disconnecting cancels the generation and does not save a partial response.

Responses generation is serialized through a process-local FIFO queue: one
request runs at a time, including both streaming and non-streaming requests.
Waiting requests use asynchronous tickets rather than occupying inference
threads; a disconnected waiter is removed. Backend errors and early tool-call
completion release the slot and close the underlying generator. This queue is
specific to `/v1/responses`; Chat/Anthropic requests retain their own existing
scheduling behavior. It does not alter the model's cache allocation or context
limit. A waiting SSE connection receives keepalive comments rather than model
output until it is admitted.

Function tools use the flat Responses schema and are executed by the client:

```python
tools = [{
    "type": "function", "name": "get_temperature", "strict": False,
    "description": "Read a temperature sensor.",
    "parameters": {"type": "object", "properties": {}},
}]
response = client.responses.create(model="local", input="Read the temperature.", tools=tools)
outputs = []
for item in response.output:
    if item.type == "function_call":
        # Validate the name and arguments, then execute your own implementation.
        # This example supplies a simulated measurement.
        if item.name == "get_temperature":
            outputs.append({"type": "function_call_output", "call_id": item.call_id,
                            "output": '{"celsius": 25}'})
if outputs:
    response = client.responses.create(model="local", previous_response_id=response.id,
                                       input=outputs, tools=tools)
    print(response.output_text)
```

Return an output for every pending function call. `call_id` identifies the call;
it differs from the output item's `id`. Both `tool_choice="auto"` and `"none"`
are supported. The backend detects tool calls after generating their markup,
so when tools are enabled, text and function arguments are buffered until the
result is known. Plain-text requests without tools stream incrementally.

`store` defaults to `true`. Stored responses and history live only in CPU RAM,
with a one-hour TTL, at most 64 responses, and a 64 MiB total serialized-size
budget. Oldest entries are evicted when limits are reached; a single history
over the budget returns an error (use `store=false`). This budget counts text
and embedded image data, not GPU KV tensors. Restarting clears stored history.
All callers with access to this API share this storage; it is not per-user
storage. Treat response IDs as private conversation references.

Stored data is kept as immutable snapshots. Retrieving a response copies only
the response, not its potentially large input history; continuing a conversation
copies only the history. Large copies occur outside the storage lock, and
request preparation/retrieval run outside the HTTP event loop.

- `GET /v1/responses/{response_id}` retrieves a stored result.
- `DELETE /v1/responses/{response_id}` deletes that stored result.
- `store=false` supports stateless operation: replay prior input, `response.output`,
  and new input items in the next request.

Response storage does not purge or resize the GPU KV cache. Context capacity
remains controlled by the loaded model and existing generation settings.
`prompt_cache_retention="in-memory"` is accepted for local best-effort KV reuse;
`"24h"` is rejected because this backend cannot guarantee that retention.
Neither `prompt_cache_key` nor per-turn `client_metadata` is inserted into the
model prompt. Cache hits depend on matching token prefixes, model state and
available backend cache pages, not the response ID or the cache key alone.
Keep instructions and tool definitions consistent between tool turns; tool
definitions are normalized by the existing prompt renderer, so reordering
equivalent tools does not change their rendered order.
`truncation="disabled"` (default) rejects oversized input instead of silently
discarding its prefix, including when ExLlamaV3 image embeddings exceed the
loaded capacity. Explicit `truncation="auto"` uses the existing backend clipping
behavior. `max_output_tokens` is an output ceiling; available context and EOS may
end generation earlier.

This is a compatibility subset, not every OpenAI hosted feature:

- Client custom tools are adapted to a JSON `input` string for the local model
  and returned as `custom_tool_call` with the original raw text, including
  newlines. Streaming includes `response.custom_tool_call_input.delta` and
  `.done`; replay uses `custom_tool_call_output`. Text, regex, and Lark formats
  are supported. Lark imports are limited to `common`; grammar definitions are
  limited to 65,536 characters and grammar-validated input to 262,144 characters.
  Regex matching has a 250 ms timeout. Complex Lark grammars can be expensive;
  this is intended for trusted local client grammars such as patch syntax.
  Dependencies are listed in `requirements/responses.txt` and included by the
  full/portable requirement sets.
- Function `strict=true` schemas and custom grammars are checked after generation,
  before any executable call is delivered. This is output validation, not
  constrained decoding: an invalid call fails the response, rather than being
  repaired or executed. Tool schema references must be local. Undeclared tools
  and missing/duplicate call IDs also fail before delivery.
- `parallel_tool_calls=false` adds a one-tool instruction and validates that the
  result contains at most one call. If the model generates multiple calls, the
  response fails without delivering them. This option is separate from the FIFO
  queue, which serializes generation requests regardless of tool settings.
- Assistant input `phase` values `commentary` and `final_answer` are preserved
  through history conversion; the local backend does not infer new phase labels.
- Hosted tools, background jobs, Conversations, WebSockets, response compaction,
  uploaded files, structured text JSON schemas, and forced tool selection
  are not implemented. Unsupported request fields and
  unsupported tool types return errors.
- Local reasoning is exposed as plaintext `reasoning_text`; no encrypted content
  or summaries are generated. Codex's `include=["reasoning.encrypted_content"]`
  and `reasoning.summary` preferences are accepted, but the response explicitly
  reports `reasoning.summary=null` and carries only available plaintext reasoning.
  Replaying encrypted content from a hosted model remains unsupported.
  `reasoning.effort` is passed to the chat template, whose support varies by model.
- Function tool namespaces are flattened for the local template and restored as
  `namespace`/`name` in response items. `text.verbosity` becomes a concise/detailed
  answer instruction, not an enforced length constraint.
- Usage totals come from the Chat backend. Unknown cache and reasoning token
  breakdowns are omitted, so clients requiring strict validation of every hosted
  usage field are not supported. Standard SDK parsing is supported. ExLlamaV3
  Responses use request-local generated token counts and native finish reasons,
  including speculative output limits; available cached-token counts are included.
- `user`, `safety_identifier`, and `prompt_cache_key` are accepted for client
  compatibility; they do not provide identity isolation, moderation, or a new
  caching policy. Service tiers only accept `auto`/`default` with local behavior.
  Codex's `client_metadata` is also accepted as inert client metadata; it is not
  inserted into the prompt or used as authorization.

##### Codex configuration

The route is enabled automatically with `--api`; there is no separate Responses
switch on TextGen. A Codex provider must use `wire_api="responses"` and a base URL
ending in `/v1`. For a local-only Codex profile, use:

```toml
model_provider = "textgen"
model = "Qwen3.8-27B-EXL3-3.5bpw" # Replace with your loaded model name.
web_search = "disabled"          # TextGen has no OpenAI-hosted search service.

[model_providers.textgen]
name = "TextGen"
base_url = "http://127.0.0.1:5000/v1"
wire_api = "responses"
```

This can be placed in a separate Codex profile rather than replacing your other
provider settings. If you enable TextGen API authentication, configure the
provider's `env_key` to reference an environment variable containing that key.
Function and custom client tools are supported; hosted web search and encrypted
history from a different provider are not. Start a new
conversation when switching from a hosted reasoning model to the local model.
Validation errors identify the rejected field in both `error.message` and
`error.param`, including when the client UI displays only the message.

Protocol reference: [OpenAI official Responses API documentation](https://developers.openai.com/api/reference/resources/responses/methods/create).

#### Chat completions

Works best with instruction-following models. If the "instruction_template" variable is not provided, it will be detected automatically from the model metadata.

```shell
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

#### Completions

```shell
curl http://127.0.0.1:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "This is a cake recipe:\n\n1.",
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

#### SSE streaming

```shell
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "stream": true
  }'
```

#### Tool/Function calling

Use a model with tool calling support (Qwen, Mistral, GPT-OSS, etc). Tools are passed via the `tools` parameter and the prompt is automatically formatted using the model's Jinja2 template.

When the model decides to call a tool, the response will have `finish_reason: "tool_calls"` and a `tool_calls` array with structured function names and arguments. You then execute the tool, send the result back as a `role: "tool"` message, and continue until the model responds with `finish_reason: "stop"`.

Some models call multiple tools in parallel (Qwen, Mistral), while others call one at a time (GPT-OSS). The loop below handles both styles.

```python
import json
import requests

url = "http://127.0.0.1:5000/v1/chat/completions"

# Define your tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time in a given timezone",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone string"},
                },
                "required": ["timezone"]
            }
        }
    },
]


def execute_tool(name, arguments):
    """Replace this with your actual tool implementations."""
    if name == "get_weather":
        return {"temperature": 22, "condition": "sunny", "humidity": 45}
    elif name == "get_time":
        return {"time": "2:30 PM", "timezone": "JST"}
    return {"error": f"Unknown tool: {name}"}


messages = [{"role": "user", "content": "What time is it in Tokyo and what's the weather like there?"}]

# Tool-calling loop: keep going until the model gives a final answer
for _ in range(10):
    response = requests.post(url, json={"messages": messages, "tools": tools}).json()
    choice = response["choices"][0]

    if choice["finish_reason"] == "tool_calls":
        # Add the assistant's response (with tool_calls) to history
        messages.append({
            "role": "assistant",
            "content": choice["message"]["content"],
            "tool_calls": choice["message"]["tool_calls"],
        })

        # Execute each tool and add results to history
        for tool_call in choice["message"]["tool_calls"]:
            name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            result = execute_tool(name, arguments)

            print(f"Tool call: {name}({arguments}) => {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(result),
            })
    else:
        # Final answer
        print(f"\nAssistant: {choice['message']['content']}")
        break
```

#### Multimodal/vision (llama.cpp and ExLlamaV3)

##### With /v1/chat/completions (recommended!)

```shell
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Please describe what you see in this image."},
          {"type": "image_url", "image_url": {"url": "https://github.com/turboderp-org/exllamav3/blob/master/examples/media/cat.png?raw=true"}}
        ]
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

For base64-encoded images, just replace the inner "url" value with this format: `data:image/FORMAT;base64,BASE64_STRING` where FORMAT is the file type (png, jpeg, gif, etc.) and BASE64_STRING is your base64-encoded image data.

##### With /v1/completions

```shell
curl http://127.0.0.1:5000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "About image <__media__> and image <__media__>, what I can say is that the first one"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://github.com/turboderp-org/exllamav3/blob/master/examples/media/cat.png?raw=true"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://github.com/turboderp-org/exllamav3/blob/master/examples/media/strawberry.png?raw=true"
            }
          }
        ]
      }
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

For base64-encoded images, just replace the inner "url" values with this format: `data:image/FORMAT;base64,BASE64_STRING` where FORMAT is the file type (png, jpeg, gif, etc.) and BASE64_STRING is your base64-encoded image data.

#### List models

```shell
curl -k http://127.0.0.1:5000/v1/internal/model/list \
  -H "Content-Type: application/json"
```

#### Load model

```shell
curl -k http://127.0.0.1:5000/v1/internal/model/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
    "args": {
      "ctx_size": 32768,
      "cache_type": "q8_0"
    }
  }'
```

You can also set a default instruction template for all subsequent API requests by passing `instruction_template` (a template name from `user_data/instruction-templates/`) or `instruction_template_str` (a raw Jinja2 string):

```shell
curl -k http://127.0.0.1:5000/v1/internal/model/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
    "instruction_template": "Alpaca"
  }'
```

#### Chat completions with characters

```shell
curl http://127.0.0.1:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello! Who are you?"
      }
    ],
    "mode": "chat-instruct",
    "character": "Example",
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

#### Image generation

```shell
curl http://127.0.0.1:5000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "an orange tree",
    "steps": 9,
    "cfg_scale": 0,
    "batch_size": 1,
    "batch_count": 1
  }'
```

You need to load an image model first. You can do this via the UI, or by adding `--image-model your_model_name` when launching the server.

The output is a JSON object containing a `data` array. Each element has a `b64_json` field with the base64-encoded PNG image:

```json
{
  "created": 1764791227,
  "data": [
    {
      "b64_json": "iVBORw0KGgo..."
    }
  ]
}
```

#### Logits

```shell
curl -k http://127.0.0.1:5000/v1/internal/logits \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Who is best, Asuka or Rei? Answer:",
    "use_samplers": false
  }'
```

#### Logits after sampling parameters

```shell
curl -k http://127.0.0.1:5000/v1/internal/logits \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Who is best, Asuka or Rei? Answer:",
    "use_samplers": true,
    "top_k": 3
  }'
```

#### Python chat example

```python
import requests

url = "http://127.0.0.1:5000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

history = []

while True:
    user_message = input("> ")
    history.append({"role": "user", "content": user_message})
    data = {
        "messages": history,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20
    }

    response = requests.post(url, headers=headers, json=data, verify=False)
    assistant_message = response.json()['choices'][0]['message']['content']
    history.append({"role": "assistant", "content": assistant_message})
    print(assistant_message)
```

#### Python chat example with streaming

Start the script with `python -u` to see the output in real time.

```python
import requests
import sseclient  # pip install sseclient-py
import json

url = "http://127.0.0.1:5000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

history = []

while True:
    user_message = input("> ")
    history.append({"role": "user", "content": user_message})
    data = {
        "stream": True,
        "messages": history,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20
    }

    stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
    client = sseclient.SSEClient(stream_response)

    assistant_message = ''
    for event in client.events():
        payload = json.loads(event.data)
        chunk = payload['choices'][0]['delta']['content']
        assistant_message += chunk
        print(chunk, end='')

    print()
    history.append({"role": "assistant", "content": assistant_message})
```

#### Python completions example with streaming

Start the script with `python -u` to see the output in real time.

```python
import json
import requests
import sseclient  # pip install sseclient-py

url = "http://127.0.0.1:5000/v1/completions"

headers = {
    "Content-Type": "application/json"
}

data = {
    "prompt": "This is a cake recipe:\n\n1.",
    "max_tokens": 512,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "stream": True,
}

stream_response = requests.post(url, headers=headers, json=data, verify=False, stream=True)
client = sseclient.SSEClient(stream_response)

print(data['prompt'], end='')
for event in client.events():
    payload = json.loads(event.data)
    print(payload['choices'][0]['text'], end='')

print()
```

#### Python example with API key

Replace

```python
headers = {
    "Content-Type": "application/json"
}
```

with

```python
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer yourPassword123"
}
```

in any of the examples above.

#### Python parallel requests example

The API supports handling multiple requests in parallel. For ExLlamaV3, this works out of the box. For llama.cpp, you need to pass `--parallel N` to set the number of concurrent slots.

```python
import concurrent.futures
import requests

url = "http://127.0.0.1:5000/v1/chat/completions"
prompts = [
    "Write a haiku about the ocean.",
    "Explain quantum computing in simple terms.",
    "Tell me a joke about programmers.",
]

def send_request(prompt):
    response = requests.post(url, json={
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
    })
    return response.json()["choices"][0]["message"]["content"]

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(send_request, prompts))

for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

### Environment variables

The following environment variables can be used (they take precedence over everything else):

| Variable Name          | Description                                                                                        | Example Value              |
|------------------------|------------------------------------|----------------------------|
| `OPENEDAI_PORT`           | Port number         |             5000               |
| `OPENEDAI_CERT_PATH`      | SSL certificate file path         |            cert.pem                |
| `OPENEDAI_KEY_PATH`       | SSL key file path                    |             key.pem               |
| `OPENEDAI_DEBUG`          | Enable debugging (set to 1)    | 1                          |
| `OPENEDAI_EMBEDDING_MODEL` | Embedding model (if applicable) |          sentence-transformers/all-mpnet-base-v2                  |
| `OPENEDAI_EMBEDDING_DEVICE` | Embedding device (if applicable) |           cuda                 |

### Third-party application setup

You can usually force an application that uses the OpenAI API to connect to the local API by using the following environment variables:

```shell
OPENAI_API_HOST=http://127.0.0.1:5000
```

or

```shell
OPENAI_API_KEY=sk-111111111111111111111111111111111111111111111111
OPENAI_API_BASE=http://127.0.0.1:5000/v1
```

With the [official python openai client](https://github.com/openai/openai-python) (v1.x), the address can be set like this:

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-111111111111111111111111111111111111111111111111",
    base_url="http://127.0.0.1:5000/v1"
)

response = client.chat.completions.create(
    model="x",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

With the [official Node.js openai client](https://github.com/openai/openai-node) (v4.x):

```js
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: "http://127.0.0.1:5000/v1",
});

const response = await client.chat.completions.create({
  model: "x",
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(response.choices[0].message.content);
```
### Embeddings (alpha)

Embeddings requires `sentence-transformers` installed, but chat and completions will function without it loaded. The embeddings endpoint is currently using the HuggingFace model: `sentence-transformers/all-mpnet-base-v2` for embeddings. This produces 768 dimensional embeddings. The model is small and fast. This model and embedding size may change in the future.

| model name             | dimensions | input max tokens | speed | size | Avg. performance |
| ---------------------- | ---------- | ---------------- | ----- | ---- | ---------------- |
| all-mpnet-base-v2      | 768        | 384              | 2800  | 420M | 63.3             |
| all-MiniLM-L6-v2       | 384        | 256              | 14200 | 80M  | 58.8             |

In short, the all-MiniLM-L6-v2 model is 5x faster, 5x smaller ram, 2x smaller storage, and still offers good quality. Stats from (https://www.sbert.net/docs/pretrained_models.html). To change the model from the default you can set the environment variable `OPENEDAI_EMBEDDING_MODEL`, ex. "OPENEDAI_EMBEDDING_MODEL=all-MiniLM-L6-v2".

Warning: You cannot mix embeddings from different models even if they have the same dimensions. They are not comparable.

### Compatibility

| API endpoint              | notes                                                                       |
| ------------------------- | --------------------------------------------------------------------------- |
| /v1/chat/completions      | Use with instruction-following models. Supports streaming, tool calls.      |
| /v1/completions           | Text completion endpoint.                                                   |
| /v1/embeddings            | Using SentenceTransformer embeddings.                                       |
| /v1/images/generations    | Image generation, response_format='b64_json' only.                         |
| /v1/moderations           | Basic support via embeddings.                                               |
| /v1/models                | Lists models. Currently loaded model first.                                 |
| /v1/models/{id}           | Returns model info.                                                         |
| /v1/audio/\*              | Supported.                                                                  |
| /v1/images/edits          | Not yet supported.                                                          |
| /v1/images/variations     | Not yet supported.                                                          |

#### Applications

Almost everything needs the `OPENAI_API_KEY` and `OPENAI_API_BASE` environment variables set, but there are some exceptions.

| Compatibility | Application/Library  | Website                                                                        | Notes                                                                                     |
| ------------- | -------------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| ✅❌          | openai-python        | https://github.com/openai/openai-python                                        | Use `OpenAI(base_url="http://127.0.0.1:5000/v1")`. Only the endpoints from above work.   |
| ✅❌          | openai-node          | https://github.com/openai/openai-node                                          | Use `new OpenAI({baseURL: "http://127.0.0.1:5000/v1"})`. See example above.              |
| ✅            | anse                 | https://github.com/anse-app/anse                                               | API Key & URL configurable in UI, Images also work.                                       |
| ✅            | shell_gpt            | https://github.com/TheR1D/shell_gpt                                            | OPENAI_API_HOST=http://127.0.0.1:5000                                                    |
| ✅            | gpt-shell            | https://github.com/jla/gpt-shell                                               | OPENAI_API_BASE=http://127.0.0.1:5000/v1                                                 |
| ✅            | gpt-discord-bot      | https://github.com/openai/gpt-discord-bot                                      | OPENAI_API_BASE=http://127.0.0.1:5000/v1                                                 |
| ✅            | OpenAI for Notepad++ | https://github.com/Krazal/nppopenai                                            | api_url=http://127.0.0.1:5000 in the config file, or environment variables.               |
| ✅            | vscode-openai        | https://marketplace.visualstudio.com/items?itemName=AndrewButson.vscode-openai | OPENAI_API_BASE=http://127.0.0.1:5000/v1                                                 |
| ✅❌          | langchain            | https://github.com/hwchase17/langchain                                         | Use `base_url="http://127.0.0.1:5000/v1"`. Results depend on model and prompt formatting. |
