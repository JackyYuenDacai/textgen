# TextGen：从 Chat 兼容层走向真正的 Native Responses

结合你现在这版代码，我会把"真正的 Responses"定义成：

> **Responses 不再依赖
> `ChatCompletionRequest → chat_completions() → from_chat()` 这一整条
> Chat 中间层，而是拥有自己的 canonical item model、生成状态机和
> streaming event pipeline。Chat Completions 反而只是这个核心上的
> adapter。**

你现在的实现已经走到大概 **60--70% 的协议兼容**了，但架构核心仍然是
Chat：`/v1/responses` 最终还是调用 `stream_chat_completions()` / Chat
generation，然后 `from_chat()` 或 `StreamConverter` 再翻译回 Responses。

## 1. 先抽一个与 Chat 无关的内部 `GenerationItem` 层

不要再把内部历史首先表示成：

``` python
{
    "role": "assistant",
    "content": "...",
    "tool_calls": [...]
}
```

而改成类似：

``` text
MessageItem
ReasoningItem
FunctionCallItem
FunctionCallOutputItem
CustomToolCallItem
CustomToolCallOutputItem
```

你的 `responses.py` 已经能解析这些 item，但目前在 `prepare()`
里又把它们压回 `messages`，然后构造
`ChatCompletionRequest`。这就是现在最应该拆掉的一层。

目标应该变成：

``` text
Responses request
      ↓
Canonical Items
      ↓
Prompt Renderer
      ↓
Generation Engine
      ↓
Canonical Output Events
      ↓
Responses serializer
```

而不是现在：

``` text
Responses request
      ↓
Canonical-ish items
      ↓
Chat messages
      ↓
ChatCompletionRequest
      ↓
Chat generation
      ↓
ChatCompletion
      ↓
Responses conversion
```

## 2. 把 generator 改成产出语义事件，而不是 Chat chunk

这是最关键的一刀。

现在 backend 大致吐：

``` python
{
    "choices": [{
        "delta": {
            "content": "...",
            "reasoning_content": "...",
            "tool_calls": [...]
        }
    }]
}
```

然后 `StreamConverter` 判断应该变成哪个 Responses event。

真正干净的实现应该让 generation core 直接吐：

``` python
TextDelta(...)
ReasoningDelta(...)
ToolCallStarted(...)
ToolCallArgumentsDelta(...)
ToolCallCompleted(...)
Usage(...)
GenerationCompleted(...)
```

例如：

``` python
@dataclass
class TextDelta:
    text: str

@dataclass
class ReasoningDelta:
    text: str

@dataclass
class ToolCallDelta:
    call_id: str
    name: str | None
    arguments_delta: str

@dataclass
class GenerationDone:
    finish_reason: str
    usage: Usage
```

然后 Responses 层只负责：

``` text
TextDelta
→ response.output_text.delta
```

Chat adapter 则负责：

``` text
TextDelta
→ choices[0].delta.content
```

Anthropic adapter：

``` text
TextDelta
→ content_block_delta
```

**这样 Responses、Chat、Anthropic 才真正是三个平级 API，而不是
Responses/Anthropic 寄生在 Chat 上。**

## 3. Tool calling 成为 generation state machine 的一级状态

你当前 non-stream path 是：

``` python
chat = chat_completions(...)
response = from_chat(request, chat, history)
```

然后 `from_chat()` 再从：

``` python
message["reasoning_content"]
message["tool_calls"]
message["content"]
```

拼成 Responses output item。

建议改成 generator 内部直接维护：

``` text
GENERATING_REASONING
       ↓
GENERATING_TEXT
       ↓
GENERATING_TOOL_CALL
       ↓
WAITING_FOR_TOOL_RESULT
       ↓
COMPLETED
```

这样当模型开始生成 tool call：

``` json
{
  "type": "function_call",
  "call_id": "...",
  "name": "get_weather",
  "arguments": "..."
}
```

它天然就是一个 output item，而不是先伪装成 `assistant.tool_calls[]`
再恢复。

这会直接解决你现在 streaming tool call 必须 buffer 的问题。你当前
`StreamConverter` 大致有：

``` python
self.buffer_text = bool(request.tools) and request.tool_choice != 'none'
```

也就是说只要存在工具能力，就需要先缓存文本来判断最终输出结构。

真正 item-native 后，这个 buffer 理论上可以删掉。

## 4. `previous_response_id` 保存 item graph，而不是可重新转换的 chat history

你现在已经有 ResponseStore，大致：

``` python
store.put(
    response,
    history + response["output"],
    input_count=len(history)
)
```

而 `previous_response_id` 会从 store 恢复 history。

这里建议再往前走一步。

不要思考：

``` text
previous response
→ reconstruct messages
→ prompt
```

而是：

``` text
previous response
→ canonical items
→ context planner
→ prompt
```

例如：

``` python
class ConversationState:
    items: list[Item]

    def append_input(...)
    def append_output(...)
    def unresolved_tool_calls(...)
    def render_for_model(...)
```

这样 `response.output` 和下一次请求的 `previous_response_id`
引用的是**同一种 item representation**。

这才真正符合 Responses 的 mental model。

## 5. Reasoning 也应该成为 item，而不是 `reasoning_content` 字符串

你现在已经能输出类似：

``` python
{
    "type": "reasoning",
    "id": "rs_...",
    "summary": [],
    "content": [{
        "type": "reasoning_text",
        "text": "..."
    }]
}
```

但来源目前仍然是：

``` python
message["reasoning_content"]
```

建议 backend API 直接改成：

``` python
yield ReasoningDelta(text)
```

于是：

``` text
model tokens
  ├─ reasoning channel → ReasoningItem
  └─ final channel     → MessageItem
```

特别是 Qwen3、GPT-OSS、DeepSeek 一类模型，本身就可能存在 reasoning/final
channel 概念。

这时候不要在 Responses adapter 层判断 `<think>` 或
`reasoning_content`。应该由 **model/backend parser** 判断 token 属于哪个
semantic channel。

## 最终架构

``` text
                         ┌─────────────────┐
Responses API ──────────▶│                 │
                         │ Canonical Input │
Chat Completions ───────▶│     Items       │
                         │                 │
Anthropic Messages ─────▶│                 │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Context Planner │
                         │                 │
                         │ history         │
                         │ tools           │
                         │ reasoning       │
                         │ multimodal      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Prompt Renderer │
                         │ / Chat Template │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Generation Core │
                         └────────┬────────┘
                                  │
                                  ▼
                       Semantic Generation Events
                                  │
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
          Responses          ChatCompletion      Anthropic
          serializer          serializer         serializer
```

这时候：

``` python
OAIcompletions.stream_chat_completions(...)
```

就应该逐步变成类似：

``` python
generation.stream(...)
```

而：

``` python
def stream_chat_completions(request):
    canonical = chat_adapter.parse(request)

    for event in generation.stream(canonical):
        yield chat_adapter.serialize(event)
```

Responses：

``` python
def stream_responses(request):
    canonical = responses_adapter.parse(request)

    for event in generation.stream(canonical):
        yield responses_adapter.serialize(event)
```

## 下一步最值得动的 3 个文件/模块

基于现在 repo，我不会先继续给 `responses.py` 加更多 OpenAI 字段。

我会先做这一轮重构：

``` text
modules/api/
    generation.py          ← 新增
    canonical_items.py     ← 新增

    completions.py
    responses.py
    anthropic.py
```

其中 `canonical_items.py`：

``` python
@dataclass
class Message:
    role: str
    content: list

@dataclass
class Reasoning:
    content: list

@dataclass
class FunctionCall:
    id: str
    call_id: str
    name: str
    arguments: str

@dataclass
class FunctionCallOutput:
    call_id: str
    output: object
```

`generation.py`：

``` python
class GenerationEvent: ...

class TextStart(GenerationEvent): ...
class TextDelta(GenerationEvent): ...
class TextDone(GenerationEvent): ...

class ReasoningStart(GenerationEvent): ...
class ReasoningDelta(GenerationEvent): ...
class ReasoningDone(GenerationEvent): ...

class ToolCallStart(GenerationEvent): ...
class ToolCallDelta(GenerationEvent): ...
class ToolCallDone(GenerationEvent): ...

class UsageEvent(GenerationEvent): ...
class DoneEvent(GenerationEvent): ...
```

然后第一阶段**完全不改变外部 API 行为**。

只是把现在：

``` text
backend
→ Chat chunk
→ Responses
```

改成：

``` text
backend
→ GenerationEvent
→ Chat / Responses
```

这一步做好以后，后面实现 `conversation`、`input_items`、更完整
streaming、parallel tools、structured output、compaction 都会容易很多。

## 不需要追求和 OpenAI 云端实现一模一样

**encrypted reasoning 不需要实现。**

你这里是 local LLM。当前遇到 `encrypted_content` 直接拒绝，并保留本地
plaintext reasoning，是合理的设计。

类似下面这些 OpenAI 托管基础设施能力：

-   hosted web search
-   hosted file search
-   Code Interpreter
-   computer use
-   background cloud execution

也不应该为了"真正 Responses"而硬仿。

真正应该兼容的是：

> **API semantics，而不是 OpenAI 的云基础设施。**

所以目标可以定义为：

> **Native Responses architecture + OpenAI Responses wire
> compatibility + local-native execution semantics**

而不是"100% clone OpenAI"。

## 第一刀应该改哪里

如果只选现在第一刀改哪里，我会先把 `completions.py` 中的生成核心从
`ChatCompletion` 数据结构里抽出来。

你现在 `responses.py` 已经相当完整，继续改它的边际收益反而不大。

**真正卡住你的，是 generation core 仍然认为世界的基本单位是 chat
message。**
