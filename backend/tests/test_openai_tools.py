"""Function calling through the OpenAI-compatible endpoint (fake provider)."""

import json

from test_openai_api import URL, events

from llm_gateway.providers.base import ToolCall

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
QUESTION = "What is the weather in Paris?"


def ask(client, **body):
    body.setdefault("model", "small:1b")
    body.setdefault("messages", [{"role": "user", "content": QUESTION}])
    body.setdefault("tools", [WEATHER])
    return client.post(URL, json=body)


def test_tool_call_comes_back_in_openai_format(client, provider):
    provider.tool_calls_for[QUESTION] = [ToolCall(name="get_weather", arguments={"city": "Paris"})]
    response = ask(client)
    assert response.status_code == 200, response.text
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    (call,) = choice["message"]["tool_calls"]
    assert call["type"] == "function" and call["id"].startswith("call_")
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    assert data["gateway"]["tool_calls_recovered_from_text"] is False
    assert provider.chat_tools[-1] == [WEATHER]

    # Stored like any other API call, readable in History.
    (stored,) = client.get("/api/requests").json()
    assert stored["source"] == "api" and stored["prompt"] == QUESTION
    execution = stored["executions"][0]
    assert execution["status"] == "completed"
    assert execution["response"] == '[tool call] get_weather({"city": "Paris"})'


def test_tool_results_are_passed_back_to_the_model(client, provider):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": QUESTION},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "18 C, sunny"},
    ]
    provider.content_for["18 C, sunny"] = "It is 18 C and sunny in Paris."
    response = ask(client, messages=messages)
    assert response.status_code == 200, response.text
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"] == {"role": "assistant", "content": "It is 18 C and sunny in Paris."}

    sent = provider.chat_messages[-1]
    assert sent[0] == {"role": "system", "content": "Be brief."}
    assert sent[2]["tool_calls"] == [
        {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
    ]
    assert sent[3] == {"role": "tool", "content": "18 C, sunny", "tool_name": "get_weather"}

    (stored,) = client.get("/api/requests").json()
    assert stored["prompt"] == QUESTION and stored["system"] == "Be brief."
    assert "Result: 18 C, sunny" in stored["source_material"]
    assert 'Called get_weather({"city": "Paris"})' in stored["source_material"]


def test_calls_written_as_text_are_converted_only_for_offered_functions(client, provider):
    provider.content_for[QUESTION] = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
    data = ask(client).json()
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert data["gateway"]["tool_calls_recovered_from_text"] is True

    provider.content_for[QUESTION] = '{"name": "delete_everything", "arguments": {}}'
    data = ask(client).json()
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "delete_everything" in data["choices"][0]["message"]["content"]


def test_model_without_tool_support_is_refused(client):
    response = ask(client, model="thinker:8b")
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "tools_not_supported"
    assert "small:1b" in error["message"]


def test_auto_switches_to_a_model_that_can_call_tools(client, provider):
    provider.tool_calls_for[QUESTION] = [ToolCall(name="get_weather", arguments={"city": "x"})]
    data = ask(client, model="auto").json()
    assert data["model"] == "small:1b"
    (stored,) = client.get("/api/requests").json()
    assert stored["routing"]["selected_model"] == "small:1b"

    # A reasoning prompt routes to thinker:8b, which cannot call tools.
    prompt = "Weigh the trade-offs step by step and reason it through: which plan is better?"
    provider.tool_calls_for[prompt] = []
    data = ask(client, model="auto", messages=[{"role": "user", "content": prompt}]).json()
    assert data["model"] == "small:1b"
    routed = next(r for r in client.get("/api/requests").json() if r["prompt"] == prompt)
    assert "cannot call them" in routed["routing"]["explanation"]


def test_streamed_tool_call(client, provider):
    provider.tool_calls_for[QUESTION] = [ToolCall(name="get_weather", arguments={"city": "Paris"})]
    chunks = events(ask(client, stream=True, stream_options={"include_usage": True}))
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    (call,) = chunks[1]["choices"][0]["delta"]["tool_calls"]
    assert call["index"] == 0 and call["function"]["name"] == "get_weather"
    assert chunks[2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[3]["usage"]["prompt_tokens"] == 11


def test_tool_choice_none_sends_no_tools(client, provider):
    ask(client, tool_choice="none")
    assert provider.chat_tools[-1] == []


def test_a_failure_is_stored_and_reported(client, provider):
    from llm_gateway.providers.base import GenerationFailed

    client.get("/v1/models")
    provider.error = GenerationFailed("boom")
    response = ask(client)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "generation_failed"
    provider.error = None
    (stored,) = client.get("/api/requests").json()
    assert stored["executions"][0]["status"] == "failed"
