"""The OpenAI-compatible endpoint (fake provider, temp database)."""

import json

from llm_gateway.providers.base import GenerationFailed, ProviderUnavailable

URL = "/v1/chat/completions"


def ask(client, **body):
    body.setdefault("model", "small:1b")
    body.setdefault("messages", [{"role": "user", "content": "hello"}])
    return client.post(URL, json=body, headers={"Authorization": "Bearer anything"})


def events(response):
    """Parsed `data:` lines of a streamed response."""
    lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def test_models_lists_auto_first(client):
    data = client.get("/v1/models").json()
    assert data["object"] == "list"
    assert [m["id"] for m in data["data"]] == ["auto", "small:1b", "thinker:8b"]


def test_completion_on_a_named_model(client, provider):
    response = ask(client, max_tokens=50, temperature=0.3)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["object"] == "chat.completion" and data["id"].startswith("chatcmpl-")
    assert data["model"] == "small:1b"
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "echo: hello"}
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    sent = provider.requests[-1]
    assert sent.max_tokens == 50 and sent.temperature == 0.3
    assert sent.think is False  # other tools get fast answers unless they ask

    stored = client.get(f"/api/requests/{data['gateway']['request_id']}").json()
    assert stored["source"] == "api" and stored["routing_mode"] == "manual"
    assert stored["executions"][0]["status"] == "completed"
    # It is in History like any other request.
    assert client.get("/api/requests").json()[0]["id"] == stored["id"]


def test_auto_routes_and_reports_the_decision(client):
    data = ask(client, model="auto").json()
    assert data["model"] in ("small:1b", "thinker:8b")
    assert data["gateway"]["routing"]["category"]
    stored = client.get(f"/api/requests/{data['gateway']['request_id']}").json()
    assert stored["routing_mode"] == "automatic" and stored["routing"]["shadow"] is False


def test_conversation_and_system_messages_reach_the_model(client, provider):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Write a Python function to add two numbers."},
        {"role": "assistant", "content": "def add(a, b): return a + b"},
        {"role": "user", "content": [{"type": "text", "text": "Now subtract."}]},
    ]
    data = ask(client, model="auto", messages=messages, reasoning_effort="high").json()
    sent = provider.requests[-1]
    assert sent.system == "Be brief." and sent.prompt == "Now subtract."
    assert [(m.role, m.content) for m in sent.history] == [
        ("user", "Write a Python function to add two numbers."),
        ("assistant", "def add(a, b): return a + b"),
    ]
    assert sent.think is True
    # Routed by the opening message (coding), not by "Now subtract."
    assert data["gateway"]["routing"]["category"] == "coding"

    # A judge is given the earlier turns too.
    request_id = data["gateway"]["request_id"]
    text = client.post("/api/judge-package", json={"request_ids": [request_id]}).json()["text"]
    assert "def add(a, b)" in text


def test_streaming_sends_pieces_then_done(client):
    response = ask(client, stream=True, stream_options={"include_usage": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks = events(response)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "echo: hello"
    assert chunks[1]["model"] == "small:1b" and chunks[1]["object"] == "chat.completion.chunk"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 3

    stored = client.get(f"/api/requests/{chunks[-2]['gateway']['request_id']}").json()
    assert stored["executions"][0]["response"] == "echo: hello"


def test_streamed_thinking_is_kept_apart_from_the_answer(client):
    chunks = events(ask(client, model="thinker:8b", stream=True, reasoning_effort="medium"))
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert {"reasoning_content": "hmm"} in deltas
    assert "hmm" not in "".join(d.get("content", "") for d in deltas)


def test_unknown_model_and_unsupported_messages_are_rejected(client):
    response = ask(client, model="gpt-4o")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
    assert "/v1/models" in response.json()["error"]["message"]

    for messages in (
        [{"role": "assistant", "content": "hi"}],
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
    ):
        response = ask(client, messages=messages)
        assert response.status_code == 400, messages
        assert response.json()["error"]["type"] == "invalid_request_error"
    assert client.get("/api/requests").json() == []  # nothing was stored


def test_failures_use_the_openai_error_shape(client, provider):
    client.get("/v1/models")  # fill the model cache while the provider still answers
    provider.error = GenerationFailed("boom")
    response = ask(client)
    assert response.status_code == 502
    assert response.json()["error"] == {
        "message": "boom",
        "type": "server_error",
        "code": "generation_failed",
        "param": None,
    }
    # In a stream the headers are already sent, so the error is an event.
    chunks = events(ask(client, stream=True))
    assert chunks[-1]["error"]["code"] == "generation_failed"


def test_unreachable_provider_is_a_503(client, provider):
    client.get("/v1/models")  # fill the model cache while the provider still answers
    provider.error = ProviderUnavailable("down")
    assert ask(client).status_code == 503
