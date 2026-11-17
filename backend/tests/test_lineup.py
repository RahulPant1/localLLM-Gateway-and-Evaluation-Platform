"""Hiding, restoring and adding models to existing tests (fake provider, temp database)."""

import json

from test_runs import wait_batch
from test_tool_use import SUITE, call, make_suite_client, wait_run

from llm_gateway.providers.base import ModelInfo


def new_model(provider, name="fresh:4b", tools=True):
    caps = ["completion", "tools"] if tools else ["completion"]
    provider.models.append(ModelInfo(name=name, size_bytes=1, capabilities=caps, num_ctx=4096))


def make_batch(client, models):
    response = client.post(
        "/api/batches", json={"name": "B", "prompts": ["one", "two"], "models": models}
    )
    assert response.status_code == 201, response.text
    return wait_batch(client, response.json()["id"])


def test_hidden_model_disappears_everywhere_and_comes_back(client):
    batch = make_batch(client, ["small:1b", "thinker:8b"])
    single = client.post("/api/requests", json={"prompt": "solo", "models": ["thinker:8b"]})
    assert single.status_code == 201

    assert client.post("/api/lineup/hide", json={"model": "thinker:8b"}).status_code == 200

    names = [m["name"] for m in client.get("/api/models").json()["models"]]
    assert "thinker:8b" not in names
    detail = client.get(f"/api/batches/{batch['id']}").json()
    assert detail["models"] == ["small:1b"]
    assert detail["counts"]["completed"] == 2
    assert all(e["model"] == "small:1b" for r in detail["requests"] for e in r["executions"])
    # A request answered only by the hidden model is not listed at all.
    prompts = [r["prompt"] for r in client.get("/api/requests").json()]
    assert "solo" not in prompts
    # It cannot be run while hidden.
    refused = client.post("/api/requests", json={"prompt": "x", "models": ["thinker:8b"]})
    assert refused.status_code == 409
    assert "hidden" in refused.json()["detail"]

    lineup = client.get("/api/lineup").json()
    entry = next(m for m in lineup["models"] if m["name"] == "thinker:8b")
    assert entry["hidden"] is True

    client.post("/api/lineup/restore", json={"model": "thinker:8b"})
    detail = client.get(f"/api/batches/{batch['id']}").json()
    assert detail["models"] == ["small:1b", "thinker:8b"]
    assert detail["counts"]["completed"] == 4
    assert "solo" in [r["prompt"] for r in client.get("/api/requests").json()]


def test_run_all_adds_a_new_model_to_every_batch_and_tool_run(make_client, provider, tmp_path):
    client = make_suite_client(make_client, tmp_path)
    batch = make_batch(client, ["small:1b"])
    provider.tool_calls_for["area of 10"] = [call("area", base=10)]
    run_id = client.post(
        "/api/tool-use/runs", json={"suite": "mini", "models": ["small:1b"]}
    ).json()["id"]
    wait_run(client, run_id)

    new_model(provider)
    lineup = client.get("/api/lineup").json()
    entry = next(m for m in lineup["models"] if m["name"] == "fresh:4b")
    assert entry["not_run"] == 2 + len(SUITE["cases"])

    result = client.post("/api/lineup/run-all", json={"model": "fresh:4b"}).json()
    assert (result["queued_answers"], result["queued_tool_cases"]) == (2, 2)

    detail = wait_batch(client, batch["id"])
    assert detail["models"] == ["small:1b", "fresh:4b"]
    assert detail["counts"]["completed"] == 4
    run = wait_run(client, run_id)
    assert run["models"] == ["small:1b", "fresh:4b"]
    fresh = next(m for m in run["per_model"] if m["model"] == "fresh:4b")
    assert fresh["passed"] == 2

    lineup = client.get("/api/lineup").json()
    entry = next(m for m in lineup["models"] if m["name"] == "fresh:4b")
    assert entry["not_run"] == 0
    assert entry["unjudged"] == 2
    # Running it again finds nothing left to do.
    again = client.post("/api/lineup/run-all", json={"model": "fresh:4b"}).json()
    assert again["queued_answers"] == again["queued_tool_cases"] == 0


def test_run_all_skips_tool_tests_for_models_without_tool_support(client, provider):
    make_batch(client, ["small:1b"])
    new_model(provider, "plain:4b", tools=False)
    result = client.post("/api/lineup/run-all", json={"model": "plain:4b"}).json()
    assert result["queued_answers"] == 2
    assert any("no tool support" in n for n in result["notes"])


def test_run_all_brings_a_hidden_model_back(client):
    make_batch(client, ["small:1b"])
    client.post("/api/lineup/hide", json={"model": "thinker:8b"})
    result = client.post("/api/lineup/run-all", json={"model": "thinker:8b"}).json()
    assert result["queued_answers"] == 2
    assert "back in the lineup" in json.dumps(result["notes"])


def test_run_all_unknown_model(client):
    assert client.post("/api/lineup/run-all", json={"model": "nope"}).status_code == 404


def test_a_batch_and_a_tool_run_do_not_alternate(make_client, provider, tmp_path):
    """Two long jobs take turns as whole jobs, so models are not swapped on every call."""
    client = make_suite_client(make_client, tmp_path)
    provider.delay = 0.02
    batch = client.post(
        "/api/batches", json={"name": "B", "prompts": ["a", "b", "c"], "models": ["small:1b"]}
    ).json()
    run_id = client.post(
        "/api/tool-use/runs", json={"suite": "mini", "models": ["small:1b"]}
    ).json()["id"]
    wait_batch(client, batch["id"])
    wait_run(client, run_id)
    kinds = [kind for kind, _ in provider.timeline]
    # One switch between the two jobs, not one per generation.
    switches = sum(1 for a, b in zip(kinds, kinds[1:], strict=False) if a != b)
    assert switches == 1, kinds
