"""Automatic routing end to end against the fake provider."""

from test_runs import make_batch, submit, wait_batch, wait_request

from llm_gateway.config import Settings
from llm_gateway.main import create_app
from llm_gateway.providers.base import GenerationFailed, ModelInfo, ProviderUnavailable
from llm_gateway.schemas import PromptInput
from llm_gateway.store import Store

CLEAR_QA = "What is the capital of Australia?"
CLEAR_REASONING = "Analyse the trade-offs between renting and buying a home in a growing city."
UNCLEAR = "Blue green seven lamp."


def auto(client, prompt, **extra):
    response = client.post("/api/requests", json={"prompt": prompt, **extra})
    assert response.status_code == 201, response.text
    return response.json()


def wait_routed(client, request_id):
    import time

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = client.get(f"/api/requests/{request_id}").json()
        done = all(e["status"] in ("completed", "failed") for e in record["executions"])
        if record["routing"] and done:
            return record
        time.sleep(0.01)
    raise AssertionError(f"not routed: {record}")


def test_automatic_request_is_routed_then_run(client, provider):
    created = auto(client, CLEAR_QA)
    assert created["routing_mode"] == "automatic"
    assert created["executions"] == [] and created["routing"] is None

    done = wait_routed(client, created["id"])
    routing = done["routing"]
    assert routing["category"] == "general_qa"
    assert routing["selected_model"] == "small:1b"
    assert routing["fallback_model"] == "thinker:8b"
    assert routing["decided_by"] == "signals"
    assert routing["shadow"] is False and routing["fallback_used"] is False
    assert 0.6 <= routing["confidence"] <= 0.95
    assert "general Q&A" in routing["explanation"] and "small:1b" in routing["explanation"]
    assert routing["signals"]["matched"]["general_qa"]
    assert [e["model"] for e in done["executions"]] == ["small:1b"]
    assert done["executions"][0]["status"] == "completed"
    assert provider.classifier_calls == 0  # clear request: no classifier model needed


def test_category_decides_the_model(client):
    done = wait_routed(client, auto(client, CLEAR_REASONING)["id"])
    assert done["routing"]["category"] == "reasoning"
    assert done["executions"][0]["model"] == "thinker:8b"


def test_manual_request_has_no_routing(client):
    done = wait_request(client, submit(client)["id"])
    assert done["routing"] is None and done["routing_mode"] == "manual"


def test_unsure_request_asks_the_classifier_model(client, provider):
    provider.classifier_answer = '{"category": "reasoning", "complexity": "medium"}'
    done = wait_routed(client, auto(client, UNCLEAR)["id"])
    routing = done["routing"]

    assert provider.classifier_calls == 1
    assert (routing["decided_by"], routing["category"]) == ("llm", "reasoning")
    assert routing["llm"]["model"] == "small:1b"
    assert routing["confidence"] == 0.6
    assert done["executions"][0]["model"] == "thinker:8b"


def test_invalid_classifier_answer_falls_back_to_the_rules(client, provider):
    provider.classifier_answer = "I think this is about lamps."
    done = wait_routed(client, auto(client, UNCLEAR)["id"])
    routing = done["routing"]

    assert routing["decided_by"] == "signals_fallback"
    assert routing["category"] == "general_qa"
    assert "invalid classifier answer" in routing["llm"]["error"]
    assert routing["confidence"] <= 0.5
    assert done["executions"][0]["status"] == "completed"  # the request still ran


def test_classifier_switched_off(make_client, provider):
    client = make_client(router_llm="off")
    done = wait_routed(client, auto(client, UNCLEAR)["id"])
    assert provider.classifier_calls == 0
    assert done["routing"]["decided_by"] == "signals_fallback"
    assert client.get("/api/status").json()["router"]["llm_enabled"] is False


def test_unsure_between_categories_with_the_same_model_skips_the_classifier(client, provider):
    # Classification and structured output both route to small:1b in the test registry.
    done = wait_routed(
        client, auto(client, "Classify these fruits by colour, as JSON: apple, banana, plum.")["id"]
    )
    routing = done["routing"]
    assert provider.classifier_calls == 0
    assert routing["decided_by"] == "signals"
    assert "same model" in routing["explanation"]
    assert routing["confidence"] >= 0.7


def failing_for(provider, bad_model, error):
    original = provider.generate

    async def generate(request, model):
        if model.name == bad_model and not request.response_schema:
            raise error
        return await original(request, model)

    provider.generate = generate


def test_fallback_model_runs_when_the_selected_model_fails(client, provider):
    failing_for(provider, "thinker:8b", GenerationFailed("out of memory"))
    done = wait_routed(client, auto(client, CLEAR_REASONING)["id"])

    assert [(e["model"], e["status"]) for e in done["executions"]] == [
        ("thinker:8b", "failed"),
        ("small:1b", "completed"),
    ]
    assert done["routing"]["fallback_used"] is True
    assert done["routing"]["selected_model"] == "thinker:8b"  # the decision itself is unchanged


def test_fallback_is_tried_only_once(client, provider):
    provider.generate = _always_failing(GenerationFailed("out of memory"))
    done = wait_routed(client, auto(client, CLEAR_REASONING)["id"])
    assert [(e["model"], e["status"]) for e in done["executions"]] == [
        ("thinker:8b", "failed"),
        ("small:1b", "failed"),
    ]


def _always_failing(error):
    async def generate(request, model):
        raise error

    return generate


def test_no_fallback_when_the_provider_is_down_mid_run(client, provider):
    client.get("/api/models")
    provider.generate = _always_failing(ProviderUnavailable("connection lost"))
    done = wait_routed(client, auto(client, CLEAR_REASONING)["id"])
    assert len(done["executions"]) == 1
    assert done["routing"]["fallback_used"] is False


def test_uninstalled_first_choice_is_skipped_at_decision_time(client):
    done = wait_routed(
        client, auto(client, "Write a Python function that reverses a string.")["id"]
    )
    routing = done["routing"]
    assert routing["category"] == "coding"
    assert routing["selected_model"] == "thinker:8b"  # coder:7b is not installed
    assert "not installed: coder:7b" in routing["explanation"]


def test_category_without_a_suitable_model_escalates(client):
    prompt = "Patient is a 45-year-old female presenting with fatigue and weight gain. TSH 12.4."
    done = wait_routed(client, auto(client, prompt)["id"])
    routing = done["routing"]
    assert routing["category"] == "medical"
    assert routing["selected_model"] == "small:1b"  # last resort
    assert routing["escalate"] is True
    assert "medical" in routing["escalation_reason"]


def test_routing_still_decides_when_the_provider_is_down(client, provider):
    provider.error = ProviderUnavailable("Cannot reach Ollama")
    done = wait_routed(client, auto(client, CLEAR_QA)["id"])
    assert done["routing"]["selected_model"] == "small:1b"
    assert "availability unknown" in done["routing"]["explanation"]
    assert done["executions"][0]["error_code"] == "provider_unavailable"


def test_no_models_installed_is_recorded_without_an_execution(client, provider):
    provider.models = []
    done = wait_routed(client, auto(client, CLEAR_QA)["id"])
    assert done["executions"] == []
    assert done["routing"]["selected_model"] is None
    assert done["routing"]["escalate"] is True
    assert done["routing"]["confidence"] == 0


def test_unprofiled_model_is_listed_but_not_chosen(client, provider):
    provider.models.append(ModelInfo(name="newcomer:2b", size_bytes=1, num_ctx=4096))
    models = {m["name"]: m for m in client.get("/api/models").json()["models"]}
    assert models["newcomer:2b"]["profiled"] is False
    assert models["small:1b"]["profiled"] is True
    assert models["small:1b"]["strengths"] == "Quick general model"
    assert "reasoning" in models["thinker:8b"]["categories"]

    done = wait_routed(client, auto(client, CLEAR_QA)["id"])
    assert done["routing"]["selected_model"] != "newcomer:2b"
    # Manual selection still works.
    manual = wait_request(client, submit(client, models=["newcomer:2b"])["id"])
    assert manual["executions"][0]["status"] == "completed"


def test_follow_up_to_a_routed_answer_is_not_routed_again(client, provider):
    done = wait_routed(client, auto(client, CLEAR_REASONING)["id"])
    execution = done["executions"][0]
    response = client.post(f"/api/executions/{execution['id']}/follow-ups", json={"prompt": "Why?"})
    follow_up = wait_request(client, response.json()["id"])
    assert follow_up["routing"] is None
    assert [e["model"] for e in follow_up["executions"]] == ["thinker:8b"]


def test_preview_decides_without_running_or_storing(client, provider):
    response = client.post("/api/route", json={"prompt": CLEAR_REASONING})
    assert response.status_code == 200
    body = response.json()
    assert (body["category"], body["selected_model"]) == ("reasoning", "thinker:8b")
    assert provider.calls == []
    assert client.get("/api/requests").json() == []

    # The preview only asks the classifier model when told to.
    client.post("/api/route", json={"prompt": UNCLEAR})
    assert provider.classifier_calls == 0
    client.post("/api/route", json={"prompt": UNCLEAR, "use_llm": True})
    assert provider.classifier_calls == 1


def test_batch_records_shadow_decisions_but_runs_every_model(client, provider):
    created = make_batch(client, prompts=[CLEAR_QA, CLEAR_REASONING])
    batch = wait_batch(client, created["id"])

    assert batch["counts"]["completed"] == 4
    picks = [r["routing"]["selected_model"] for r in batch["requests"]]
    assert picks == ["small:1b", "thinker:8b"]
    assert all(r["routing"]["shadow"] for r in batch["requests"])
    assert all(r["routing_mode"] == "manual" for r in batch["requests"])
    for request in batch["requests"]:
        assert [e["model"] for e in request["executions"]] == ["small:1b", "thinker:8b"]


def test_broken_registry_still_routes_with_low_confidence(make_client, tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("policy: [oops")
    client = make_client(registry_path=path)
    assert "could not be loaded" in client.get("/api/status").json()["router"]["registry_error"]

    done = wait_routed(client, auto(client, CLEAR_QA)["id"])
    assert done["routing"]["selected_model"] in ("small:1b", "thinker:8b")
    assert done["routing"]["confidence"] <= 0.5
    assert done["executions"][0]["status"] == "completed"


def test_request_left_unrouted_by_a_restart_is_closed(tmp_path, provider, registry_path):
    from fastapi.testclient import TestClient

    database = tmp_path / "restart.db"
    store = Store(database)
    record = store.create_request(PromptInput(prompt="never routed"), [], routing_mode="automatic")
    store.close()

    settings = Settings(_env_file=None, database_path=database, registry_path=registry_path)
    with TestClient(create_app(provider=provider, settings=settings)) as client:
        reopened = client.get(f"/api/requests/{record.id}").json()
    assert reopened["routing"]["decided_by"] == "none"
    assert reopened["routing"]["selected_model"] is None
    assert "interrupted" in reopened["routing"]["explanation"]
