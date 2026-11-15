"""Requests, batches, history, export and evaluation import (fake provider, temp database)."""

import time

import pytest

from llm_gateway.config import Settings
from llm_gateway.main import create_app
from llm_gateway.providers.base import GenerationFailed, ProviderUnavailable
from llm_gateway.store import Store

TERMINAL = {"completed", "failed"}


def wait_request(client, request_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/api/requests/{request_id}").json()
        if all(e["status"] in TERMINAL for e in record["executions"]):
            return record
        time.sleep(0.01)
    raise AssertionError(f"request did not finish: {record}")


def wait_batch(client, batch_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        batch = client.get(f"/api/batches/{batch_id}").json()
        if batch["counts"]["queued"] == 0 and batch["counts"]["running"] == 0:
            return batch
        time.sleep(0.01)
    raise AssertionError(f"batch did not finish: {batch['counts']}")


def submit(client, **body):
    body.setdefault("prompt", "hello")
    body.setdefault("models", ["small:1b"])
    response = client.post("/api/requests", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_request_returns_queued_then_completes(client):
    created = submit(client)
    assert [e["status"] for e in created["executions"]] == ["queued"]
    assert created["routing_mode"] == "manual"

    done = wait_request(client, created["id"])
    execution = done["executions"][0]
    assert execution["status"] == "completed"
    assert execution["response"] == "echo: hello"
    assert execution["model"] == "small:1b"
    assert execution["execution_ms"] >= 0 and execution["queue_wait_ms"] >= 0
    assert execution["started_at"] and execution["finished_at"]
    assert execution["metrics"]["num_ctx"] == 4096


def test_source_material_is_appended_to_the_prompt(client):
    created = submit(client, prompt="Summarise.", source_material="The sky is blue.")
    done = wait_request(client, created["id"])
    assert done["source_material"] == "The sky is blue."
    assert done["executions"][0]["response"] == (
        "echo: Summarise.\n\nSource material:\nThe sky is blue."
    )


def test_unavailable_model_is_recorded_as_failed_and_others_still_run(client, provider):
    created = submit(client, models=["missing:1b", "small:1b"])
    done = wait_request(client, created["id"])
    by_model = {e["model"]: e for e in done["executions"]}

    assert by_model["missing:1b"]["status"] == "failed"
    assert by_model["missing:1b"]["error_code"] == "model_not_found"
    assert by_model["small:1b"]["status"] == "completed"
    assert provider.calls == ["small:1b"]


def test_provider_down_is_recorded_not_lost(client, provider):
    provider.error = ProviderUnavailable("Cannot reach Ollama")
    done = wait_request(client, submit(client)["id"])
    execution = done["executions"][0]
    assert (execution["status"], execution["error_code"]) == ("failed", "provider_unavailable")
    assert "Cannot reach Ollama" in execution["error_detail"]
    # The request itself is still in history.
    assert client.get("/api/requests").json()[0]["id"] == done["id"]


def test_generation_failure_keeps_earlier_results(client, provider):
    first = wait_request(client, submit(client, prompt="one")["id"])
    client.get("/api/models")
    provider.generate = _failing(GenerationFailed("out of memory"))
    second = wait_request(client, submit(client, prompt="two")["id"])

    assert second["executions"][0]["error_code"] == "generation_failed"
    again = client.get(f"/api/requests/{first['id']}").json()
    assert again["executions"][0]["response"] == "echo: one"


def _failing(error):
    async def generate(request, model, on_delta=None):
        raise error

    return generate


def test_unexpected_error_is_contained(client, provider):
    provider.generate = _failing(RuntimeError("bug"))
    done = wait_request(client, submit(client)["id"])
    assert done["executions"][0]["error_code"] == "internal_error"
    # The queue is free for the next request.
    assert client.get("/api/status").json()["queue"]["busy"] is False


def test_comparison_runs_models_one_at_a_time(client, provider):
    provider.delay = 0.02
    created = submit(client, models=["small:1b", "thinker:8b", "small:1b"])
    assert len(created["executions"]) == 2  # duplicate model removed
    done = wait_request(client, created["id"])

    assert provider.max_running == 1
    assert {e["status"] for e in done["executions"]} == {"completed"}


def test_history_is_newest_first_and_paged(client):
    ids = [wait_request(client, submit(client, prompt=f"p{i}")["id"])["id"] for i in range(3)]
    listed = [r["id"] for r in client.get("/api/requests").json()]
    assert listed == ids[::-1]
    page = client.get("/api/requests", params={"limit": 1, "offset": 1}).json()
    assert [r["id"] for r in page] == [ids[1]]


def test_validation(client):
    assert client.post("/api/requests", json={"prompt": "", "models": ["a"]}).status_code == 422
    assert client.post("/api/requests", json={"prompt": "x", "models": []}).status_code == 422
    assert client.get("/api/requests/nope").status_code == 404
    assert client.get("/api/batches/nope").status_code == 404


def test_history_survives_restart_and_unfinished_work_is_failed(tmp_path, provider, registry_path):
    database = tmp_path / "restart.db"
    store = Store(database)
    from llm_gateway.schemas import PromptInput

    record = store.create_request(PromptInput(prompt="left hanging"), ["small:1b"])
    store.mark_running(record.executions[0].id)
    store.close()

    from fastapi.testclient import TestClient

    settings = Settings(_env_file=None, database_path=database, registry_path=registry_path)
    with TestClient(create_app(provider=provider, settings=settings)) as client:
        execution = client.get(f"/api/requests/{record.id}").json()["executions"][0]
    assert (execution["status"], execution["error_code"]) == ("failed", "interrupted")


# ---- batches ----


def make_batch(client, **body):
    body.setdefault("prompts", ["alpha", "beta", "gamma"])
    response = client.post("/api/batches", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def test_batch_runs_every_prompt_on_every_model(client, provider):
    created = make_batch(client, name="trial")
    assert created["name"] == "trial"
    assert created["models"] == ["small:1b", "thinker:8b"]  # all installed
    assert created["request_count"] == 3

    batch = wait_batch(client, created["id"])
    assert batch["counts"] == {"queued": 0, "running": 0, "completed": 6, "failed": 0}
    assert [r["prompt"] for r in batch["requests"]] == ["alpha", "beta", "gamma"]
    for request in batch["requests"]:
        assert [e["model"] for e in request["executions"]] == ["small:1b", "thinker:8b"]
        assert request["executions"][0]["response"] == f"echo: {request['prompt']}"


def test_batch_groups_work_by_model_with_loaded_model_first(client, provider):
    from llm_gateway.providers.base import LoadedModel

    provider.loaded = [LoadedModel(name="thinker:8b", size_bytes=1, vram_bytes=1)]
    wait_batch(client, make_batch(client)["id"])
    assert provider.calls == ["thinker:8b"] * 3 + ["small:1b"] * 3


def test_batch_with_chosen_models_and_rich_prompts(client):
    created = make_batch(
        client,
        prompts=["plain", {"prompt": "rich", "system": "Be brief", "source_material": "doc"}],
        models=["small:1b"],
        think=False,
        max_tokens=50,
    )
    batch = wait_batch(client, created["id"])
    rich = batch["requests"][1]
    assert (rich["system"], rich["source_material"]) == ("Be brief", "doc")
    assert (rich["think"], rich["max_tokens"]) == (False, 50)
    assert batch["counts"]["completed"] == 2


def test_batch_survives_a_failing_model(client):
    created = make_batch(client, models=["small:1b", "missing:1b"])
    batch = wait_batch(client, created["id"])
    assert batch["counts"]["completed"] == 3
    assert batch["counts"]["failed"] == 3


def test_batch_needs_provider_only_when_models_are_omitted(client, provider):
    provider.error = ProviderUnavailable("down")
    assert client.post("/api/batches", json={"prompts": ["a"]}).status_code == 503
    created = make_batch(client, prompts=["a"], models=["small:1b"])
    assert wait_batch(client, created["id"])["counts"]["failed"] == 1


def test_batch_requests_are_kept_out_of_plain_history(client):
    wait_batch(client, make_batch(client)["id"])
    assert client.get("/api/requests").json() == []
    assert len(client.get("/api/requests", params={"include_batches": True}).json()) == 3
    assert len(client.get("/api/batches").json()) == 1


# ---- export and evaluation import ----


def test_blind_export_hides_model_and_thinking(client):
    batch = wait_batch(client, make_batch(client, models=["small:1b", "missing:1b"])["id"])
    export = client.get(f"/api/batches/{batch['id']}/export").json()

    assert export["blind"] is True
    assert len(export["items"]) == 3  # failed executions are left out
    assert all(item["model"] is None for item in export["items"])
    assert "thinking" not in export["items"][0]
    assert export["rubric"]["dimensions"][0]["name"] == "correctness"
    assert "independent evaluator" in export["instructions"]

    named = client.get(f"/api/batches/{batch['id']}/export", params={"blind": False}).json()
    assert {item["model"] for item in named["items"]} == {"small:1b"}


def test_export_download_sets_filename(client):
    batch = wait_batch(client, make_batch(client)["id"])
    response = client.get(f"/api/batches/{batch['id']}/export", params={"download": True})
    assert "attachment" in response.headers["content-disposition"]


SCORES = {"correctness": 8, "relevance": 10, "completeness": 6, "instruction_adherence": 8}


def test_evaluations_round_trip_and_overall_is_computed(client):
    batch = wait_batch(client, make_batch(client, prompts=["alpha"])["id"])
    items = client.get(f"/api/batches/{batch['id']}/export").json()["items"]
    payload = {
        "evaluations": [
            {
                "execution_id": items[0]["execution_id"],
                "judge": "claude-code",
                "scores": SCORES,
                "summary": "Fine.",
                "strengths": ["clear"],
                "weaknesses": ["thin"],
            },
            {
                "execution_id": items[1]["execution_id"],
                "judge": "claude-code",
                "scores": SCORES | {"groundedness": 3},
            },
        ]
    }
    assert client.post("/api/evaluations", json=payload).json() == {"stored": 2}

    detail = client.get(f"/api/batches/{batch['id']}").json()
    assert detail["evaluated_count"] == 2
    by_id = {e["id"]: e for e in detail["requests"][0]["executions"]}
    first = by_id[items[0]["execution_id"]]["evaluations"][0]
    assert first["overall_score"] == 8.0  # groundedness not applicable: mean of four
    assert first["scores"]["groundedness"] is None
    assert (first["judge"], first["strengths"], first["rubric_version"]) == (
        "claude-code",
        ["clear"],
        "v1",
    )
    second = by_id[items[1]["execution_id"]]["evaluations"][0]
    assert second["overall_score"] == 7.0  # mean of five

    # A second judge adds a record; nothing is overwritten.
    payload["evaluations"] = [payload["evaluations"][0] | {"judge": "human"}]
    client.post("/api/evaluations", json=payload)
    detail = client.get(f"/api/batches/{batch['id']}").json()
    by_id = {e["id"]: e for e in detail["requests"][0]["executions"]}
    assert [v["judge"] for v in by_id[items[0]["execution_id"]]["evaluations"]] == [
        "claude-code",
        "human",
    ]


@pytest.mark.parametrize(
    "scores", [SCORES | {"correctness": 11}, {"correctness": 5}, SCORES | {"relevance": -1}]
)
def test_evaluation_scores_are_validated(client, scores):
    batch = wait_batch(client, make_batch(client, prompts=["alpha"])["id"])
    execution_id = batch["requests"][0]["executions"][0]["id"]
    payload = {"evaluations": [{"execution_id": execution_id, "judge": "j", "scores": scores}]}
    assert client.post("/api/evaluations", json=payload).status_code == 422


def test_evaluation_import_is_all_or_nothing(client):
    batch = wait_batch(client, make_batch(client, prompts=["alpha"])["id"])
    good = batch["requests"][0]["executions"][0]["id"]
    payload = {
        "evaluations": [
            {"execution_id": good, "judge": "j", "scores": SCORES},
            {"execution_id": "does-not-exist", "judge": "j", "scores": SCORES},
        ]
    }
    response = client.post("/api/evaluations", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"]["execution_ids"] == ["does-not-exist"]
    assert client.get(f"/api/batches/{batch['id']}").json()["evaluated_count"] == 0


def test_running_execution_shows_the_text_generated_so_far(client, provider):
    from llm_gateway.providers.base import GenerationResult

    release = []

    async def generate(request, model, on_delta=None):
        import asyncio

        on_delta("thinking", "let me see")
        on_delta("content", "Par")
        while not release:
            await asyncio.sleep(0.01)
        on_delta("content", "is.")
        return GenerationResult(model=model.name, response="Paris.", num_ctx=model.num_ctx)

    provider.generate = generate
    created = submit(client)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        execution = client.get(f"/api/requests/{created['id']}").json()["executions"][0]
        if execution["response"]:
            break
        time.sleep(0.01)
    assert execution["status"] == "running"
    assert execution["response"] == "Par" and execution["thinking"] == "let me see"
    # The partial text is not in the database: History lists show it only once complete.
    assert client.get("/api/requests").json()[0]["executions"][0]["response"] is None

    release.append(True)
    done = wait_request(client, created["id"])["executions"][0]
    assert done["status"] == "completed" and done["response"] == "Paris."
    assert done["thinking"] is None  # the final result decides, not the partial text


def exhausted_provider(provider):
    """Make thinking runs use the whole context on reasoning and answer nothing."""
    from llm_gateway.providers.base import GenerationResult

    async def generate(request, model, on_delta=None):
        provider.requests.append(request)
        if request.think is False:
            return GenerationResult(model=model.name, response="Paris.", num_ctx=model.num_ctx)
        return GenerationResult(
            model=model.name,
            response="",
            thinking="1. The 2. capital 3. of",
            done_reason="length",
            num_ctx=model.num_ctx,
            output_tokens=8167,
            warnings=["output_cut_off", "context_window_full"],
        )

    provider.generate = generate


def test_empty_answer_after_thinking_is_a_failure_and_can_be_retried(client, provider):
    exhausted_provider(provider)
    first = wait_request(client, submit(client, prompt="Capital?", system="Be brief.")["id"])
    execution = first["executions"][0]
    assert execution["status"] == "failed" and execution["error_code"] == "thinking_exhausted"
    assert execution["error_detail"].startswith("Ran out of room while thinking")
    assert "8167" in execution["error_detail"]
    # What the model did produce is kept for inspection.
    assert execution["thinking"] == "1. The 2. capital 3. of"
    assert execution["metrics"]["output_tokens"] == 8167
    assert execution["warnings"] == ["output_cut_off", "context_window_full"]
    assert execution["execution_ms"] is not None

    response = client.post(f"/api/executions/{execution['id']}/retry", json={})
    assert response.status_code == 201
    retry = wait_request(client, response.json()["id"])
    # The second attempt joins the same run, after the failed one.
    assert retry["id"] == first["id"] and retry["think"] is None
    failed, second = retry["executions"]
    assert failed["status"] == "failed" and failed["think"] is None
    assert second["model"] == execution["model"] and second["think"] is False
    assert second["status"] == "completed" and second["response"] == "Paris."
    assert provider.requests[-1].think is False and provider.requests[-1].system == "Be brief."
    assert len(client.get("/api/requests").json()) == 1  # no extra run in History

    # A judge is given the answer that succeeded, not the failure.
    package = client.post("/api/judge-package", json={"request_ids": [first["id"]]}).json()
    assert package["item_count"] == 1 and "Paris." in package["text"]

    assert client.post("/api/executions/nope/retry", json={}).status_code == 404


def test_an_ordinary_empty_answer_stays_completed(client, provider):
    from llm_gateway.providers.base import GenerationResult

    async def generate(request, model, on_delta=None):
        return GenerationResult(model=model.name, response="", num_ctx=model.num_ctx)

    provider.generate = generate
    done = wait_request(client, submit(client)["id"])
    assert done["executions"][0]["status"] == "completed"


def test_retrying_a_follow_up_keeps_its_place_in_the_conversation(client, provider):
    root = wait_request(client, submit(client, think=False)["id"])["executions"][0]
    exhausted_provider(provider)
    follow = client.post(
        f"/api/executions/{root['id']}/follow-ups", json={"prompt": "And Spain?", "think": True}
    ).json()
    failed = wait_request(client, follow["id"])["executions"][0]
    assert failed["error_code"] == "thinking_exhausted"

    retry = client.post(f"/api/executions/{failed['id']}/retry", json={}).json()
    wait_request(client, retry["id"])
    turns = client.get(f"/api/executions/{root['id']}/thread").json()["turns"]
    assert [t["prompt"] for t in turns] == ["hello", "And Spain?"]
    assert turns[-1]["execution"]["response"] == "Paris."
    assert [m.content for m in provider.requests[-1].history] == ["hello", "echo: hello"]


def test_retry_inside_a_batch_stays_in_the_batch_and_its_export(client, provider):
    exhausted_provider(provider)
    batch = wait_batch(client, make_batch(client, prompts=["alpha"], models=["small:1b"])["id"])
    assert batch["counts"]["failed"] == 1
    failed = batch["requests"][0]["executions"][0]
    assert client.get(f"/api/batches/{batch['id']}/export").json()["items"] == []

    client.post(f"/api/executions/{failed['id']}/retry", json={})
    after = wait_batch(client, batch["id"])
    assert after["request_count"] == 1
    assert after["counts"] == {"queued": 0, "running": 0, "completed": 1, "failed": 1}
    items = client.get(f"/api/batches/{batch['id']}/export").json()["items"]
    assert [item["response"] for item in items] == ["Paris."]
