"""Follow-up chat: linked turns, history sent to the model, context limits."""

import sqlite3

from test_runs import submit, wait_request

from llm_gateway.providers.base import GenerationFailed, GenerationResult
from llm_gateway.store import MIGRATIONS, Store


def follow_up(client, execution_id, prompt="and then?", **extra):
    return client.post(
        f"/api/executions/{execution_id}/follow-ups", json={"prompt": prompt, **extra}
    )


def first_turn(client, **body):
    done = wait_request(client, submit(client, **body)["id"])
    return done, done["executions"][0]


def capture(provider, prompt_tokens=None, output_tokens=None, thinking=None):
    """Make the fake provider record what it is sent and report token counts."""
    seen = []

    async def generate(request, model):
        seen.append(request)
        return GenerationResult(
            model=model.name,
            response=f"answer to {request.prompt}",
            thinking=thinking,
            num_ctx=model.num_ctx,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    provider.generate = generate
    return seen


def test_follow_up_sends_the_whole_conversation(client, provider):
    seen = capture(provider, thinking="private reasoning")
    _, first = first_turn(
        client, prompt="Summarise.", system="Be brief", source_material="The doc.", think=False
    )
    second = wait_request(client, follow_up(client, first["id"], "Shorter.").json()["id"])
    third_response = follow_up(client, second["executions"][0]["id"], "In French.")
    assert third_response.status_code == 201
    wait_request(client, third_response.json()["id"])

    sent = seen[-1]
    assert sent.prompt == "In French."
    assert sent.system == "Be brief"  # inherited from the first turn
    assert sent.think is False
    assert [(m.role, m.content) for m in sent.history] == [
        ("user", "Summarise.\n\nSource material:\nThe doc."),
        ("assistant", "answer to Summarise.\n\nSource material:\nThe doc."),
        ("user", "Shorter."),
        ("assistant", "answer to Shorter."),
    ]
    assert "private reasoning" not in str(sent.history)
    assert second["parent_execution_id"] == first["id"]
    assert second["executions"][0]["model"] == first["model"]


def test_thread_can_be_opened_from_any_turn(client, provider):
    capture(provider)
    _, first = first_turn(client, prompt="one")
    second = wait_request(client, follow_up(client, first["id"], "two").json()["id"])
    third = wait_request(
        client, follow_up(client, second["executions"][0]["id"], "three").json()["id"]
    )

    for execution_id in (first["id"], second["executions"][0]["id"], third["executions"][0]["id"]):
        thread = client.get(f"/api/executions/{execution_id}/thread").json()
        assert [t["prompt"] for t in thread["turns"]] == ["one", "two", "three"]
        assert thread["model"] == "small:1b"
        assert thread["reply_to"] == third["executions"][0]["id"]
    assert client.get("/api/executions/nope/thread").status_code == 404


def test_history_lists_a_conversation_once(client, provider):
    capture(provider)
    request, first = first_turn(client)
    wait_request(client, follow_up(client, first["id"]).json()["id"])

    listed = client.get("/api/requests").json()
    assert [r["id"] for r in listed] == [request["id"]]
    assert listed[0]["executions"][0]["follow_up_count"] == 1


def test_each_compared_model_can_be_continued_separately(client, provider):
    seen = capture(provider)
    done = wait_request(client, submit(client, models=["small:1b", "thinker:8b"])["id"])
    thinker = next(e for e in done["executions"] if e["model"] == "thinker:8b")
    wait_request(client, follow_up(client, thinker["id"]).json()["id"])

    assert seen[-1].model == "thinker:8b"
    assert [m.content for m in seen[-1].history] == ["hello", "answer to hello"]


def test_cannot_continue_an_unfinished_or_failed_answer(client, provider):
    provider.generate = _failing()
    _, failed = first_turn(client)
    response = follow_up(client, failed["id"])
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "parent_not_completed"
    assert follow_up(client, "nope").status_code == 404
    assert follow_up(client, failed["id"], prompt="").status_code == 422


def _failing():
    async def generate(request, model):
        raise GenerationFailed("out of memory")

    return generate


def test_after_a_failed_turn_the_reply_continues_from_the_last_good_answer(client, provider):
    seen = capture(provider)
    _, first = first_turn(client, prompt="one")
    provider.generate = _failing()
    wait_request(client, follow_up(client, first["id"], "two").json()["id"])

    thread = client.get(f"/api/executions/{first['id']}/thread").json()
    assert [t["execution"]["status"] for t in thread["turns"]] == ["completed", "failed"]
    assert thread["reply_to"] == first["id"]

    seen = capture(provider)
    wait_request(client, follow_up(client, thread["reply_to"], "two again").json()["id"])
    assert [m.content for m in seen[-1].history] == ["one", "answer to one"]
    thread = client.get(f"/api/executions/{first['id']}/thread").json()
    assert [t["prompt"] for t in thread["turns"]] == ["one", "two again"]  # newest branch


def test_context_usage_counts_the_answer_but_not_the_thinking(client, provider):
    # 4096-token window. Input used 1000; output 2000 tokens, half of it thinking.
    capture(provider, prompt_tokens=1000, output_tokens=2000, thinking="x" * 17)
    _, first = first_turn(client, prompt="1234567")  # the answer is 17 characters long
    context = client.get(f"/api/executions/{first['id']}/thread").json()["context"]
    assert context == {
        "used_tokens": 2000,
        "limit_tokens": 4096,
        "fraction": 0.488,
        "state": "ok",
    }


def test_context_warning_then_refusal_when_full(client, provider):
    capture(provider, prompt_tokens=3000, output_tokens=0)  # 73% of 4096
    _, first = first_turn(client)
    assert client.get(f"/api/executions/{first['id']}/thread").json()["context"]["state"] == (
        "warning"
    )
    capture(provider, prompt_tokens=3900, output_tokens=0)  # 95%
    second = wait_request(client, follow_up(client, first["id"]).json()["id"])
    thread = client.get(f"/api/executions/{first['id']}/thread").json()
    assert thread["context"]["state"] == "full"

    response = follow_up(client, second["executions"][0]["id"])
    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "context_full"


def test_existing_database_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(f"{MIGRATIONS[0]}; PRAGMA user_version=1;")
    old.execute(
        "INSERT INTO requests (id, prompt, routing_mode, created_at)"
        " VALUES ('r1', 'kept', 'manual', '2026-11-15T00:00:00+00:00')"
    )
    old.commit()
    old.close()

    store = Store(path)
    record = store.get_request("r1")
    assert (record.prompt, record.parent_execution_id) == ("kept", None)
