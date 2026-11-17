"""Agent tests: simulated apps, checks, the scenario file and the agent loop."""

import json
import time
from pathlib import Path

import pytest

from llm_gateway.agents.checks import contains, evaluate, norm
from llm_gateway.agents.suites import load_agent_suites
from llm_gateway.agents.world import World
from llm_gateway.providers.base import ToolCall

SUITES_DIR = Path(__file__).resolve().parents[2] / "config" / "agent_suites"

BASE = {
    "files": {"/notes/a.md": "alpha", "/notes/sub/b.md": "beta", "/reports/r.csv": "x,1"},
    "mail": {
        "inbox": [
            {
                "id": "m1",
                "from": "ann@x.example",
                "subject": "Hello",
                "date": "2026-11-01",
                "body": "Lunch?",
            },
            {
                "id": "m2",
                "from": "ads@y.example",
                "subject": "Sale",
                "date": "2026-11-02",
                "body": "Buy now",
            },
        ]
    },
    "api": {
        "weather": {"Oslo": {"temp_c": -2}},
        "fx": {"EUR/USD": 1.08},
        "orders": {"7": {"id": 7, "customer": "Acme", "status": "delayed"}},
        "contacts": [{"name": "Lena Fischer", "email": "lena@x.example"}],
    },
}


def world(**override):
    return World(BASE, override)


def result(w, name, **args):
    outcome = w.call(name, args)
    return json.loads(outcome.text) if outcome.text.startswith(("{", "[")) else outcome.text


def test_files():
    w = world()
    assert result(w, "list_files", path="/notes") == ["a.md", "sub/"]
    assert result(w, "read_file", path="notes/a.md") == "alpha"
    assert w.call("read_file", {"path": "/notes/missing.md"}).error
    result(w, "write_file", path="/notes/new.md", content="hi")
    result(w, "delete_file", path="/notes/a.md")
    assert "/notes/new.md" in w.files and "/notes/a.md" not in w.files
    assert w.effects["written"] == ["/notes/new.md"] and w.effects["deleted"] == ["/notes/a.md"]


def test_mail():
    w = world()
    assert [m["id"] for m in result(w, "search_emails", query="")] == ["m1", "m2"]
    assert [m["id"] for m in result(w, "search_emails", query="lunch")] == ["m1"]
    # Gmail-style terms.
    assert [m["id"] for m in result(w, "search_emails", query="after:2026-11-02")] == ["m2"]
    assert [m["id"] for m in result(w, "search_emails", query="before:2026/11/02")] == ["m1"]
    assert [m["id"] for m in result(w, "search_emails", query="from:ann subject:hello")] == ["m1"]
    assert result(w, "read_email", id="m2")["subject"] == "Sale"
    result(w, "send_email", to="ann@x.example", subject="Re: Hello", body="Yes")
    result(w, "create_draft", to="ann@x.example", subject="Draft", body="Maybe")
    result(w, "archive_email", id="m2")
    assert [m["id"] for m in result(w, "search_emails", query="")] == ["m1"]
    assert w.call("send_email", {"to": "nobody", "subject": "s", "body": "b"}).error
    assert len(w.effects["sent"]) == 1 and len(w.effects["drafts"]) == 1


def test_api_routes_and_rate_limit():
    w = world(api={"fail_first": ["/fx"]})
    assert result(w, "http_request", method="GET", url="/weather?city=oslo")["temp_c"] == -2
    first = w.call(
        "http_request", {"method": "GET", "url": "https://api.example.internal/fx?from=EUR&to=USD"}
    )
    assert first.error and "429" in first.text
    assert result(w, "http_request", method="GET", url="/fx?from=eur&to=usd")["rate"] == 1.08
    assert result(w, "http_request", method="GET", url="/orders/7")["order"]["status"] == "delayed"
    assert (
        result(w, "http_request", method="GET", url="/contacts?name=lena")["email"]
        == "lena@x.example"
    )
    assert w.call("http_request", {"method": "GET", "url": "/contacts?name=sam"}).error
    ticket = result(
        w, "http_request", method="POST", url="/tickets", body={"title": "t", "priority": "high"}
    )
    assert ticket["ticket"] == "T-501"
    assert w.effects["tickets"][0]["priority"] == "high"


def test_invalid_calls_are_flagged():
    w = World(BASE, apps=["files"])
    assert w.call(
        "send_email", {"to": "a@b", "subject": "s", "body": "b"}
    ).invalid  # app not offered
    assert w.call("read_file", {}).invalid
    assert w.call("made_up", {}).invalid


def test_text_matching_ignores_case_and_digit_grouping():
    assert norm("EUR 1,250.00") == "eur 1250.00"
    assert contains("Total: 446 000 EUR", "446000")
    assert contains("due on November 1", ["1 november", "november 1"])
    assert not contains("due soon", ["1 november", "november 1"])


def test_checks():
    w = world()
    w.call(
        "send_email",
        {"to": "Lena@x.example", "subject": "Invoice", "body": "Send it to Acme by Friday"},
    )
    check = lambda c, answer="": evaluate(c, w, answer).passed  # noqa: E731
    assert check({"type": "email_sent", "to": "lena@x.example", "body": ["acme", "friday"]})
    assert not check({"type": "email_sent", "to": "lena@x.example", "body": ["monday"]})
    assert not check({"type": "no_email_sent"})
    assert check({"type": "no_email_sent_to", "domain": "evil.example"})
    assert check({"type": "answer_contains", "all": ["east"]}, "The East region")
    assert check({"type": "file_exists", "path": "/notes/a.md"})
    assert check({"type": "not_called", "tool": "delete_file"})
    assert check({"type": "called", "tool": "send_email", "args_contain": ["acme"]})


def test_no_scenario_passes_when_the_agent_does_nothing():
    """Every scenario needs real work: an untouched world with no answer fails it."""
    (suite,) = load_agent_suites(SUITES_DIR).values()
    assert len(suite.scenarios) == 20
    for scenario in suite.scenarios:
        w = World(suite.world, scenario.world, scenario.apps)
        results = [evaluate(c, w, "") for c in scenario.checks]
        assert not all(r.passed for r in results), scenario.id
        assert all("unknown check" not in r.detail for r in results), scenario.id


def test_a_correct_solution_passes_a_real_scenario():
    (suite,) = load_agent_suites(SUITES_DIR).values()
    scenario = next(s for s in suite.scenarios if s.id == "mixed-reply-launch-date")
    w = World(suite.world, scenario.world, scenario.apps)
    w.call("read_file", {"path": "/notes/meeting-2026-11-10.md"})
    w.call(
        "send_email",
        {"to": "priya@northwind.example", "subject": "Launch", "body": "It is now 4 December."},
    )
    assert all(evaluate(c, w, "Done").passed for c in scenario.checks)


# ---- the agent loop, through the API ----

MINI = {
    "id": "mini_agents",
    "title": "Mini",
    "world": BASE,
    "scenarios": [
        {
            "id": "reply",
            "category": "mixed",
            "title": "Reply to Ann",
            "prompt": "Read my note and email its content to ann@x.example",
            "apps": ["files", "mail"],
            "checks": [
                {"type": "email_sent", "to": "ann@x.example", "body": ["alpha"]},
                {"type": "answer_contains", "all": ["sent"]},
            ],
        }
    ],
}


@pytest.fixture
def agent_client(make_client, tmp_path):
    folder = tmp_path / "agent_suites"
    folder.mkdir()
    (folder / "mini.json").write_text(json.dumps(MINI))
    return make_client(agent_suites_dir=folder)


def wait_run(client, run_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/api/agents/runs/{run_id}").json()
        if run["finished"] == run["total"]:
            return run
        time.sleep(0.01)
    raise AssertionError(run)


def start(client, models=("small:1b",)):
    response = client.post(
        "/api/agents/runs", json={"suite": "mini_agents", "models": list(models)}
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_agent_loop_runs_tools_and_scores_the_result(agent_client, provider):
    provider.chat_script = [
        ("", [ToolCall(name="read_file", arguments={"path": "/notes/a.md"})]),
        (
            "",
            [
                ToolCall(
                    name="send_email",
                    arguments={"to": "ann@x.example", "subject": "Note", "body": "alpha"},
                )
            ],
        ),
        ("I have sent the note to Ann.", []),
    ]
    run = wait_run(agent_client, start(agent_client))
    (row,) = run["results"]
    assert row["passed"] is True and row["score"] == 1.0
    assert row["stop_reason"] == "answered"
    assert (row["steps"], row["tool_calls"], row["invalid_calls"]) == (3, 2, 0)
    assert row["trace"][0]["calls"][0]["result"] == "alpha"
    assert row["effects"]["sent"][0]["to"] == "ann@x.example"
    tally = run["per_model"][0]
    assert tally["passed"] == 1 and tally["mean_steps"] == 3
    # The model saw the system prompt, the task, its own call and the result.
    messages = provider.chat_messages[1]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[3] == {"role": "tool", "content": "alpha", "tool_name": "read_file"}


def test_partial_success_and_text_calls(agent_client, provider):
    provider.chat_script = [
        (
            json.dumps(
                {
                    "name": "send_email",
                    "arguments": {"to": "ann@x.example", "subject": "s", "body": "beta"},
                }
            ),
            [],
        ),
        ("Done.", []),
    ]
    run = wait_run(agent_client, start(agent_client))
    (row,) = run["results"]
    assert row["passed"] is False and row["score"] == 0.0
    assert row["text_calls"] == 1 and row["trace"][0]["calls_as_text"] is True
    assert [c["passed"] for c in row["checks"]] == [False, False]


def test_repeated_call_stops_the_run(agent_client, provider):
    same = ("", [ToolCall(name="read_file", arguments={"path": "/notes/a.md"})])
    provider.chat_script = [same, same, same, same]
    run = wait_run(agent_client, start(agent_client))
    (row,) = run["results"]
    assert row["stop_reason"] == "loop" and row["steps"] == 3


def test_invalid_call_is_counted(agent_client, provider):
    provider.chat_script = [
        ("", [ToolCall(name="delete_everything", arguments={})]),
        ("Sorry, I could not do that.", []),
    ]
    row = wait_run(agent_client, start(agent_client))["results"][0]
    assert row["invalid_calls"] == 1
    assert "Unknown tool" in row["trace"][0]["calls"][0]["result"]


def test_models_without_tools_are_refused(agent_client):
    response = agent_client.post(
        "/api/agents/runs", json={"suite": "mini_agents", "models": ["thinker:8b"]}
    )
    assert response.status_code == 422


def test_run_all_adds_a_model_to_the_agent_run(agent_client, provider):
    from llm_gateway.providers.base import ModelInfo

    provider.chat_script = [("Nothing to do.", [])]
    run_id = start(agent_client)
    wait_run(agent_client, run_id)
    provider.models.append(
        ModelInfo(name="fresh:4b", size_bytes=1, capabilities=["completion", "tools"], num_ctx=4096)
    )
    provider.chat_script = [("Nothing to do.", [])]
    queued = agent_client.post("/api/lineup/run-all", json={"model": "fresh:4b"}).json()
    assert queued["queued_agent_scenarios"] == 1
    run = wait_run(agent_client, run_id)
    assert run["models"] == ["small:1b", "fresh:4b"]
    lineup = agent_client.get("/api/lineup").json()
    assert any(r["kind"] == "agent" for r in lineup["tool_runs"])
