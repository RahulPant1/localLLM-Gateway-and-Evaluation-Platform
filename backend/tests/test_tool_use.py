"""Tool-calling checker and test runs (fake provider, temp database)."""

import json
import time

from llm_gateway.providers.base import ToolCall
from llm_gateway.tooluse.checker import extract_text_calls, score

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "area",
            "parameters": {
                "type": "object",
                "properties": {"base": {"type": "integer"}, "unit": {"type": "string"}},
                "required": ["base"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    },
]
EXPECTED = [{"name": "area", "arguments": {"base": [10], "unit": ["units", ""]}}]


def call(name, **arguments):
    return ToolCall(name=name, arguments=arguments)


def check(calls, expected=EXPECTED):
    return score(expected, calls, TOOLS)


def test_correct_call_passes_and_optional_may_be_left_out():
    assert check([call("area", base=10)]).passed
    assert check([call("area", base=10, unit="Units")]).passed


def test_wrong_function_missing_and_unknown_arguments_fail():
    assert "expected area" in check([call("weather", city="x")]).reason
    assert "missing argument 'base'" in check([call("area")]).reason
    assert "unknown argument 'colour'" in check([call("area", base=10, colour="red")]).reason


def test_values_compare_by_meaning_but_types_are_strict():
    assert check([call("area", base=10.0)]).passed  # a whole number sent as float
    assert not check([call("area", base="10")]).passed  # a number sent as text
    assert not check([call("area", base=11)]).passed
    assert not check([call("area", base=True)]).passed


def test_no_call_and_wrong_count_fail():
    assert check([]).reason == "made no tool call"
    assert "made 2 calls" in check([call("area", base=10), call("area", base=10)]).reason


def test_parallel_calls_match_in_any_order():
    expected = [
        {"name": "weather", "arguments": {"city": ["Paris"]}},
        {"name": "weather", "arguments": {"city": ["Rome"]}},
    ]
    assert check([call("weather", city="Rome"), call("weather", city="paris")], expected).passed
    assert not check([call("weather", city="Rome"), call("weather", city="Rome")], expected).passed


def test_irrelevance_passes_only_without_a_call():
    assert check([], expected=[]).passed
    assert not check([call("weather", city="x")], expected=[]).passed


def test_nested_values_and_lists():
    expected = [{"name": "area", "arguments": {"base": [{"a": [1], "b": [[2, 3]]}]}}]
    assert check([call("area", base={"a": 1, "b": [2, 3]})], expected).passed
    assert not check([call("area", base={"a": 1, "b": [3, 2]})], expected).passed


def test_calls_written_as_text_are_recovered():
    found = extract_text_calls('Sure! {"name": "area", "arguments": {"base": 10}}')
    assert [(c.name, c.arguments) for c in found] == [("area", {"base": 10})]
    fenced = extract_text_calls('```json\n[{"name": "a", "parameters": {"x": 1}}]\n```')
    assert [(c.name, c.arguments) for c in fenced] == [("a", {"x": 1})]
    tagged = extract_text_calls('<tool_call>{"name": "a", "arguments": {}}</tool_call>')
    assert [c.name for c in tagged] == ["a"]
    assert extract_text_calls("The area is 50.") == []
    assert extract_text_calls('{"colour": "red"}') == []


# ---- runs ----

SUITE = {
    "id": "mini",
    "title": "Mini suite",
    "categories": {"simple": "one call", "irrelevance": "no call"},
    "cases": [
        {
            "id": "c1",
            "category": "simple",
            "prompt": "area of 10",
            "tools": TOOLS,
            "expected": EXPECTED,
        },
        {
            "id": "c2",
            "category": "irrelevance",
            "prompt": "tell a joke",
            "tools": TOOLS,
            "expected": [],
        },
    ],
}


def wait_run(client, run_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = client.get(f"/api/tool-use/runs/{run_id}").json()
        if run["finished"] == run["total"]:
            return run
        time.sleep(0.01)
    raise AssertionError(f"run did not finish: {run}")


def make_suite_client(make_client, tmp_path):
    suites = tmp_path / "suites"
    suites.mkdir()
    (suites / "mini.json").write_text(json.dumps(SUITE))
    return make_client(tool_suites_dir=suites)


def test_run_scores_each_case(make_client, provider, tmp_path):
    provider.tool_calls_for["area of 10"] = [call("area", base=10)]
    provider.tool_calls_for["tell a joke"] = [call("weather", city="x")]
    client = make_suite_client(make_client, tmp_path)

    suites = client.get("/api/tool-use/suites").json()
    assert suites[0]["case_count"] == 2
    created = client.post("/api/tool-use/runs", json={"suite": "mini", "models": ["small:1b"]})
    assert created.status_code == 201

    run = wait_run(client, created.json()["id"])
    by_case = {r["case_id"]: r for r in run["results"]}
    assert by_case["c1"]["passed"] is True
    assert by_case["c2"]["passed"] is False
    assert "although no function fits" in by_case["c2"]["reason"]
    tally = run["per_model"][0]
    assert (tally["passed"], tally["wrong"], tally["errors"]) == (1, 1, 0)
    assert tally["by_category"]["simple"]["passed"] == 1
    # The prompt is the only message: nothing is added that could help the model.
    assert provider.chat_messages[0] == [{"role": "user", "content": "area of 10"}]


def test_call_written_as_text_is_scored_but_marked_not_native(make_client, provider, tmp_path):
    provider.content_for["area of 10"] = '{"name": "area", "arguments": {"base": 10}}'
    client = make_suite_client(make_client, tmp_path)
    run_id = client.post(
        "/api/tool-use/runs", json={"suite": "mini", "models": ["small:1b"]}
    ).json()["id"]
    run = wait_run(client, run_id)
    by_case = {r["case_id"]: r for r in run["results"]}
    assert by_case["c1"]["passed"] is True
    assert by_case["c1"]["call_format"] == "text"
    assert by_case["c2"]["call_format"] == "none"
    tally = run["per_model"][0]
    assert (tally["passed"], tally["passed_native"]) == (2, 1)


def test_models_without_tool_support_are_refused(make_client, provider, tmp_path):
    from llm_gateway.providers.base import ModelInfo

    provider.models.append(
        ModelInfo(name="plain:4b", size_bytes=1, capabilities=["completion"], num_ctx=4096)
    )
    client = make_suite_client(make_client, tmp_path)
    response = client.post("/api/tool-use/runs", json={"suite": "mini", "models": ["plain:4b"]})
    assert response.status_code == 422
    assert "does not support tool calling" in response.json()["detail"]


def test_provider_failure_is_an_error_not_a_wrong_answer_and_can_resume(
    make_client, provider, tmp_path
):
    from llm_gateway.providers.base import GenerationTimeout

    client = make_suite_client(make_client, tmp_path)
    provider.error = None
    original_chat = provider.chat
    state = {"fail": True}

    async def flaky(*args, **kwargs):
        if state["fail"]:
            raise GenerationTimeout("too slow")
        return await original_chat(*args, **kwargs)

    provider.chat = flaky
    run_id = client.post(
        "/api/tool-use/runs", json={"suite": "mini", "models": ["small:1b"]}
    ).json()["id"]
    run = wait_run(client, run_id)
    assert run["per_model"][0]["errors"] == 2
    assert {r["error_code"] for r in run["results"]} == {"generation_timeout"}

    state["fail"] = False
    client.post(f"/api/tool-use/runs/{run_id}/resume")
    run = wait_run(client, run_id)
    assert run["per_model"][0]["errors"] == 0


def test_unknown_suite_and_model(make_client, tmp_path):
    client = make_suite_client(make_client, tmp_path)
    assert (
        client.post("/api/tool-use/runs", json={"suite": "x", "models": ["a"]}).status_code == 404
    )
    assert (
        client.post("/api/tool-use/runs", json={"suite": "mini", "models": ["nope:1b"]}).status_code
        == 404
    )
