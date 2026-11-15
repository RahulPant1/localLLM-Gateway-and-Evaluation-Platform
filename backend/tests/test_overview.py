"""The Overview summary endpoint (fake provider, temp database)."""

from datetime import UTC, datetime

from test_runs import make_batch, submit, wait_batch, wait_request

from llm_gateway.overview import build_overview
from llm_gateway.providers.base import GenerationFailed
from llm_gateway.schemas import Execution, RequestRecord


def scores(value):
    return dict.fromkeys(
        ["correctness", "relevance", "completeness", "instruction_adherence"], value
    )


def judge(client, execution_id, value, **extra):
    item = {"execution_id": execution_id, "judge": "claude-code", "scores": scores(value), **extra}
    assert client.post("/api/evaluations", json={"evaluations": [item]}).status_code == 201


def test_empty_database_gives_zeros_not_errors(client):
    data = client.get("/api/overview").json()
    assert data["requests"] == 0 and data["days"] is None
    assert data["routing"]["by_model"] == [] and data["routing"]["mean_confidence"] is None
    performance = data["performance"]
    assert performance["executions"] == 0 and performance["warm_rate"] is None
    assert performance["cloud_estimate"]["usd"] == 0
    assert data["by_model"] == [] and data["by_category"] == []
    assert data["evaluation"]["by_model"] == []
    assert data["evaluation"]["router_check"]["prompts"] == 0


def test_counts_completed_and_failed_per_model(client, provider):
    wait_request(client, submit(client, models=["small:1b", "thinker:8b"])["id"])
    provider.error = GenerationFailed("boom")
    wait_request(client, submit(client, models=["small:1b"])["id"])

    data = client.get("/api/overview?days=7").json()
    assert data["requests"] == 2 and data["days"] == 7
    assert data["performance"]["executions"] == 3
    assert data["performance"]["completed"] == 2 and data["performance"]["failed"] == 1
    rows = {row["key"]: row for row in data["by_model"]}
    assert rows["small:1b"]["executions"] == 2 and rows["small:1b"]["failed"] == 1
    assert rows["thinker:8b"]["completed"] == 1 and rows["thinker:8b"]["median_ms"] is not None
    # Manual requests carry no routing decision.
    assert [row["key"] for row in data["by_category"]] == ["not_routed"]
    assert data["routing"]["live"] == 0


def test_routing_counts_live_and_shadow_decisions(client):
    created = client.post("/api/requests", json={"prompt": "What is the capital of France?"})
    wait_request(client, created.json()["id"])
    wait_batch(client, make_batch(client, prompts=["alpha", "beta"])["id"])

    routing = client.get("/api/overview").json()["routing"]
    assert routing["live"] == 1 and routing["shadow"] == 2
    assert sum(share["count"] for share in routing["by_model"]) == 3
    assert abs(sum(share["share"] for share in routing["by_model"]) - 1) < 0.01
    assert sum(routing["decided_by"].values()) == 3
    assert 0 < routing["mean_confidence"] <= 1


def test_scores_and_router_check(client):
    batch = wait_batch(client, make_batch(client, prompts=["alpha", "beta"])["id"])
    first, second = batch["requests"]
    picks = [r["routing"]["selected_model"] for r in batch["requests"]]
    assert picks == ["small:1b", "small:1b"]
    by_model = lambda r: {e["model"]: e["id"] for e in r["executions"]}  # noqa: E731
    # Prompt 1: the router's pick scores best. Prompt 2: the other model does.
    judge(client, by_model(first)["small:1b"], 9)
    judge(client, by_model(first)["thinker:8b"], 6, critical_issues=["made up a fact"])
    judge(client, by_model(second)["small:1b"], 5)

    evaluation = client.get("/api/overview").json()["evaluation"]
    assert evaluation["scored"] == 3 and evaluation["unscored"] == 1
    assert evaluation["judges"] == ["claude-code"]
    rows = {row["model"]: row for row in evaluation["by_model"]}
    assert rows["small:1b"]["overall"] == 7.0 and rows["small:1b"]["evaluated"] == 2
    assert rows["thinker:8b"]["critical_issues"] == 1
    assert rows["thinker:8b"]["dimensions"]["groundedness"] is None
    # Only prompt 1 has the pick and another model both scored.
    check = evaluation["router_check"]
    assert check["prompts"] == 1 and check["picked_top"] == 1

    judge(client, by_model(second)["thinker:8b"], 8)
    check = client.get("/api/overview").json()["evaluation"]["router_check"]
    assert check == {
        "prompts": 2,
        "picked_top": 1,
        "router_mean": 7.0,
        "best_fixed_model": "small:1b",
        "best_fixed_mean": 7.0,
    }


def test_days_filter_and_validation(client):
    wait_request(client, submit(client)["id"])
    assert client.get("/api/overview?days=1").json()["requests"] == 1
    assert client.get("/api/overview?days=0").status_code == 422


def test_performance_numbers_from_provider_metrics():
    def execution(n, load_ms, generation_ms, tokens, status="completed"):
        return Execution(
            id=f"e{n}",
            request_id="r",
            model="m",
            status=status,
            created_at=datetime.now(UTC),
            queue_wait_ms=100.0 * n,
            execution_ms=load_ms + generation_ms,
            metrics={
                "load_ms": load_ms,
                "generation_ms": generation_ms,
                "prompt_tokens": 100,
                "output_tokens": tokens,
            },
        )

    request = RequestRecord(
        id="r",
        prompt="p",
        routing_mode="manual",
        created_at=datetime.now(UTC),
        executions=[
            execution(1, 6000, 4000, 200),  # swap
            execution(2, 100, 2000, 100),  # warm
            execution(3, 150, 2000, 100),  # warm
            execution(4, 0, 0, 0, status="failed"),
        ],
    )
    data = build_overview([request], pricing=("cloud-x", 1.0, 10.0))
    performance = data.performance
    assert performance.output_tokens == 400 and performance.prompt_tokens == 300
    assert performance.generation_seconds == 8.0
    assert performance.tokens_per_second == 50.0
    assert performance.warm_rate == 0.667
    assert performance.mean_swap_ms == 6000
    assert performance.mean_queue_wait_ms == 250
    assert performance.cloud_estimate.usd == 0.0043  # (300 * 1 + 400 * 10) / 1e6
    row = data.by_model[0]
    assert row.median_ms == 2150 and row.slowest_ms == 10000 and row.failed == 1

    assert build_overview([request]).performance.cloud_estimate is None


def test_scores_by_task_type_and_domain(client):
    batch = wait_batch(client, make_batch(client, prompts=["alpha", "beta", "gamma"])["id"])
    ids = [{e["model"]: e["id"] for e in r["executions"]} for r in batch["requests"]]
    # Two coding prompts and one that the judge labelled loosely.
    judge(client, ids[0]["small:1b"], 9, task_type="coding", domain="general")
    judge(client, ids[0]["thinker:8b"], 5, task_type="coding", domain="general")
    judge(client, ids[1]["small:1b"], 7, task_type="Coding", domain="science-technology")
    judge(client, ids[1]["thinker:8b"], 7, task_type="coding", domain="science_technology")
    judge(client, ids[2]["small:1b"], 4, task_type="poetry slam", domain="medical")
    judge(client, ids[2]["thinker:8b"], 8)  # no labels: an older or partial reply

    stored = client.get(f"/api/batches/{batch['id']}").json()["requests"][2]["executions"]
    assert [e["evaluations"][0]["task_type"] for e in stored] == ["other", None]

    evaluation = client.get("/api/overview").json()["evaluation"]
    assert evaluation["models"] == ["small:1b", "thinker:8b"]
    tasks = {row["key"]: row for row in evaluation["by_task"]}
    assert set(tasks) == {"coding", "other"}
    coding = tasks["coding"]
    assert coding["label"] == "Coding" and coding["prompts"] == 2
    assert coding["models"]["small:1b"] == {"mean": 8.0, "answers": 2}
    assert coding["models"]["thinker:8b"] == {"mean": 6.0, "answers": 2}
    assert coding["best_model"] == "small:1b"
    # The prompt's label applies to all of its answers, labelled or not.
    assert tasks["other"]["best_model"] == "thinker:8b"
    domains = {row["key"]: row["prompts"] for row in evaluation["by_domain"]}
    assert domains == {"general": 1, "science_technology": 1, "medical": 1}


def test_unlabelled_scores_are_grouped_as_not_categorised(client):
    done = wait_request(client, submit(client)["id"])
    judge(client, done["executions"][0]["id"], 6)
    row = client.get("/api/overview").json()["evaluation"]["by_task"][0]
    assert row["key"] == "uncategorised" and row["label"] == "Not categorised"
    assert row["best_model"] is None


def test_judge_is_given_the_task_type_lists(client):
    done = wait_request(client, submit(client)["id"])
    text = client.post("/api/judge-package", json={"request_ids": [done["id"]]}).json()["text"]
    assert "`summarization`" in text and "`science_technology`" in text and '"task_type"' in text
    names = [t["name"] for t in client.get("/api/task-types").json()["task_types"]]
    assert "coding" in names and names[-1] == "other"
