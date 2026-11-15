"""Rubric, judge export and the evals script (the Claude Code judging route)."""

import importlib.util
import json
from pathlib import Path

import pytest
from test_runs import make_batch, wait_batch

from llm_gateway.config import PROJECT_ROOT, Settings
from llm_gateway.rubric import DIMENSIONS, load_judge_instructions, load_rubric
from llm_gateway.schemas import DimensionScores, overall_score

SCORES = {"correctness": 8, "relevance": 10, "completeness": 6, "instruction_adherence": 8}

spec = importlib.util.spec_from_file_location(
    "evals_script", Path(__file__).parents[1] / "scripts" / "evals.py"
)
evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evals)


# ---- rubric ----


def test_shipped_rubric_and_judge_prompt_are_valid():
    rubric = load_rubric(PROJECT_ROOT / "config" / "rubric.yaml")
    assert rubric.error is None
    assert rubric.version == "v1"
    assert [d.name for d in rubric.dimensions] == list(DIMENSIONS)
    assert all(d.description and d.anchors for d in rubric.dimensions)

    prompt = load_judge_instructions(PROJECT_ROOT / "config" / "judge_prompt.md")
    for name in DIMENSIONS:
        assert name in prompt
    assert "execution_id" in prompt


def test_weights_change_the_overall_score():
    scores = DimensionScores(**SCORES)
    equal = dict.fromkeys(DIMENSIONS, 1.0)
    assert overall_score(scores, equal) == 8.0
    assert overall_score(scores, equal | {"correctness": 3.0}) == 8.0  # correctness is 8 too
    assert overall_score(scores, equal | {"relevance": 3.0}) == 8.67
    assert overall_score(scores, equal | {"completeness": 0.0}) == 8.67
    # Groundedness only counts when it was scored.
    assert overall_score(DimensionScores(**SCORES, groundedness=3), equal) == 7.0


@pytest.mark.parametrize(
    "content",
    [
        "version: v2\ndimensions: [{name: correctness}]",
        "version: v2\ndimensions: [{name: correctness, weight: -1}]",
        "- just\n- a list",
        "version: [",
    ],
)
def test_broken_rubric_falls_back_to_equal_weights(tmp_path, content):
    path = tmp_path / "rubric.yaml"
    path.write_text(content)
    rubric = load_rubric(path)
    assert rubric.error and str(path) in rubric.error
    assert rubric.weights == dict.fromkeys(DIMENSIONS, 1.0)


def test_missing_judge_prompt_uses_a_short_default(tmp_path):
    assert "groundedness" in load_judge_instructions(tmp_path / "absent.md")


def custom_rubric(tmp_path, version="v7", relevance=3.0):
    path = tmp_path / "rubric.yaml"
    dimensions = "\n".join(
        f"  - {{name: {name}, weight: {relevance if name == 'relevance' else 1.0}}}"
        for name in DIMENSIONS
    )
    path.write_text(f"version: {version}\ndimensions:\n{dimensions}\n")
    return path


def test_api_uses_the_configured_rubric(make_client, tmp_path):
    client = make_client(rubric_path=custom_rubric(tmp_path))
    assert client.get("/api/rubric").json()["version"] == "v7"
    assert client.get("/api/status").json()["evaluator"]["rubric_version"] == "v7"

    batch = wait_batch(client, make_batch(client, prompts=["alpha"], models=["small:1b"])["id"])
    execution_id = batch["requests"][0]["executions"][0]["id"]
    payload = {"evaluations": [{"execution_id": execution_id, "judge": "j", "scores": SCORES}]}
    assert client.post("/api/evaluations", json=payload).status_code == 201

    stored = client.get(f"/api/batches/{batch['id']}").json()
    evaluation = stored["requests"][0]["executions"][0]["evaluations"][0]
    assert evaluation["rubric_version"] == "v7"  # filled in from the rubric
    assert evaluation["overall_score"] == 8.67  # relevance weighted x3

    # A judge may state the rubric version it used.
    payload["evaluations"][0]["rubric_version"] = "v1"
    client.post("/api/evaluations", json=payload)
    stored = client.get(f"/api/batches/{batch['id']}").json()
    versions = [v["rubric_version"] for v in stored["requests"][0]["executions"][0]["evaluations"]]
    assert versions == ["v7", "v1"]


# ---- export ----


def test_export_carries_rubric_and_instructions_and_can_skip_a_judge(client):
    batch = wait_batch(client, make_batch(client, prompts=["alpha", "beta"])["id"])
    export = client.get(f"/api/batches/{batch['id']}/export").json()
    assert len(export["items"]) == 4
    assert export["rubric"]["version"] == "v1"
    assert len(export["rubric"]["dimensions"]) == 5
    assert "Do not give an overall score" in export["instructions"]
    assert all(item["history"] == [] for item in export["items"])

    first = export["items"][0]["execution_id"]
    client.post(
        "/api/evaluations",
        json={"evaluations": [{"execution_id": first, "judge": "claude-code", "scores": SCORES}]},
    )
    remaining = client.get(
        f"/api/batches/{batch['id']}/export", params={"exclude_judge": "claude-code"}
    ).json()
    assert len(remaining["items"]) == 3
    assert first not in [item["execution_id"] for item in remaining["items"]]
    # Another judge still gets everything.
    other = client.get(f"/api/batches/{batch['id']}/export", params={"exclude_judge": "human"})
    assert len(other.json()["items"]) == 4


def test_scores_endpoint_gives_means_per_model_and_judge(client):
    batch = wait_batch(client, make_batch(client, prompts=["alpha", "beta"])["id"])
    evaluations = []
    for request in batch["requests"]:
        for execution in request["executions"]:
            good = execution["model"] == "thinker:8b"
            scores = SCORES | {"correctness": 10 if good else 4}
            evaluations.append({"execution_id": execution["id"], "judge": "j", "scores": scores})
    evaluations[0]["critical_issues"] = ["invented a fact"]
    client.post("/api/evaluations", json={"evaluations": evaluations})

    rows = {r["model"]: r for r in client.get(f"/api/batches/{batch['id']}/scores").json()}
    assert rows["thinker:8b"]["dimensions"]["correctness"] == 10
    assert rows["small:1b"]["dimensions"]["correctness"] == 4
    assert rows["thinker:8b"]["overall"] > rows["small:1b"]["overall"]
    assert rows["small:1b"]["evaluated"] == 2
    assert rows["small:1b"]["dimensions"]["groundedness"] is None
    assert rows["small:1b"]["critical_issues"] == 1
    assert client.get("/api/batches/nope/scores").status_code == 404


def package(client, **body):
    return client.post("/api/judge-package", json=body)


def test_judge_package_is_one_self_contained_text(client):
    batch = wait_batch(client, make_batch(client, name="Pack", prompts=["alpha", "beta"])["id"])
    result = package(client, batch_ids=[batch["id"]]).json()
    text = result["text"]

    assert (result["item_count"], result["rubric_version"], result["blind"]) == (4, "v1", True)
    assert text.startswith("# Instructions for the judge")
    assert "# Rubric (version v1)" in text and "## instruction_adherence (weight 1)" in text
    assert "9-10: Fully correct" in text
    assert '# Items to judge (4, from batch "Pack")' in text
    assert "small:1b" not in text and "thinker:8b" not in text  # blind

    # The items block is valid JSON carrying every execution id.
    items = json.loads(text.split("```json\n")[-1].split("\n```")[0])
    ids = {e["id"] for r in batch["requests"] for e in r["executions"]}
    assert {item["execution_id"] for item in items} == ids
    assert set(items[0]) == {"execution_id", "prompt", "response"}

    named = package(client, batch_ids=[batch["id"]], blind=False).json()
    assert "thinker:8b" in named["text"]


def test_judge_package_from_history_runs_and_batches_together(client):
    from test_runs import submit, wait_request

    compare = wait_request(
        client, submit(client, prompt="one", models=["small:1b", "thinker:8b"])["id"]
    )
    single = wait_request(client, submit(client, prompt="two")["id"])
    batch = wait_batch(client, make_batch(client, name="Mix", prompts=["three"])["id"])

    result = package(
        client, request_ids=[compare["id"], single["id"], compare["id"]], batch_ids=[batch["id"]]
    ).json()
    assert result["item_count"] == 5  # 2 + 1 + 2; the repeated id counts once
    assert '(5, from batch "Mix" and 2 history runs)' in result["text"]

    only_history = package(client, request_ids=[single["id"]]).json()
    assert only_history["item_count"] == 1
    assert "(1, from 1 history run)" in only_history["text"]

    # Scores for a history run are stored and shown on that run.
    execution_id = compare["executions"][0]["id"]
    client.post(
        "/api/evaluations",
        json={
            "evaluations": [
                {"execution_id": execution_id, "judge": "claude-code", "scores": SCORES}
            ]
        },
    )
    stored = client.get(f"/api/requests/{compare['id']}").json()
    assert stored["executions"][0]["evaluations"][0]["overall_score"] == 8.0
    remaining = package(client, request_ids=[compare["id"]], exclude_judge="claude-code").json()
    assert remaining["item_count"] == 1


def test_judge_package_rejects_unknown_or_empty_selection(client):
    assert package(client).status_code == 422
    response = package(client, request_ids=["ghost"], batch_ids=["phantom"])
    assert response.status_code == 404
    assert response.json()["detail"]["ids"] == ["ghost", "phantom"]


def test_judge_prompt_endpoint(client):
    assert "independent evaluator" in client.get("/api/judge-prompt").json()["instructions"]


# ---- the evals script, as Claude Code uses it ----


@pytest.fixture
def workspace(client, tmp_path, registry_path):
    """A finished batch, plus settings pointing the script at the same database."""
    batch = wait_batch(
        client, make_batch(client, name="Trial run", prompts=["alpha", "beta"])["id"]
    )
    settings = Settings(
        _env_file=None, database_path=tmp_path / "test.db", registry_path=registry_path
    )
    return batch, settings


def run(settings, capsys, *argv):
    code = evals.main(list(argv), settings=settings)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_script_export_judge_import_summary_round_trip(workspace, client, tmp_path, capsys):
    batch, settings = workspace
    code, out, _ = run(settings, capsys, "list")
    assert code == 0 and "'Trial run'" in out and "2 prompts x 2 models" in out

    exported = tmp_path / "batch.json"
    code, out, _ = run(
        settings, capsys, "export", "trial run", "--judge", "claude-code", "--out", str(exported)
    )
    assert code == 0 and "Wrote 4 items" in out
    data = json.loads(exported.read_text())
    assert data["blind"] is True
    assert all("model" not in item and "thinking" not in item for item in data["items"])
    assert data["rubric"]["version"] == "v1" and data["instructions"]

    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "evaluations": [
                    {"execution_id": item["execution_id"], "judge": "claude-code", "scores": SCORES}
                    for item in data["items"][:3]
                ]
            }
        )
    )
    code, out, _ = run(settings, capsys, "import", str(results))
    assert code == 0 and "Stored 3 evaluations (rubric v1)" in out

    # Resuming exports only what this judge has not scored.
    code, out, _ = run(
        settings,
        capsys,
        "export",
        batch["id"][:8],
        "--judge",
        "claude-code",
        "--out",
        str(exported),
    )
    assert "Wrote 1 items" in out

    code, out, _ = run(settings, capsys, "summary", "Trial")
    assert code == 0 and "small:1b" in out and "thinker:8b" in out and "8.00" in out
    # The running app sees the scores the script stored.
    assert client.get(f"/api/batches/{batch['id']}").json()["evaluated_count"] == 3


def test_script_named_export_and_printing(workspace, capsys):
    _, settings = workspace
    code, out, _ = run(settings, capsys, "export", "Trial run", "--named")
    assert code == 0
    assert {item["model"] for item in json.loads(out)["items"]} == {"small:1b", "thinker:8b"}


def test_script_reports_problems_without_storing_anything(workspace, client, tmp_path, capsys):
    batch, settings = workspace
    code, _, err = run(settings, capsys, "export", "no such batch")
    assert code == 1 and "No batch matches" in err

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    code, _, err = run(settings, capsys, "import", str(bad))
    assert code == 1 and "not valid JSON" in err

    execution_id = batch["requests"][0]["executions"][0]["id"]
    bad.write_text(
        json.dumps(
            {
                "evaluations": [
                    {"execution_id": execution_id, "judge": "j", "scores": SCORES},
                    {"execution_id": execution_id, "judge": "j", "scores": {"correctness": 12}},
                ]
            }
        )
    )
    code, _, err = run(settings, capsys, "import", str(bad))
    assert code == 1 and "evaluations.1.scores.correctness" in err

    bad.write_text(
        json.dumps({"evaluations": [{"execution_id": "ghost", "judge": "j", "scores": SCORES}]})
    )
    code, _, err = run(settings, capsys, "import", str(bad))
    assert code == 1 and "Unknown execution ids: ghost" in err
    assert client.get(f"/api/batches/{batch['id']}").json()["evaluated_count"] == 0

    code, out, _ = run(settings, capsys, "summary", "Trial run")
    assert code == 0 and "no evaluations yet" in out
