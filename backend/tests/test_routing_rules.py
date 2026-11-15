"""The router's deterministic parts: quick rules, registry and policy. No model involved."""

from pathlib import Path

import pytest
import yaml

from llm_gateway.config import PROJECT_ROOT
from llm_gateway.providers.base import ModelInfo
from llm_gateway.routing import classifier, policy, signals
from llm_gateway.routing.registry import Registry, load_registry
from llm_gateway.routing.types import CATEGORIES

DATA = Path(__file__).parent / "data"
THRESHOLD = 0.4


def load(name):
    return yaml.safe_load((DATA / name).read_text())


def accuracy(items):
    reports = [(item, signals.analyze(item["prompt"])) for item in items]
    correct = sum(report.category == item["category"] for item, report in reports)
    sure_but_wrong = [
        item["prompt"]
        for item, report in reports
        if report.category != item["category"] and report.clarity >= THRESHOLD
    ]
    return correct / len(items), sure_but_wrong


def test_labelled_set_covers_every_category():
    for name in ("routing_prompts.yaml", "routing_prompts_heldout.yaml"):
        assert {item["category"] for item in load(name)} == set(CATEGORIES)


def test_rules_on_the_tuning_set():
    share, sure_but_wrong = accuracy(load("routing_prompts.yaml"))
    assert share >= 0.9
    # A wrong answer is tolerable when the rules know they are unsure (the
    # classifier model is asked). Being wrong and sure is the real failure.
    assert sure_but_wrong == []


def test_rules_on_the_held_out_set():
    """Prompts the rules were never tuned on. The floor is deliberately modest."""
    share, sure_but_wrong = accuracy(load("routing_prompts_heldout.yaml"))
    assert share >= 0.65
    assert len(sure_but_wrong) <= 3


@pytest.mark.parametrize(
    ("prompt", "category"),
    [
        ("Write a Python function that merges two sorted lists.", "coding"),
        ("What is the capital of Australia?", "general_qa"),
        ("Classify this review as positive, negative or neutral: 'meh'", "classification"),
        ("Extract all the dates from this text: 'Due 3 March, paid 5 March.'", "extraction"),
        ("Generate a JSON object describing a book.", "structured_output"),
        ("Analyse the trade-offs between renting and buying a home.", "reasoning"),
        ("Patient is a 45-year-old female presenting with fatigue. TSH 12.4.", "medical"),
    ],
)
def test_clear_examples(prompt, category):
    report = signals.analyze(prompt)
    assert report.category == category
    assert report.clarity >= THRESHOLD


def test_casual_health_question_is_not_medical():
    report = signals.analyze("Is it okay to take ibuprofen for a headache after drinking coffee?")
    assert report.category == "general_qa"


def test_source_material_counts_for_domain_rules():
    report = signals.analyze(
        "Summarise this for the family.",
        source_material="Discharge summary: 68-year-old male, troponin elevated, aspirin 75 mg.",
    )
    assert report.category == "medical"


def test_nothing_matching_means_no_clarity():
    report = signals.analyze("Blue green seven lamp.")
    assert (report.category, report.runner_up, report.clarity) == ("general_qa", None, 0.0)


def test_complexity_levels():
    assert signals.analyze("What is 2 + 2? One word.").complexity == "low"
    assert signals.analyze("Compare SQL and NoSQL trade-offs for a startup.").complexity == "medium"
    deep = (
        "Design a comprehensive, production-ready architecture step by step, covering edge cases."
    )
    assert signals.analyze(deep).complexity == "high"
    long_input = signals.analyze("Summarise this.", source_material="word " * 2000)
    assert long_input.complexity == "medium"
    assert long_input.estimated_tokens > 1500


# ---- registry ----


def test_shipped_registry_is_valid_and_covers_every_category():
    registry = load_registry(PROJECT_ROOT / "config" / "models.yaml")
    assert registry.error is None
    assert set(registry.policy) == set(CATEGORIES)
    assert registry.classifier_model in registry.models
    assert registry.groups_for("structured_output", "low")[0] == ["qwen3.5:4b", "qwen2.5-coder:7b"]
    assert registry.groups_for("general_qa", "high")[0] == ["deepseek-r1:8b"]
    assert registry.is_profiled("deepseek-r1:8b") and not registry.is_profiled("llama3.2:3b")
    assert "coding" in registry.categories_of("qwen2.5-coder:7b")


@pytest.mark.parametrize(
    "content",
    [
        "policy: [not, a, mapping]",
        "policy: {cooking: {default: [a]}}",
        "policy: {coding: {extreme: [a]}}",
        "models: [",
        "just a string",
    ],
)
def test_broken_registry_is_reported_not_fatal(tmp_path, content):
    path = tmp_path / "models.yaml"
    path.write_text(content)
    registry = load_registry(path)
    assert registry.error and str(path) in registry.error
    assert registry.policy == {}


def test_missing_registry_file_is_reported(tmp_path):
    assert load_registry(tmp_path / "absent.yaml").error


# ---- policy ----


def info(name, num_ctx=8192):
    return ModelInfo(name=name, size_bytes=1, num_ctx=num_ctx)


REGISTRY = Registry(
    policy={
        "coding": {"default": ["coder", "general"]},
        "structured_output": {"default": [["general", "coder"], "thinker"]},
        "reasoning": {"default": ["thinker", "general"], "low": ["general", "thinker"]},
        "medical": {"default": ["medic", "general"]},
    },
    last_resort=["general", "thinker"],
)
ALL = {name: info(name) for name in ("coder", "general", "thinker")} | {
    "medic": info("medic", 4096)
}


def choose(category, complexity="low", installed=ALL, loaded=(), tokens=50):
    return policy.select(
        REGISTRY,
        category,
        complexity,
        installed=installed,
        loaded=set(loaded),
        estimated_tokens=tokens,
    )


def test_first_choice_and_fallback():
    selection = choose("coding")
    assert (selection.model, selection.fallback) == ("coder", "general")
    assert selection.escalate is False


def test_complexity_changes_the_order():
    assert choose("reasoning", "low").model == "general"
    assert choose("reasoning", "medium").model == "thinker"


def test_loaded_model_wins_only_among_equals():
    assert choose("structured_output").model == "general"
    assert choose("structured_output", loaded=["coder"]).model == "coder"
    # "thinker" is loaded but ranked lower, so it does not jump the queue.
    assert choose("structured_output", loaded=["thinker"]).model == "general"
    assert choose("coding", loaded=["general"]).model == "coder"


def test_uninstalled_model_is_skipped():
    installed = {k: v for k, v in ALL.items() if k != "coder"}
    selection = choose("coding", installed=installed)
    assert (selection.model, selection.fallback) == ("general", None)
    assert "not installed: coder" in selection.notes


def test_no_suitable_model_uses_last_resort_and_escalates():
    selection = choose("medical", installed={"thinker": info("thinker")})
    assert selection.model == "thinker"
    assert selection.escalate and "medical" in selection.escalation_reason


def test_unlisted_category_uses_last_resort():
    assert choose("extraction").model == "general"


def test_input_too_long_for_first_choice_moves_to_one_that_fits():
    selection = choose("medical", tokens=3500)  # medic holds 4096; 75% budget is 3072
    assert (selection.model, selection.fallback) == ("general", "medic")
    assert selection.escalate is False


def test_input_too_long_for_every_model_escalates():
    selection = choose("medical", tokens=20_000)
    assert selection.model == "medic"
    assert selection.escalate and "cut off" in selection.escalation_reason


def test_high_complexity_escalates():
    selection = choose("coding", "high")
    assert selection.escalate and "High-complexity" in selection.escalation_reason


def test_nothing_installed():
    selection = choose("coding", installed={})
    assert (selection.model, selection.fallback) == (None, None)
    assert selection.escalate


def test_unknown_availability_uses_policy_order():
    selection = choose("coding", installed=None)
    assert selection.model == "coder"
    assert any("availability unknown" in note for note in selection.notes)


# ---- classifier answer validation ----


def test_classifier_answer_parsing():
    parsed = classifier.parse('{"category": "coding", "complexity": "high"}')
    assert (parsed.category, parsed.complexity) == ("coding", "high")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "coding",
        '{"category": "cooking", "complexity": "low"}',
        '{"category": "coding"}',
        '{"category": "coding", "complexity": "extreme"}',
        '["coding", "low"]',
        '{"category": "coding", "complexity": "low"',
    ],
)
def test_invalid_classifier_answers_are_rejected(raw):
    with pytest.raises(classifier.ClassifierUnavailable):
        classifier.parse(raw)
