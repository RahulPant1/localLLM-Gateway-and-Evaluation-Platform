"""Turning pasted text and uploaded files into batch prompts."""

import json

import pytest

from llm_gateway.batch_input import apply_smart_split, numbered_view, parse_file, split_text


def prompts(parsed):
    return [p.prompt for p in parsed.prompts]


def test_separator_lines_win():
    text = "First prompt\n\nwith two paragraphs\n---\nSecond\n---\n\n"
    parsed = split_text(text)
    assert parsed.method == "separator"
    assert prompts(parsed) == ["First prompt\n\nwith two paragraphs", "Second"]


def test_numbered_list_from_a_chat_model():
    text = (
        "Here are three test prompts:\n\n"
        "1. Summarise this passage:\n\n"
        "The quick brown fox jumps.\n\n"
        "2) Translate 'hello' into French.\n"
        "**Prompt 3:** What is 2 + 2?\n"
    )
    parsed = split_text(text)
    assert parsed.method == "numbered"
    assert prompts(parsed) == [
        "Summarise this passage:\n\nThe quick brown fox jumps.",
        "Translate 'hello' into French.",
        "What is 2 + 2?",
    ]
    assert parsed.notes  # the introduction was left out


def test_blank_lines_and_one_per_line():
    assert prompts(split_text("Alpha question\n\n\nBeta question\n")) == [
        "Alpha question",
        "Beta question",
    ]
    assert split_text("Alpha\n\nBeta").method == "blank_lines"
    parsed = split_text("Alpha\nBeta\nGamma")
    assert parsed.method == "lines" and prompts(parsed) == ["Alpha", "Beta", "Gamma"]


def test_a_forced_method_and_code_fences():
    text = "```\nOne\nTwo\n```"
    assert prompts(split_text(text, "lines")) == ["One", "Two"]
    assert prompts(split_text("One\nstill one", "blank_lines")) == ["One\nstill one"]


def test_json_jsonl_and_csv_files():
    data = [
        "Plain string prompt",
        {"prompt": "With source", "source_material": "Some text", "system": "Be brief"},
    ]
    parsed = parse_file(json.dumps(data), "set.json")
    assert prompts(parsed) == ["Plain string prompt", "With source"]
    assert parsed.prompts[1].source_material == "Some text"
    assert parsed.prompts[1].system == "Be brief"
    assert prompts(parse_file(json.dumps({"prompts": ["a", "b"]}), "x.json")) == ["a", "b"]
    jsonl = '{"prompt": "one"}\n\n{"prompt": "two"}\n'
    assert prompts(parse_file(jsonl, "x.jsonl")) == ["one", "two"]
    table = 'id,Prompt,system\n1,"Hello, world",\n2,Second one,Be brief\n'
    parsed = parse_file(table, "x.csv")
    assert prompts(parsed) == ["Hello, world", "Second one"]
    assert parsed.prompts[1].system == "Be brief"
    with pytest.raises(ValueError):
        parse_file("a,b\n1,2\n", "x.csv")


def test_smart_split_cuts_the_original_text_at_the_marked_lines():
    text = "Intro line\nFirst task\ncontinues here\n\nSecond task\nThanks!"
    lines, view = numbered_view(text)
    assert view.splitlines()[0] == "[1] Intro line"
    parsed = apply_smart_split(lines, {"starts": [2, 5], "skip": [1, 6]})
    assert prompts(parsed) == ["First task\ncontinues here", "Second task"]
    with pytest.raises(ValueError):
        apply_smart_split(lines, {"starts": [], "skip": []})
    # Out-of-range or skipped starts are ignored.
    assert prompts(apply_smart_split(lines, {"starts": [2, 99], "skip": []})) == [
        "First task\ncontinues here\n\nSecond task\nThanks!"
    ]


def test_smart_split_keeps_referenced_material_and_drops_headings():
    text = (
        "Here are prompts:\n"  # 1
        "### Summarisation\n"  # 2
        "Summarise this passage:\n"  # 3
        "\n"  # 4
        "The council voted to extend the bike lanes.\n"  # 5
        "\n"  # 6
        "### Coding\n"  # 7
        "Write a palindrome check.\n"  # 8
        "Hope this helps!"  # 9
    )
    lines, _ = numbered_view(text)
    # The model wrongly skips the passage (5) and starts a prompt at a heading (2).
    parsed = apply_smart_split(lines, {"starts": [2, 8], "skip": [1, 5, 7, 9]})
    assert prompts(parsed) == [
        "Summarise this passage:\n\nThe council voted to extend the bike lanes.",
        "Write a palindrome check.",
    ]


def test_parse_endpoint(client, provider):
    response = client.post("/api/batches/parse", json={"text": "1. Alpha\n2. Beta"})
    assert response.status_code == 200
    assert response.json()["method"] == "numbered"
    assert [p["prompt"] for p in response.json()["prompts"]] == ["Alpha", "Beta"]

    upload = client.post(
        "/api/batches/parse", json={"text": '["x", "y"]', "filename": "set.json"}
    ).json()
    assert upload["method"] == "json" and len(upload["prompts"]) == 2
    bad = client.post("/api/batches/parse", json={"text": "{nope", "filename": "set.json"})
    assert bad.status_code == 422

    # Smart: the fake model answers with line numbers.
    provider.classifier_answer = '{"starts": [2, 3], "skip": [1]}'
    smart = client.post(
        "/api/batches/parse", json={"text": "Here you go:\nAlpha\nBeta", "method": "smart"}
    ).json()
    assert [p["prompt"] for p in smart["prompts"]] == ["Alpha", "Beta"]
    assert smart["model"] == "small:1b"  # no gemma installed: the classifier model
