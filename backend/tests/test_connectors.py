"""Real-app connectors (files on disk; Gmail and web against mock HTTP) and agent runs."""

import asyncio
import base64
import json
import os
import time

import httpx2
import pytest

from llm_gateway.connectors.files import LocalFiles
from llm_gateway.connectors.gmail import Gmail, body_text
from llm_gateway.connectors.web import WebApis
from llm_gateway.providers.base import ToolCall


def data(outcome):
    return json.loads(outcome.text)


# ---- local files ----


@pytest.fixture
def shared(tmp_path):
    root = tmp_path / "shared"
    (root / "notes").mkdir(parents=True)
    (root / "notes" / "todo.md").write_text("- renew the domain\n- call the bank\n")
    (root / "report.txt").write_text("Q3 revenue was 446,000 EUR.")
    (root / "image.bin").write_bytes(b"\x89PNG\0\0\0binary")
    (root / ".git").mkdir()
    (tmp_path / "secret.txt").write_text("outside the shared folder")
    os.symlink(tmp_path / "secret.txt", root / "link.txt")
    return root


def test_files_list_read_and_search(shared):
    files = LocalFiles(shared)
    assert files.status().available
    assert data(files.call("list_files", {"path": "/"})) == ["notes/", "image.bin", "report.txt"]
    assert "call the bank" in files.call("read_file", {"path": "notes/todo.md"}).text
    (bank,) = data(files.call("search_files", {"query": "bank"}))
    assert bank["path"] == "/notes/todo.md"
    assert "call the bank" in bank["excerpts"][0]
    assert [h["path"] for h in data(files.call("search_files", {"query": "report"}))] == [
        "/report.txt"
    ]
    assert files.call("read_file", {"path": "image.bin"}).error


def test_long_files_are_read_in_parts(shared):
    (shared / "long.md").write_text("A" * 6000 + "END")
    files = LocalFiles(shared)
    first = files.call("read_file", {"path": "long.md"}).text
    assert first.startswith("[/long.md: part 1 of 2. Call read_file with part=2")
    second = files.call("read_file", {"path": "long.md", "part": 2}).text
    assert second.startswith("[/long.md: part 2 of 2.]") and second.endswith("END")
    assert files.call("read_file", {"path": "long.md", "part": 3}).error


def test_files_cannot_leave_the_folder(shared):
    files = LocalFiles(shared)
    for path in ("../secret.txt", "/../secret.txt", "link.txt", "notes/../../secret.txt"):
        outcome = files.call("read_file", {"path": path})
        assert outcome.error and "outside" in outcome.text, path
    assert "link.txt" not in files.call("list_files", {"path": ""}).text
    assert data(files.call("search_files", {"query": "outside"})) == []


def test_search_excerpts_bring_the_rows_after_a_heading(shared):
    table = (
        "intro\n\n## Agent tasks\n| Model | Tasks passed |\n|---|---|\n"
        "| qwen | 18 |\n| gemma | 15 |\n"
    )
    (shared / "results.md").write_text(table)
    (hit,) = data(LocalFiles(shared).call("search_files", {"query": "tasks passed"}))
    assert "| gemma | 15 |" in hit["excerpts"][0]


def test_files_off_without_a_folder():
    files = LocalFiles(None)
    assert not files.status().available
    assert files.call("list_files", {"path": "/"}).error


# ---- web APIs ----


def web_transport(request: httpx2.Request) -> httpx2.Response:
    url = str(request.url)
    if "geocoding" in url:
        return httpx2.Response(
            200,
            json={
                "results": [
                    {"name": "Lisbon", "country": "Portugal", "latitude": 1, "longitude": 2}
                ]
            },
        )
    if "forecast" in url:
        return httpx2.Response(
            200,
            json={
                "current": {
                    "time": "now",
                    "temperature_2m": 18.5,
                    "weather_code": 61,
                    "wind_speed_10m": 12,
                }
            },
        )
    if "frankfurter" in url:
        return httpx2.Response(200, json={"date": "2026-11-16", "rates": {"USD": 275.35}})
    if "wikipedia" in url:
        if "Nope" in url:
            return httpx2.Response(404, json={})
        return httpx2.Response(
            200, json={"title": "Ada Lovelace", "extract": "English mathematician."}
        )
    return httpx2.Response(404)


def test_web_tools():
    asyncio.run(_web_tools())


async def _web_tools():
    web = WebApis(True, httpx2.AsyncClient(transport=httpx2.MockTransport(web_transport)))
    weather = data(await web.call("get_weather", {"city": "Lisbon"}))
    assert weather["temperature_c"] == 18.5 and weather["conditions"] == "light rain"
    money = data(await web.call("convert_currency", {"amount": 250, "from": "eur", "to": "usd"}))
    assert money["result"] == 275.35 and money["from"] == "EUR"
    assert data(await web.call("wikipedia_summary", {"topic": "Ada Lovelace"}))["summary"]
    assert (await web.call("wikipedia_summary", {"topic": "Nope"})).error
    assert (
        await web.call("convert_currency", {"amount": "lots", "from": "EUR", "to": "USD"})
    ).invalid
    assert not WebApis(False).status().available


# ---- Gmail ----


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


MESSAGE = {
    "id": "abc123def",
    "snippet": "Invoice &amp; payment",
    "payload": {
        "headers": [
            {"name": "From", "value": "Acme <billing@acme.example>"},
            {"name": "Subject", "value": "Invoice 4471"},
            {"name": "Date", "value": "Mon, 16 Nov 2026"},
        ],
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/html", "body": {"data": b64("<p>HTML version</p>")}},
            {"mimeType": "text/plain", "body": {"data": b64("Please pay EUR 1,250.")}},
        ],
    },
}


def test_body_text_prefers_plain_and_falls_back_to_html():
    assert body_text(MESSAGE["payload"]) == "Please pay EUR 1,250."
    only_html = {
        "mimeType": "text/html",
        "body": {"data": b64("<p>Hi&nbsp;there</p><script>x</script>")},
    }
    assert body_text(only_html) == "Hi\xa0there"


@pytest.fixture
def gmail_files(tmp_path):
    client = tmp_path / "client.json"
    client.write_text(json.dumps({"installed": {"client_id": "id", "client_secret": "secret"}}))
    token = tmp_path / "token.json"
    token.write_text(
        json.dumps(
            {"refresh_token": "r", "access_token": "old", "expiry": 0, "email": "me@x.example"}
        )
    )
    return client, token


def test_gmail_refreshes_the_token_and_reads_mail(gmail_files):
    asyncio.run(_gmail_reads(gmail_files))


async def _gmail_reads(gmail_files):
    client_path, token_path = gmail_files
    seen = []

    def transport(request: httpx2.Request) -> httpx2.Response:
        seen.append((request.method, request.url.path, request.headers.get("authorization")))
        if request.url.path == "/token":
            assert b"grant_type=refresh_token" in request.content
            return httpx2.Response(200, json={"access_token": "fresh", "expires_in": 3600})
        if request.url.path.endswith("/messages"):
            assert request.url.params["q"] == "from:acme"
            return httpx2.Response(200, json={"messages": [{"id": "abc123def"}]})
        if request.url.path.endswith("/messages/abc123def"):
            return httpx2.Response(200, json=MESSAGE)
        return httpx2.Response(404)

    gmail = Gmail(
        client_path, token_path, httpx2.AsyncClient(transport=httpx2.MockTransport(transport))
    )
    assert gmail.status().available and "me@x.example" in gmail.status().detail
    found = data(await gmail.call("search_emails", {"query": "from:acme"}))
    assert found == [
        {
            "id": "abc123def",
            "from": "Acme <billing@acme.example>",
            "subject": "Invoice 4471",
            "date": "Mon, 16 Nov 2026",
            "snippet": "Invoice & payment",
        }
    ]
    email = data(await gmail.call("read_email", {"id": "abc123def"}))
    assert email["body"] == "Please pay EUR 1,250."
    # Refreshed once, then reused; the new token is saved.
    assert [s for s in seen if s[1] == "/token"] == [("POST", "/token", None)]
    assert all(auth == "Bearer fresh" for _, path, auth in seen if path != "/token")
    assert json.loads(token_path.read_text())["access_token"] == "fresh"
    assert (await gmail.call("read_email", {"id": "../../etc"})).error
    # Only reading tools exist.
    assert {t["function"]["name"] for t in gmail.tools()} == {"search_emails", "read_email"}


def test_gmail_not_connected(tmp_path):
    gmail = Gmail(tmp_path / "missing.json", tmp_path / "token.json")
    assert not gmail.status().available


# ---- agent runs from the Playground ----


def wait_request(client, request_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/api/requests/{request_id}").json()
        if record["executions"][0]["status"] in ("completed", "failed"):
            return record
        time.sleep(0.01)
    raise AssertionError(record)


def test_agent_run_on_local_files(make_client, provider, shared):
    client = make_client(files_root=shared, web_apis=False)
    apps = {c["name"]: c for c in client.get("/api/connectors").json()}
    assert apps["files"]["available"] and not apps["web"]["available"]
    assert not apps["gmail"]["available"]

    provider.chat_script = [
        ("", [ToolCall(name="search_files", arguments={"query": "revenue"})]),
        ("", [ToolCall(name="read_file", arguments={"path": "/report.txt"})]),
        ("Q3 revenue was 446,000 EUR (from report.txt).", []),
    ]
    response = client.post(
        "/api/agent-runs",
        json={"prompt": "What was Q3 revenue?", "model": "small:1b", "apps": ["files"]},
    )
    assert response.status_code == 201, response.text
    record = wait_request(client, response.json()["id"])
    execution = record["executions"][0]
    assert execution["status"] == "completed"
    assert execution["response"].startswith("Q3 revenue was 446,000")
    agent = execution["metrics"]["agent"]
    assert agent["apps"] == ["files"] and agent["stop_reason"] == "answered"
    assert agent["tool_calls"] == 2 and len(agent["trace"]) == 3
    assert record["source"] == "agent"
    # The tool results are kept for the judge.
    assert "446,000" in record["source_material"]
    # The model was told that tool output is data, not instructions.
    system = provider.chat_messages[0][0]["content"]
    assert "not instructions" in system and "local files" in system


def test_agent_run_refuses_apps_that_are_not_set_up(make_client):
    client = make_client(web_apis=False)
    response = client.post(
        "/api/agent-runs", json={"prompt": "x", "model": "small:1b", "apps": ["gmail"]}
    )
    assert response.status_code == 422 and "gmail" in response.json()["detail"]
    response = client.post(
        "/api/agent-runs", json={"prompt": "x", "model": "thinker:8b", "apps": ["files"]}
    )
    assert response.status_code == 422
