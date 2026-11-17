"""Simulated apps for agent tests: a file system, a mailbox and a company API.

Everything is in memory and deterministic, so a scenario can be run again and
gives the same tool results. Nothing real is read, sent or changed. Each tool
returns text for the model; every side effect is also recorded in `effects`,
which the checks read after the run.
"""

import copy
import json
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

API_BASE = "https://api.example.internal"

# The tools each app offers, in the OpenAI / Ollama function format.
TOOLS: dict[str, list[dict]] = {
    "files": [
        {
            "name": "list_files",
            "description": "List the files and folders inside a folder.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Folder, e.g. /notes"}},
                "required": ["path"],
            },
        },
        {
            "name": "read_file",
            "description": "Read a text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Full path of the file"}},
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Create a text file, or replace the whole content of an existing one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "The complete new content"},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "delete_file",
            "description": "Delete a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    ],
    "mail": [
        {
            "name": "search_emails",
            "description": (
                "Search the inbox. Returns id, sender, subject and date of each match. "
                "Words are matched in sender, subject and body (all must appear). Also "
                "understands from:, subject:, after:YYYY-MM-DD and before:YYYY-MM-DD. "
                "An empty query lists every email."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "read_email",
            "description": "Read one email in full.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
        {
            "name": "send_email",
            "description": "Send an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
        {
            "name": "create_draft",
            "description": "Save an email as a draft without sending it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
        {
            "name": "archive_email",
            "description": "Archive an email so it no longer shows in the inbox.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    ],
    "api": [
        {
            "name": "http_request",
            "description": (
                f"Call the company API at {API_BASE}. Endpoints: "
                "GET /weather?city=NAME (current weather); "
                "GET /fx?from=CUR&to=CUR (exchange rate); "
                "GET /orders/ID and GET /orders?customer=NAME (orders); "
                "GET /contacts?name=NAME (a colleague's email address); "
                "POST /tickets with body {title, body, priority: low|normal|high} (support ticket)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["GET", "POST"]},
                    "url": {
                        "type": "string",
                        "description": "Full URL or path, e.g. /fx?from=EUR&to=USD",
                    },
                    "body": {"type": "object", "description": "JSON body for POST"},
                },
                "required": ["method", "url"],
            },
        },
    ],
}

TOOL_APP = {tool["name"]: app for app, tools in TOOLS.items() for tool in tools}
MAX_RESULT_CHARS = 2000


def tool_schemas(apps: list[str]) -> list[dict]:
    return [{"type": "function", "function": t} for app in apps for t in TOOLS[app]]


def _merge(base: dict, override: dict) -> dict:
    """Scenario changes on top of the shared world: dicts merge, other values replace."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


@dataclass
class ToolOutcome:
    text: str
    # The call itself was wrong: unknown tool or a required argument missing.
    invalid: bool = False
    # The call was valid but the app reported an error (file not found, 404...).
    error: bool = False


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class World:
    def __init__(self, base: dict, override: dict | None = None, apps: list[str] | None = None):
        state = _merge(base, override or {})
        self.files: dict[str, str] = state.get("files", {})
        self.inbox: list[dict] = state.get("mail", {}).get("inbox", [])
        self.api: dict = state.get("api", {})
        self.apps = apps or list(TOOLS)
        self.fail_first = set(self.api.get("fail_first", []))
        self.effects: dict = {
            "sent": [],
            "drafts": [],
            "archived": [],
            "deleted": [],
            "written": [],
            "http": [],
            "tickets": [],
            "calls": [],
        }
        self._next_ticket = 501

    # ---- dispatch ----

    def call(self, name: str, arguments: dict) -> ToolOutcome:
        self.effects["calls"].append({"name": name, "arguments": arguments})
        if TOOL_APP.get(name) not in self.apps:
            return ToolOutcome(_dump({"error": f"Unknown tool: {name}"}), invalid=True)
        spec = next(t for t in TOOLS[TOOL_APP[name]] if t["name"] == name)
        missing = [
            k for k in spec["parameters"].get("required", []) if arguments.get(k) in (None, "")
        ]
        if missing and not (name == "search_emails" and missing == ["query"]):
            return ToolOutcome(
                _dump({"error": f"Missing required argument(s): {', '.join(missing)}"}),
                invalid=True,
            )
        outcome = getattr(self, f"_{name}")(arguments)
        if len(outcome.text) > MAX_RESULT_CHARS:
            outcome.text = outcome.text[:MAX_RESULT_CHARS] + " …(truncated)"
        return outcome

    @staticmethod
    def _error(message: str) -> ToolOutcome:
        return ToolOutcome(_dump({"error": message}), error=True)

    # ---- files ----

    @staticmethod
    def _norm(path) -> str:
        path = "/" + str(path).strip().strip("/")
        return path if path != "/" else "/"

    def _list_files(self, a: dict) -> ToolOutcome:
        folder = self._norm(a["path"])
        prefix = "" if folder == "/" else folder
        entries = set()
        for path in self.files:
            if path.startswith(prefix + "/"):
                rest = path[len(prefix) + 1 :]
                entries.add(rest.split("/")[0] + ("/" if "/" in rest else ""))
        if not entries:
            return self._error(f"Folder not found or empty: {folder}")
        return ToolOutcome(_dump(sorted(entries)))

    def _read_file(self, a: dict) -> ToolOutcome:
        path = self._norm(a["path"])
        if path not in self.files:
            return self._error(f"File not found: {path}")
        return ToolOutcome(self.files[path] or "(empty file)")

    def _write_file(self, a: dict) -> ToolOutcome:
        path = self._norm(a["path"])
        self.files[path] = str(a.get("content", ""))
        self.effects["written"].append(path)
        return ToolOutcome(_dump({"ok": True, "path": path, "characters": len(self.files[path])}))

    def _delete_file(self, a: dict) -> ToolOutcome:
        path = self._norm(a["path"])
        if path not in self.files:
            return self._error(f"File not found: {path}")
        del self.files[path]
        self.effects["deleted"].append(path)
        return ToolOutcome(_dump({"ok": True, "deleted": path}))

    # ---- mail ----

    def _visible(self) -> list[dict]:
        return [m for m in self.inbox if m["id"] not in self.effects["archived"]]

    def _search_emails(self, a: dict) -> ToolOutcome:
        words, filters = [], []
        for token in str(a.get("query") or "").split():
            key, _, value = token.partition(":")
            key, value = key.lower(), value.strip("\"'").lower().replace("/", "-")
            if value and key in ("from", "subject"):
                filters.append(lambda m, k=key, v=value: v in m[k].lower())
            elif value and key == "after":
                filters.append(lambda m, v=value: m["date"] >= v)
            elif value and key == "before":
                filters.append(lambda m, v=value: m["date"] < v)
            else:
                words.append(token.lower())
        found = [
            {k: m[k] for k in ("id", "from", "subject", "date")}
            for m in self._visible()
            if all(w in f"{m['from']} {m['subject']} {m['body']}".lower() for w in words)
            and all(f(m) for f in filters)
        ]
        return ToolOutcome(_dump(found))

    def _email(self, email_id) -> dict | None:
        return next((m for m in self.inbox if m["id"] == str(email_id).strip()), None)

    def _read_email(self, a: dict) -> ToolOutcome:
        email = self._email(a["id"])
        if email is None:
            return self._error(f"No email with id {a['id']}")
        return ToolOutcome(_dump(email))

    def _outgoing(self, a: dict, box: str) -> ToolOutcome:
        to = str(a["to"]).strip()
        if "@" not in to:
            return self._error(f"Not an email address: {to}")
        self.effects[box].append(
            {"to": to, "subject": str(a.get("subject", "")), "body": str(a.get("body", ""))}
        )
        return ToolOutcome(_dump({"ok": True, box[:-1] if box == "drafts" else "sent_to": to}))

    def _send_email(self, a: dict) -> ToolOutcome:
        return self._outgoing(a, "sent")

    def _create_draft(self, a: dict) -> ToolOutcome:
        return self._outgoing(a, "drafts")

    def _archive_email(self, a: dict) -> ToolOutcome:
        email = self._email(a["id"])
        if email is None:
            return self._error(f"No email with id {a['id']}")
        if email["id"] not in self.effects["archived"]:
            self.effects["archived"].append(email["id"])
        return ToolOutcome(_dump({"ok": True, "archived": email["id"]}))

    # ---- company API ----

    def _http_request(self, a: dict) -> ToolOutcome:
        method = str(a.get("method", "GET")).upper()
        parsed = urlparse(str(a["url"]).strip())
        path = parsed.path.rstrip("/") or "/"
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        body = a.get("body") if isinstance(a.get("body"), dict) else {}
        status, payload = self._route(method, path, query, body)
        self.effects["http"].append(
            {"method": method, "path": path, "query": query, "body": body, "status": status}
        )
        text = _dump({"status": status, **payload})
        return ToolOutcome(text, error=status >= 400)

    def _route(self, method: str, path: str, query: dict, body: dict) -> tuple[int, dict]:
        if path in self.fail_first:
            self.fail_first.discard(path)
            return 429, {"error": "Too many requests. Wait a moment and try again."}
        lower = {k: str(v).strip().lower() for k, v in query.items()}
        if method == "GET" and path == "/weather":
            city = next(
                (c for c in self.api.get("weather", {}) if c.lower() == lower.get("city")), None
            )
            if city is None:
                return 404, {"error": f"Unknown city: {query.get('city')}"}
            return 200, {"city": city, **self.api["weather"][city]}
        if method == "GET" and path == "/fx":
            pair = f"{lower.get('from', '').upper()}/{lower.get('to', '').upper()}"
            rate = self.api.get("fx", {}).get(pair)
            if rate is None:
                return 404, {"error": f"No rate for {pair}"}
            return 200, {"from": pair[:3], "to": pair[4:], "rate": rate}
        orders = self.api.get("orders", {})
        if method == "GET" and path.startswith("/orders/"):
            order = orders.get(path.split("/")[-1])
            return (200, {"order": order}) if order else (404, {"error": "Order not found"})
        if method == "GET" and path == "/orders":
            customer = lower.get("customer", "")
            found = [o for o in orders.values() if customer and customer in o["customer"].lower()]
            return 200, {"orders": found}
        if method == "GET" and path == "/contacts":
            name = lower.get("name", "")
            match = next(
                (c for c in self.api.get("contacts", []) if name and name in c["name"].lower()),
                None,
            )
            return (
                (200, match) if match else (404, {"error": f"No contact named {query.get('name')}"})
            )
        if method == "POST" and path == "/tickets":
            if not body.get("title"):
                return 400, {"error": "A ticket needs a title"}
            ticket = {"id": f"T-{self._next_ticket}", **body}
            self._next_ticket += 1
            self.effects["tickets"].append(ticket)
            return 201, {"ticket": ticket["id"], "state": "open"}
        return 404, {"error": f"No endpoint {method} {path}"}
