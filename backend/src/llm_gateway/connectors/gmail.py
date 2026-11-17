"""Read-only Gmail, through the Gmail API with the gmail.readonly permission.

Setup is once: `scripts/gmail_auth.py` asks Google for permission in the
browser and saves a token in data/ (never committed). With that permission
the gateway can search and read mail, and nothing else: it cannot send,
delete, label or change anything, whatever a model asks for.
"""

import asyncio
import base64
import html
import json
import re
import time
from pathlib import Path

import httpx2

from llm_gateway.agents.world import ToolOutcome
from llm_gateway.connectors.base import (
    ConnectorStatus,
    capped,
    dump,
    error,
    function,
    invalid,
)

SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://gmail.googleapis.com/gmail/v1/users/me"
MAX_RESULTS = 10
BODY_CHARS = 5000


def load_client(path: Path) -> dict:
    """client_id and client_secret from the JSON file Google Cloud gives for a desktop app."""
    data = json.loads(path.read_text())
    client = data.get("installed") or data.get("web") or data
    return {"client_id": client["client_id"], "client_secret": client["client_secret"]}


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode("utf-8", errors="replace")


def _strip_html(markup: str) -> str:
    markup = re.sub(r"(?is)<(script|style).*?</\1>", " ", markup)
    markup = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", markup)
    text = html.unescape(re.sub(r"<[^>]+>", " ", markup))
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


def body_text(payload: dict) -> str:
    """The plain-text body of a message, or its HTML body as text."""
    found: dict[str, str] = {}

    def walk(part: dict) -> None:
        mime = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data")
        if data and mime in ("text/plain", "text/html") and mime not in found:
            found[mime] = _decode(data)
        for child in part.get("parts") or []:
            walk(child)

    walk(payload)
    if "text/plain" in found:
        return found["text/plain"].strip()
    return _strip_html(found.get("text/html", ""))


def _headers(payload: dict) -> dict:
    return {h["name"].lower(): h["value"] for h in payload.get("headers", [])}


class Gmail:
    name = "gmail"
    label = "Gmail"

    def __init__(
        self, client_path: Path, token_path: Path, client: httpx2.AsyncClient | None = None
    ) -> None:
        self.client_path = client_path
        self.token_path = token_path
        self._http = client
        self._lock = asyncio.Lock()

    def _token(self) -> dict | None:
        try:
            return json.loads(self.token_path.read_text())
        except (OSError, ValueError):
            return None

    def status(self) -> ConnectorStatus:
        token = self._token()
        if not self.client_path.is_file():
            return ConnectorStatus(
                self.name,
                self.label,
                False,
                f"Not set up: put your Google OAuth client file at {self.client_path}, then run "
                "scripts/gmail_auth.py (see docs/USAGE.md).",
            )
        if not token or not token.get("refresh_token"):
            return ConnectorStatus(
                self.name, self.label, False, "Not connected: run scripts/gmail_auth.py."
            )
        return ConnectorStatus(
            self.name, self.label, True, f"{token.get('email', 'connected')} (read-only)"
        )

    def tools(self) -> list[dict]:
        return [
            function(
                "search_emails",
                "Search the user's Gmail. Uses Gmail search syntax, e.g. 'from:anna invoice', "
                "'subject:meeting after:2026/11/01', 'is:unread newer_than:7d'. Returns id, "
                "sender, subject, date and a short snippet of each match, newest first.",
                {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": f"At most {MAX_RESULTS}"},
                },
                ["query"],
            ),
            function(
                "read_email",
                "Read one email in full by its id (from search_emails).",
                {"id": {"type": "string"}},
                ["id"],
            ),
        ]

    async def _client(self) -> httpx2.AsyncClient:
        if self._http is None:
            self._http = httpx2.AsyncClient(timeout=20.0)
        return self._http

    async def _access_token(self) -> str:
        """A valid access token, refreshed with the saved refresh token when needed."""
        async with self._lock:
            token = self._token()
            if not token or not token.get("refresh_token"):
                raise PermissionError("Gmail is not connected.")
            if token.get("access_token") and token.get("expiry", 0) > time.time() + 60:
                return token["access_token"]
            client = load_client(self.client_path)
            http = await self._client()
            response = await http.post(
                TOKEN_URL,
                data={
                    **client,
                    "grant_type": "refresh_token",
                    "refresh_token": token["refresh_token"],
                },
            )
            if response.status_code != 200:
                raise PermissionError(
                    "Google refused to refresh the Gmail token; run scripts/gmail_auth.py again."
                )
            fresh = response.json()
            token["access_token"] = fresh["access_token"]
            token["expiry"] = time.time() + int(fresh.get("expires_in", 3600))
            self.token_path.write_text(json.dumps(token))
            return token["access_token"]

    async def _get(self, path: str, params: dict | list | None = None) -> tuple[int, dict]:
        http = await self._client()
        response = await http.get(
            API + path,
            params=params,
            headers={"Authorization": f"Bearer {await self._access_token()}"},
        )
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, {}

    async def call(self, name: str, args: dict) -> ToolOutcome:
        try:
            if name == "search_emails":
                return await self._search(args)
            if name == "read_email":
                return await self._read(str(args.get("id") or "").strip())
        except PermissionError as exc:
            return error(str(exc))
        except httpx2.HTTPError as exc:
            return error(f"Gmail could not be reached: {type(exc).__name__}")
        return invalid(f"Unknown tool: {name}")

    async def _search(self, args: dict) -> ToolOutcome:
        if "query" not in args:
            return invalid("Missing required argument: query")
        try:
            limit = max(1, min(MAX_RESULTS, int(args.get("max_results") or MAX_RESULTS)))
        except (TypeError, ValueError):
            limit = MAX_RESULTS
        status, listing = await self._get(
            "/messages", {"q": str(args["query"]), "maxResults": limit}
        )
        if status != 200:
            return error(f"Gmail search failed (HTTP {status}).")
        ids = [m["id"] for m in listing.get("messages", [])]
        params = [("format", "metadata")] + [
            ("metadataHeaders", h) for h in ("From", "Subject", "Date")
        ]
        found = await asyncio.gather(*(self._get(f"/messages/{i}", params) for i in ids))
        results = []
        for code, message in found:
            if code != 200:
                continue
            headers = _headers(message.get("payload", {}))
            results.append(
                {
                    "id": message["id"],
                    "from": headers.get("from", ""),
                    "subject": headers.get("subject", ""),
                    "date": headers.get("date", ""),
                    "snippet": html.unescape(message.get("snippet", "")),
                }
            )
        return ToolOutcome(dump(results))

    async def _read(self, message_id: str) -> ToolOutcome:
        if not message_id:
            return invalid("Missing required argument: id")
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,64}", message_id):
            return error(f"Not a Gmail message id: {message_id}")
        status, message = await self._get(f"/messages/{message_id}", {"format": "full"})
        if status != 200:
            return error(f"No email with id {message_id}")
        payload = message.get("payload", {})
        headers = _headers(payload)
        return ToolOutcome(
            dump(
                {
                    "id": message_id,
                    "from": headers.get("from", ""),
                    "to": headers.get("to", ""),
                    "subject": headers.get("subject", ""),
                    "date": headers.get("date", ""),
                    "body": capped(body_text(payload), BODY_CHARS),
                }
            )
        )
