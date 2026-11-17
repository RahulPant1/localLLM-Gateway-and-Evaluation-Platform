"""Read-only access to one folder on this machine.

Paths are relative to the folder and cannot leave it: `..`, absolute paths
and symbolic links that point outside are refused. Only text files are read,
and only the first part of a large one.
"""

import os
from pathlib import Path

from llm_gateway.agents.world import ToolOutcome
from llm_gateway.connectors.base import (
    ConnectorStatus,
    capped,
    dump,
    error,
    function,
    invalid,
)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".cache", ".idea"}
MAX_READ_BYTES = 400_000
MAX_LIST = 200
MAX_SEARCH_FILES = 2000
MAX_HITS = 30
# Search results show excerpts for the first few files: the best-matching lines
# and the lines after each.
MAX_EXCERPT_FILES = 5
EXCERPTS = 2
AFTER_LINES = 8
# One read returns at most this much of a file; longer files come in parts.
PART_CHARS = 5000


class LocalFiles:
    name = "files"
    label = "Local files"

    def __init__(self, root: Path | None) -> None:
        self.root = root.expanduser().resolve() if root else None

    def status(self) -> ConnectorStatus:
        if self.root is None:
            return ConnectorStatus(
                self.name, self.label, False, "Set GATEWAY_FILES_ROOT in .env to a folder."
            )
        if not self.root.is_dir():
            return ConnectorStatus(self.name, self.label, False, f"Not a folder: {self.root}")
        return ConnectorStatus(self.name, self.label, True, f"{self.root} (read-only)")

    def tools(self) -> list[dict]:
        path = {
            "type": "string",
            "description": "Path inside the shared folder, e.g. notes/todo.md",
        }
        return [
            function(
                "list_files",
                "List files and folders inside a folder of the user's shared folder. "
                "Use '' or '/' for the top.",
                {"path": path},
                ["path"],
            ),
            function(
                "read_file",
                "Read a text file from the user's shared folder.",
                {"path": path},
                ["path"],
            ),
            function(
                "search_files",
                "Find files in the user's shared folder whose name or text contains all the words. "
                "For each file, also returns the best-matching lines with the lines after them.",
                {"query": {"type": "string"}},
                ["query"],
            ),
        ]

    def _resolve(self, raw) -> Path | None:
        """The real path inside the root, or None if it would leave the root."""
        relative = str(raw or "").strip().lstrip("/\\")
        target = (self.root / relative).resolve()
        return target if target == self.root or self.root in target.parents else None

    def _relative(self, path: Path) -> str:
        return (
            "/" + str(path.relative_to(self.root)).replace(os.sep, "/")
            if path != self.root
            else "/"
        )

    def call(self, name: str, args: dict) -> ToolOutcome:
        if self.root is None:
            return error("Local files are not set up.")
        if name == "list_files":
            return self._list(args.get("path"))
        if name == "read_file":
            if not args.get("path"):
                return invalid("Missing required argument: path")
            return self._read(args["path"], args.get("part"))
        if name == "search_files":
            if not str(args.get("query") or "").strip():
                return invalid("Missing required argument: query")
            return self._search(str(args["query"]))
        return invalid(f"Unknown tool: {name}")

    def _list(self, raw) -> ToolOutcome:
        folder = self._resolve(raw)
        if folder is None:
            return error("That path is outside the shared folder.")
        if not folder.is_dir():
            return error(f"Folder not found: {raw}")
        entries = []
        for child in sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith(".") or child.name in SKIP_DIRS:
                continue
            if child.is_dir():
                entries.append(child.name + "/")
            elif self._resolve(self._relative(child)) is not None:
                entries.append(child.name)
            if len(entries) >= MAX_LIST:
                entries.append(f"…(more than {MAX_LIST} entries)")
                break
        return ToolOutcome(dump(entries))

    @staticmethod
    def _text(path: Path) -> str | None:
        """The start of a file as text, or None for a binary file."""
        with path.open("rb") as handle:
            data = handle.read(MAX_READ_BYTES)
        if b"\0" in data[:4096]:
            return None
        return data.decode("utf-8", errors="replace")

    def _read(self, raw, part=None) -> ToolOutcome:
        path = self._resolve(raw)
        if path is None:
            return error("That path is outside the shared folder.")
        if not path.is_file():
            return error(f"File not found: {raw}")
        text = self._text(path)
        if text is None:
            return error(f"{self._relative(path)} is not a text file.")
        if not text:
            return ToolOutcome("(empty file)")
        parts = max(1, -(-len(text) // PART_CHARS))
        try:
            number = max(1, int(part or 1))
        except (TypeError, ValueError):
            return invalid("part must be a number")
        if number > parts:
            return error(f"{self._relative(path)} has only {parts} part(s).")
        chunk = text[(number - 1) * PART_CHARS : number * PART_CHARS]
        if parts == 1:
            return ToolOutcome(chunk)
        more = (
            f" Call read_file with part={number + 1} for the next part." if number < parts else ""
        )
        return ToolOutcome(f"[{self._relative(path)}: part {number} of {parts}.{more}]\n{chunk}")

    @staticmethod
    def _excerpts(text: str, words: list[str]) -> list[str]:
        """The lines that match most query words, each with the lines after it
        (like grep -A), so a table or list comes with its rows."""
        lines = text.splitlines()
        scored = [
            (sum(w in line.lower() for w in words), i)
            for i, line in enumerate(lines)
            if line.strip()
        ]
        best = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))[:EXCERPTS]
        out = []
        for _, i in sorted(best, key=lambda s: s[1]):
            block = "\n".join(line[:300] for line in lines[max(0, i - 1) : i + 1 + AFTER_LINES])
            out.append(f"line {i + 1}:\n{block}")
        return out

    def _search(self, query: str) -> ToolOutcome:
        words = query.lower().split()
        hits, scanned = [], 0
        for folder, dirs, files in os.walk(self.root):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS)
            for filename in sorted(files):
                path = Path(folder, filename)
                if filename.startswith(".") or self._resolve(self._relative(path)) is None:
                    continue
                scanned += 1
                if scanned > MAX_SEARCH_FILES:
                    break
                text = self._text(path) or ""
                name_hit = all(w in filename.lower() for w in words)
                if name_hit or all(w in text.lower() for w in words):
                    hit = {"path": self._relative(path)}
                    if text and len(hits) < MAX_EXCERPT_FILES:
                        hit["excerpts"] = self._excerpts(text, words)
                    hits.append(hit)
                    if len(hits) >= MAX_HITS:
                        return ToolOutcome(capped(dump(hits)))
        return ToolOutcome(capped(dump(hits)))
