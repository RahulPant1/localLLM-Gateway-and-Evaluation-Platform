"""SQLite persistence. All SQL in the application lives in this module.

Uses Python's built-in sqlite3. Queries take well under a millisecond on a
local file, so they are called directly from async code.
"""

import json
import sqlite3
import threading
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from llm_gateway.providers.base import ChatMessage
from llm_gateway.routing.types import RoutingDecision, RoutingDraft
from llm_gateway.schemas import (
    BatchSummary,
    Evaluation,
    EvaluationInput,
    Execution,
    PromptInput,
    RequestRecord,
    overall_score,
)

# Each entry upgrades the schema by one version. Never edit an entry that has
# shipped; append a new one.
MIGRATIONS = [
    """
    CREATE TABLE batches (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        models TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE requests (
        id TEXT PRIMARY KEY,
        batch_id TEXT REFERENCES batches(id),
        position INTEGER NOT NULL DEFAULT 0,
        prompt TEXT NOT NULL,
        system TEXT,
        source_material TEXT,
        routing_mode TEXT NOT NULL,
        think INTEGER,
        max_tokens INTEGER,
        created_at TEXT NOT NULL
    );
    CREATE INDEX requests_batch ON requests(batch_id, position);
    CREATE INDEX requests_created ON requests(created_at);
    CREATE TABLE executions (
        id TEXT PRIMARY KEY,
        request_id TEXT NOT NULL REFERENCES requests(id),
        position INTEGER NOT NULL DEFAULT 0,
        model TEXT NOT NULL,
        status TEXT NOT NULL,
        response TEXT,
        thinking TEXT,
        error_code TEXT,
        error_detail TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        queue_wait_ms REAL,
        execution_ms REAL,
        metrics TEXT,
        warnings TEXT NOT NULL DEFAULT '[]'
    );
    CREATE INDEX executions_request ON executions(request_id, position);
    CREATE INDEX executions_status ON executions(status);
    CREATE TABLE evaluations (
        id TEXT PRIMARY KEY,
        execution_id TEXT NOT NULL REFERENCES executions(id),
        judge TEXT NOT NULL,
        rubric_version TEXT NOT NULL,
        status TEXT NOT NULL,
        scores TEXT NOT NULL,
        overall_score REAL,
        summary TEXT NOT NULL DEFAULT '',
        strengths TEXT NOT NULL DEFAULT '[]',
        weaknesses TEXT NOT NULL DEFAULT '[]',
        recommendations TEXT NOT NULL DEFAULT '[]',
        critical_issues TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL
    );
    CREATE INDEX evaluations_execution ON evaluations(execution_id);
    """,
    # 2: follow-up chat. A request may continue from an earlier answer.
    """
    ALTER TABLE requests ADD COLUMN parent_execution_id TEXT REFERENCES executions(id);
    CREATE INDEX requests_parent ON requests(parent_execution_id);
    """,
    # 3: routing decisions, one per request.
    """
    CREATE TABLE routing_decisions (
        request_id TEXT PRIMARY KEY REFERENCES requests(id),
        shadow INTEGER NOT NULL DEFAULT 0,
        category TEXT NOT NULL,
        complexity TEXT NOT NULL,
        selected_model TEXT,
        fallback_model TEXT,
        confidence REAL NOT NULL,
        explanation TEXT NOT NULL,
        escalate INTEGER NOT NULL DEFAULT 0,
        escalation_reason TEXT,
        decided_by TEXT NOT NULL,
        candidates TEXT NOT NULL DEFAULT '[]',
        signals TEXT,
        llm TEXT,
        fallback_used INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
    """,
    # 4: the judge's classification of the prompt.
    """
    ALTER TABLE evaluations ADD COLUMN task_type TEXT;
    ALTER TABLE evaluations ADD COLUMN domain TEXT;
    """,
    # 5: requests from the OpenAI-compatible endpoint carry their own history.
    """
    ALTER TABLE requests ADD COLUMN history TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE requests ADD COLUMN source TEXT NOT NULL DEFAULT 'ui';
    ALTER TABLE requests ADD COLUMN temperature REAL;
    """,
    # 6: a second attempt on the same request can override the thinking setting.
    """
    ALTER TABLE executions ADD COLUMN think INTEGER;
    """,
    # 7: tool-calling test runs.
    """
    CREATE TABLE tool_runs (
        id TEXT PRIMARY KEY,
        suite TEXT NOT NULL,
        name TEXT NOT NULL,
        models TEXT NOT NULL,
        think INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
    CREATE TABLE tool_results (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES tool_runs(id),
        case_id TEXT NOT NULL,
        category TEXT NOT NULL,
        model TEXT NOT NULL,
        position INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        calls TEXT NOT NULL DEFAULT '[]',
        -- native: proper tool calls; text: calls written as JSON in the answer; none
        call_format TEXT,
        content TEXT,
        thinking TEXT,
        passed INTEGER,
        reason TEXT,
        error_code TEXT,
        error_detail TEXT,
        prompt_tokens INTEGER,
        output_tokens INTEGER,
        queue_wait_ms REAL,
        execution_ms REAL,
        finished_at TEXT,
        UNIQUE (run_id, case_id, model)
    );
    CREATE INDEX tool_results_run ON tool_results(run_id, position);
    """,
    # 8: models taken out of the lineup. Their results stay stored but are not shown.
    """
    CREATE TABLE hidden_models (
        model TEXT PRIMARY KEY,
        hidden_at TEXT NOT NULL
    );
    """,
    # 9: multi-step agent runs against simulated apps.
    """
    CREATE TABLE agent_runs (
        id TEXT PRIMARY KEY,
        suite TEXT NOT NULL,
        name TEXT NOT NULL,
        models TEXT NOT NULL,
        think INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );
    CREATE TABLE agent_results (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES agent_runs(id),
        case_id TEXT NOT NULL,
        category TEXT NOT NULL,
        model TEXT NOT NULL,
        position INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        passed INTEGER,
        -- Share of the scenario's checks that passed, 0 to 1.
        score REAL,
        checks TEXT NOT NULL DEFAULT '[]',
        trace TEXT NOT NULL DEFAULT '[]',
        effects TEXT NOT NULL DEFAULT '{}',
        answer TEXT,
        stop_reason TEXT,
        steps INTEGER,
        tool_calls INTEGER,
        invalid_calls INTEGER,
        text_calls INTEGER,
        prompt_tokens INTEGER,
        output_tokens INTEGER,
        queue_wait_ms REAL,
        execution_ms REAL,
        error_code TEXT,
        error_detail TEXT,
        finished_at TEXT,
        UNIQUE (run_id, case_id, model)
    );
    CREATE INDEX agent_results_run ON agent_results(run_id, position);
    """,
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _id() -> str:
    return uuid.uuid4().hex


_RUNS = {"tool": "tool_runs", "agent": "agent_runs"}
_RESULTS = {"tool": "tool_results", "agent": "agent_results"}

# Failures caused by the environment, not by the prompt or the model: worth
# running again once the cause is gone.
RESUMABLE_CODES = (
    "interrupted",
    "generation_timeout",
    "low_memory",
    "provider_unavailable",
)


class _Rows:
    """The complete result of one statement, already read from the database."""

    def __init__(self, rows: list, rowcount: int) -> None:
        self._rows = rows
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list:
        return self._rows

    def __iter__(self):
        return iter(self._rows)


class _SafeConnection:
    """One SQLite connection shared by the event loop and the worker threads
    that serve read requests. Every statement runs and is read to the end
    under a lock; a cursor read while another thread used the connection
    returned corrupt rows ("tuple index out of range")."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self.lock = threading.RLock()

    def execute(self, sql: str, params=()) -> _Rows:
        with self.lock:
            cursor = self._connection.execute(sql, params)
            return _Rows(cursor.fetchall(), cursor.rowcount)

    def executemany(self, sql: str, rows) -> None:
        with self.lock:
            self._connection.executemany(sql, rows)

    def executescript(self, script: str) -> None:
        with self.lock:
            self._connection.executescript(script)

    def close(self) -> None:
        with self.lock:
            self._connection.close()


class Store:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: every statement commits at once unless inside _transaction().
        connection = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        connection.row_factory = sqlite3.Row
        self._db = _SafeConnection(connection)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        # Models out of the lineup. Every read below leaves out their results.
        self.hidden: set[str] = {
            row["model"] for row in self._db.execute("SELECT model FROM hidden_models")
        }

    def close(self) -> None:
        self._db.close()

    def _migrate(self) -> None:
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        for number, script in enumerate(MIGRATIONS[version:], start=version + 1):
            self._db.executescript(f"BEGIN;{script};PRAGMA user_version={number};COMMIT;")

    @contextmanager
    def _transaction(self):
        # The lock is held for the whole transaction, so no other thread's
        # statement can land inside it.
        with self._db.lock:
            self._db.execute("BEGIN")
            try:
                yield
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    # ---- writing requests and executions ----

    def _insert_request(
        self,
        item: PromptInput,
        models: list[str],
        *,
        routing_mode: str,
        think: bool | None,
        max_tokens: int | None,
        batch_id: str | None = None,
        position: int = 0,
        parent_execution_id: str | None = None,
        history: list[ChatMessage] | None = None,
        source: str = "ui",
        temperature: float | None = None,
    ) -> str:
        request_id = _id()
        now = _now()
        self._db.execute(
            "INSERT INTO requests (id, batch_id, position, prompt, system, source_material,"
            " routing_mode, think, max_tokens, created_at, parent_execution_id, history,"
            " source, temperature) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id,
                batch_id,
                position,
                item.prompt,
                item.system or None,
                item.source_material or None,
                routing_mode,
                think,
                max_tokens,
                now,
                parent_execution_id,
                json.dumps([m.model_dump() for m in history or []]),
                source,
                temperature,
            ),
        )
        self._db.executemany(
            "INSERT INTO executions (id, request_id, position, model, status, created_at)"
            " VALUES (?,?,?,?,'queued',?)",
            [(_id(), request_id, index, model, now) for index, model in enumerate(models)],
        )
        return request_id

    def create_request(
        self,
        item: PromptInput,
        models: list[str],
        *,
        routing_mode: str = "manual",
        think: bool | None = None,
        max_tokens: int | None = None,
        parent_execution_id: str | None = None,
        history: list[ChatMessage] | None = None,
        source: str = "ui",
        temperature: float | None = None,
    ) -> RequestRecord:
        with self._transaction():
            request_id = self._insert_request(
                item,
                models,
                routing_mode=routing_mode,
                think=think,
                max_tokens=max_tokens,
                parent_execution_id=parent_execution_id,
                history=history,
                source=source,
                temperature=temperature,
            )
        return self.get_request(request_id)

    def create_batch(
        self,
        name: str,
        prompts: list[PromptInput],
        models: list[str],
        *,
        think: bool | None = None,
        max_tokens: int | None = None,
    ) -> str:
        batch_id = _id()
        with self._transaction():
            self._db.execute(
                "INSERT INTO batches (id, name, models, created_at) VALUES (?,?,?,?)",
                (batch_id, name, json.dumps(models), _now()),
            )
            for position, item in enumerate(prompts):
                self._insert_request(
                    item,
                    models,
                    routing_mode="manual",
                    think=think,
                    max_tokens=max_tokens,
                    batch_id=batch_id,
                    position=position,
                )
        return batch_id

    def add_execution(self, request_id: str, model: str, think: bool | None = None) -> Execution:
        """Add a queued execution to an existing request: the routed model, the
        fallback, or a second attempt. `think` overrides the request's setting."""
        execution_id = _id()
        position = self._db.execute(
            "SELECT COUNT(*) FROM executions WHERE request_id = ?", (request_id,)
        ).fetchone()[0]
        self._db.execute(
            "INSERT INTO executions (id, request_id, position, model, status, created_at, think)"
            " VALUES (?,?,?,?,'queued',?,?)",
            (execution_id, request_id, position, model, _now(), think),
        )
        return self.get_execution(execution_id)[0]

    def add_decision(self, request_id: str, draft: RoutingDraft, *, shadow: bool = False) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO routing_decisions (request_id, shadow, category, complexity,"
            " selected_model, fallback_model, confidence, explanation, escalate, escalation_reason,"
            " decided_by, candidates, signals, llm, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                request_id,
                shadow,
                draft.category,
                draft.complexity,
                draft.selected_model,
                draft.fallback_model,
                draft.confidence,
                draft.explanation,
                draft.escalate,
                draft.escalation_reason,
                draft.decided_by,
                json.dumps(draft.candidates),
                draft.signals.model_dump_json() if draft.signals else None,
                json.dumps(draft.llm) if draft.llm else None,
                _now(),
            ),
        )

    def mark_fallback_used(self, request_id: str) -> None:
        self._db.execute(
            "UPDATE routing_decisions SET fallback_used = 1 WHERE request_id = ?", (request_id,)
        )

    def fail_unrouted(self, request_ids: list[str] | None = None) -> int:
        """Close automatic requests whose routing never finished, so nothing waits on them."""
        where = (
            "r.routing_mode = 'automatic' AND r.parent_execution_id IS NULL"
            " AND NOT EXISTS (SELECT 1 FROM routing_decisions d WHERE d.request_id = r.id)"
        )
        params: list = []
        if request_ids is not None:
            if not request_ids:
                return 0
            where += f" AND r.id IN ({','.join('?' * len(request_ids))})"
            params = list(request_ids)
        rows = self._db.execute(f"SELECT r.id FROM requests r WHERE {where}", params).fetchall()
        for row in rows:
            self._db.execute(
                "INSERT INTO routing_decisions (request_id, category, complexity, confidence,"
                " explanation, escalate, decided_by, created_at)"
                " VALUES (?, 'general_qa', 'low', 0, ?, 0, 'none', ?)",
                (row["id"], "Routing was interrupted before a model was chosen.", _now()),
            )
        return len(rows)

    def mark_running(self, execution_id: str) -> None:
        self._db.execute(
            "UPDATE executions SET status='running', started_at=? WHERE id=?",
            (_now(), execution_id),
        )

    def complete_execution(
        self,
        execution_id: str,
        *,
        response: str,
        thinking: str | None,
        queue_wait_ms: float,
        execution_ms: float,
        metrics: dict,
        warnings: list[str],
    ) -> None:
        self._db.execute(
            "UPDATE executions SET status='completed', response=?, thinking=?, finished_at=?,"
            " queue_wait_ms=?, execution_ms=?, metrics=?, warnings=? WHERE id=?",
            (
                response,
                thinking,
                _now(),
                queue_wait_ms,
                execution_ms,
                json.dumps(metrics),
                json.dumps(warnings),
                execution_id,
            ),
        )

    def fail_execution(
        self,
        execution_id: str,
        code: str,
        detail: str,
        *,
        thinking: str | None = None,
        queue_wait_ms: float | None = None,
        execution_ms: float | None = None,
        metrics: dict | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """Record a failure. The optional values keep what the model did
        produce (its reasoning, timings, token counts) when it ran but gave
        no usable answer."""
        self._db.execute(
            "UPDATE executions SET status='failed', error_code=?, error_detail=?, finished_at=?,"
            " thinking=?, queue_wait_ms=?, execution_ms=?, metrics=?,"
            " warnings=COALESCE(?, warnings) WHERE id=?",
            (
                code,
                detail,
                _now(),
                thinking,
                queue_wait_ms,
                execution_ms,
                None if metrics is None else json.dumps(metrics),
                None if warnings is None else json.dumps(warnings),
                execution_id,
            ),
        )

    def requeue_resumable(self, batch_id: str) -> int:
        """Put a batch's interrupted or timed-out executions back in the queue.

        Completed answers and failures caused by the model are left alone.
        """
        marks = ",".join("?" * len(RESUMABLE_CODES))
        cursor = self._db.execute(
            "UPDATE executions SET status='queued', error_code=NULL, error_detail=NULL,"
            " started_at=NULL, finished_at=NULL, thinking=NULL, queue_wait_ms=NULL,"
            " execution_ms=NULL, metrics=NULL, warnings='[]'"
            f" WHERE status='failed' AND error_code IN ({marks})"
            " AND request_id IN (SELECT id FROM requests WHERE batch_id = ?)",
            (*RESUMABLE_CODES, batch_id),
        )
        return cursor.rowcount

    def fail_unfinished(self) -> int:
        """Mark executions left queued or running by a previous process as failed."""
        cursor = self._db.execute(
            "UPDATE executions SET status='failed', error_code='interrupted',"
            " error_detail='The backend stopped before this execution finished', finished_at=?"
            " WHERE status IN ('queued','running')",
            (_now(),),
        )
        tool_cursor = self._db.execute(
            "UPDATE tool_results SET status='failed', error_code='interrupted',"
            " error_detail='The backend stopped before this case finished', finished_at=?"
            " WHERE status IN ('queued','running')",
            (_now(),),
        )
        agent_cursor = self._db.execute(
            "UPDATE agent_results SET status='failed', error_code='interrupted',"
            " error_detail='The backend stopped before this scenario finished', finished_at=?"
            " WHERE status IN ('queued','running')",
            (_now(),),
        )
        return cursor.rowcount + tool_cursor.rowcount + agent_cursor.rowcount

    # ---- tool-calling and agent runs ----
    # Both kinds share one shape: a run of test cases on several models, one
    # result row per (case, model). `kind` picks the tables.

    def create_tool_run(
        self,
        suite: str,
        name: str,
        models: list[str],
        think: bool,
        cases: list[tuple[str, str]],
        *,
        kind: str = "tool",
    ) -> str:
        """Store a run: every (case, model) pair starts queued. `cases` is (id, category)."""
        run_id = _id()
        with self._transaction():
            self._db.execute(
                f"INSERT INTO {_RUNS[kind]} (id, suite, name, models, think, created_at)"
                " VALUES (?,?,?,?,?,?)",
                (run_id, suite, name, json.dumps(models), int(think), _now()),
            )
            position = 0
            for case_id, category in cases:
                for model in models:
                    self._db.execute(
                        f"INSERT INTO {_RESULTS[kind]} (id, run_id, case_id, category, model,"
                        " position, status) VALUES (?,?,?,?,?,?, 'queued')",
                        (_id(), run_id, case_id, category, model, position),
                    )
                    position += 1
        return run_id

    def tool_run(self, run_id: str, *, kind: str = "tool") -> dict | None:
        row = self._db.execute(f"SELECT * FROM {_RUNS[kind]} WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        run = dict(row)
        run["models"] = [m for m in json.loads(run["models"]) if m not in self.hidden]
        run["think"] = bool(run["think"])
        return run

    def list_tool_runs(self, *, kind: str = "tool") -> list[dict]:
        runs = [
            self.tool_run(r["id"], kind=kind)
            for r in self._db.execute(f"SELECT id FROM {_RUNS[kind]} ORDER BY created_at DESC")
        ]
        return [r for r in runs if r]

    def tool_results(
        self, run_id: str, status: str | None = None, *, kind: str = "tool"
    ) -> list[dict]:
        sql = f"SELECT * FROM {_RESULTS[kind]} WHERE run_id = ?"
        params: tuple = (run_id,)
        if status:
            sql += " AND status = ?"
            params = (run_id, status)
        results = []
        for row in self._db.execute(sql + " ORDER BY position", params):
            if row["model"] in self.hidden:
                continue
            item = dict(row)
            for column in ("calls", "checks", "trace", "effects"):
                if isinstance(item.get(column), str):
                    item[column] = json.loads(item[column])
            item["passed"] = None if item["passed"] is None else bool(item["passed"])
            results.append(item)
        return results

    def mark_tool_running(self, result_id: str, *, kind: str = "tool") -> None:
        self._db.execute(f"UPDATE {_RESULTS[kind]} SET status='running' WHERE id = ?", (result_id,))

    def finish_tool_result(
        self,
        result_id: str,
        *,
        calls: list[dict],
        call_format: str,
        content: str,
        thinking: str | None,
        passed: bool,
        reason: str,
        prompt_tokens: int | None,
        output_tokens: int | None,
        queue_wait_ms: float,
        execution_ms: float,
    ) -> None:
        self._db.execute(
            "UPDATE tool_results SET status='completed', calls=?, call_format=?, content=?,"
            " thinking=?, passed=?, reason=?, prompt_tokens=?, output_tokens=?, queue_wait_ms=?,"
            " execution_ms=?, finished_at=? WHERE id = ?",
            (
                json.dumps(calls),
                call_format,
                content,
                thinking,
                int(passed),
                reason,
                prompt_tokens,
                output_tokens,
                queue_wait_ms,
                execution_ms,
                _now(),
                result_id,
            ),
        )

    def finish_agent_result(self, result_id: str, **values) -> None:
        """Record a finished scenario. JSON columns are given as Python objects."""
        for column in ("checks", "trace", "effects"):
            values[column] = json.dumps(values[column], ensure_ascii=False)
        values["passed"] = int(values["passed"])
        columns = ", ".join(f"{name}=?" for name in values)
        self._db.execute(
            f"UPDATE agent_results SET status='completed', {columns}, finished_at=? WHERE id = ?",
            (*values.values(), _now(), result_id),
        )

    def fail_tool_result(
        self, result_id: str, code: str, detail: str, *, kind: str = "tool"
    ) -> None:
        self._db.execute(
            f"UPDATE {_RESULTS[kind]} SET status='failed', error_code=?, error_detail=?,"
            " finished_at=? WHERE id = ?",
            (code, detail, _now(), result_id),
        )

    def requeue_tool_run(self, run_id: str, *, kind: str = "tool") -> int:
        """Queue again what an interrupted run left undone: cases that never
        finished and failures caused by the environment, not the model."""
        marks = ",".join("?" * len(RESUMABLE_CODES))
        cursor = self._db.execute(
            f"UPDATE {_RESULTS[kind]} SET status='queued', error_code=NULL, error_detail=NULL,"
            " finished_at=NULL WHERE run_id = ? AND (status = 'queued' OR"
            f" (status='failed' AND error_code IN ({marks})))",
            (run_id, *RESUMABLE_CODES),
        )
        return cursor.rowcount

    # ---- the model lineup ----

    def set_hidden(self, model: str, hidden: bool) -> None:
        """Take a model out of the lineup, or bring it back. Nothing is deleted."""
        if hidden:
            self._db.execute(
                "INSERT OR IGNORE INTO hidden_models (model, hidden_at) VALUES (?,?)",
                (model, _now()),
            )
            self.hidden.add(model)
        else:
            self._db.execute("DELETE FROM hidden_models WHERE model = ?", (model,))
            self.hidden.discard(model)

    def models_with_results(self) -> set[str]:
        return {
            row["model"]
            for row in self._db.execute(
                "SELECT model FROM executions UNION SELECT model FROM tool_results"
                " UNION SELECT model FROM agent_results"
            )
        }

    def batch_coverage(self) -> dict[str, dict[str, dict]]:
        """model -> batch id -> prompts answered, failed, waiting and judged.
        Counts prompts, not attempts, and ignores whether the model is hidden."""
        coverage: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in self._db.execute(
            """
            SELECT e.model, r.batch_id,
              COUNT(DISTINCT CASE WHEN e.status = 'completed' THEN r.id END) AS answered,
              COUNT(DISTINCT CASE WHEN e.status IN ('queued','running') THEN r.id END) AS waiting,
              COUNT(DISTINCT CASE WHEN e.status = 'completed' AND EXISTS
                (SELECT 1 FROM evaluations v WHERE v.execution_id = e.id) THEN r.id END) AS judged,
              COUNT(DISTINCT r.id) AS tried
            FROM executions e JOIN requests r ON r.id = e.request_id
            WHERE r.batch_id IS NOT NULL GROUP BY e.model, r.batch_id
            """
        ):
            coverage[row["model"]][row["batch_id"]] = {
                "answered": row["answered"],
                "waiting": row["waiting"],
                "judged": row["judged"],
                # Tried but never answered (a failure that was not retried successfully).
                "failed": row["tried"] - row["answered"] - row["waiting"],
            }
        return coverage

    def add_model_to_batch(self, batch_id: str, model: str) -> list[str]:
        """Queue `model` on every prompt of the batch it has not been run on, and
        queue again its failures caused by the environment. Returns the request ids."""
        with self._transaction():
            batch = self._db.execute("SELECT models FROM batches WHERE id = ?", (batch_id,))
            models = json.loads(batch.fetchone()["models"])
            if model not in models:
                self._db.execute(
                    "UPDATE batches SET models = ? WHERE id = ?",
                    (json.dumps([*models, model]), batch_id),
                )
            missing = [
                row["id"]
                for row in self._db.execute(
                    "SELECT id FROM requests WHERE batch_id = ? AND NOT EXISTS (SELECT 1 FROM"
                    " executions e WHERE e.request_id = requests.id AND e.model = ?)"
                    " ORDER BY position",
                    (batch_id, model),
                )
            ]
            for request_id in missing:
                position = self._db.execute(
                    "SELECT COUNT(*) FROM executions WHERE request_id = ?", (request_id,)
                ).fetchone()[0]
                self._db.execute(
                    "INSERT INTO executions (id, request_id, position, model, status, created_at)"
                    " VALUES (?,?,?,?,'queued',?)",
                    (_id(), request_id, position, model, _now()),
                )
            marks = ",".join("?" * len(RESUMABLE_CODES))
            requeued = [
                row["request_id"]
                for row in self._db.execute(
                    "UPDATE executions SET status='queued', error_code=NULL, error_detail=NULL,"
                    " started_at=NULL, finished_at=NULL, thinking=NULL, queue_wait_ms=NULL,"
                    " execution_ms=NULL, metrics=NULL, warnings='[]'"
                    f" WHERE model = ? AND status='failed' AND error_code IN ({marks})"
                    " AND request_id IN (SELECT id FROM requests WHERE batch_id = ?)"
                    " RETURNING request_id",
                    (model, *RESUMABLE_CODES, batch_id),
                )
            ]
        return list(dict.fromkeys(missing + requeued))

    def tool_coverage(self, *, kind: str = "tool") -> dict[str, dict[str, dict]]:
        """model -> tool run id -> cases passed, wrong, failed and waiting."""
        coverage: dict[str, dict[str, dict]] = defaultdict(dict)
        for row in self._db.execute(
            f"""
            SELECT model, run_id,
              SUM(status = 'completed' AND passed = 1) AS passed,
              SUM(status = 'completed' AND passed = 0) AS wrong,
              SUM(status = 'failed') AS failed,
              SUM(status IN ('queued','running')) AS waiting
            FROM {_RESULTS[kind]} GROUP BY model, run_id
            """
        ):
            coverage[row["model"]][row["run_id"]] = {
                k: row[k] for k in ("passed", "wrong", "failed", "waiting")
            }
        return coverage

    def add_model_to_tool_run(
        self, run_id: str, model: str, cases: list[tuple[str, str]], *, kind: str = "tool"
    ) -> int:
        """Queue `model` on every case of the run it lacks, plus its failures caused
        by the environment. Returns how many cases were queued."""
        with self._transaction():
            run = self._db.execute(f"SELECT models FROM {_RUNS[kind]} WHERE id = ?", (run_id,))
            models = json.loads(run.fetchone()["models"])
            if model not in models:
                self._db.execute(
                    f"UPDATE {_RUNS[kind]} SET models = ? WHERE id = ?",
                    (json.dumps([*models, model]), run_id),
                )
            have = {
                row["case_id"]
                for row in self._db.execute(
                    f"SELECT case_id FROM {_RESULTS[kind]} WHERE run_id = ? AND model = ?",
                    (run_id, model),
                )
            }
            position = self._db.execute(
                f"SELECT COALESCE(MAX(position), -1) + 1 FROM {_RESULTS[kind]} WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            added = 0
            for case_id, category in cases:
                if case_id in have:
                    continue
                self._db.execute(
                    f"INSERT INTO {_RESULTS[kind]} (id, run_id, case_id, category, model,"
                    " position, status) VALUES (?,?,?,?,?,?, 'queued')",
                    (_id(), run_id, case_id, category, model, position),
                )
                position += 1
                added += 1
            marks = ",".join("?" * len(RESUMABLE_CODES))
            requeued = self._db.execute(
                f"UPDATE {_RESULTS[kind]} SET status='queued', error_code=NULL, error_detail=NULL,"
                " finished_at=NULL WHERE run_id = ? AND model = ? AND status='failed'"
                f" AND error_code IN ({marks})",
                (run_id, model, *RESUMABLE_CODES),
            ).rowcount
        return added + requeued

    def latest_tool_run(self, suite: str, *, kind: str = "tool") -> str | None:
        row = self._db.execute(
            f"SELECT id FROM {_RUNS[kind]} WHERE suite = ? ORDER BY created_at DESC LIMIT 1",
            (suite,),
        ).fetchone()
        return row["id"] if row else None

    # ---- reading ----

    def _evaluations_for(self, execution_ids: list[str]) -> dict[str, list[Evaluation]]:
        found: dict[str, list[Evaluation]] = defaultdict(list)
        if not execution_ids:
            return found
        marks = ",".join("?" * len(execution_ids))
        rows = self._db.execute(
            f"SELECT * FROM evaluations WHERE execution_id IN ({marks}) ORDER BY created_at",
            execution_ids,
        )
        for row in rows:
            data = dict(row)
            for key in ("scores", "strengths", "weaknesses", "recommendations", "critical_issues"):
                data[key] = json.loads(data[key])
            found[row["execution_id"]].append(Evaluation(**data))
        return found

    def _executions_for(self, request_ids: list[str]) -> dict[str, list[Execution]]:
        found: dict[str, list[Execution]] = defaultdict(list)
        if not request_ids:
            return found
        marks = ",".join("?" * len(request_ids))
        rows = self._db.execute(
            f"SELECT * FROM executions WHERE request_id IN ({marks}) ORDER BY position",
            request_ids,
        ).fetchall()
        ids = [row["id"] for row in rows]
        evaluations = self._evaluations_for(ids)
        follow_ups = {
            r["parent_execution_id"]: r["n"]
            for r in self._db.execute(
                "SELECT parent_execution_id, COUNT(*) AS n FROM requests"
                f" WHERE parent_execution_id IN ({','.join('?' * len(ids))})"
                " GROUP BY parent_execution_id",
                ids,
            )
        }
        for row in rows:
            if row["model"] in self.hidden:
                continue
            data = dict(row)
            data.pop("position")
            data["think"] = None if data["think"] is None else bool(data["think"])
            data["metrics"] = json.loads(data["metrics"]) if data["metrics"] else None
            data["warnings"] = json.loads(data["warnings"])
            found[row["request_id"]].append(
                Execution(
                    **data,
                    evaluations=evaluations.get(row["id"], []),
                    follow_up_count=follow_ups.get(row["id"], 0),
                )
            )
        return found

    def _decisions_for(self, request_ids: list[str]) -> dict[str, RoutingDecision]:
        if not request_ids:
            return {}
        marks = ",".join("?" * len(request_ids))
        found = {}
        for row in self._db.execute(
            f"SELECT * FROM routing_decisions WHERE request_id IN ({marks})", request_ids
        ):
            data = dict(row)
            data["candidates"] = json.loads(data["candidates"])
            data["signals"] = json.loads(data["signals"]) if data["signals"] else None
            data["llm"] = json.loads(data["llm"]) if data["llm"] else None
            found[row["request_id"]] = RoutingDecision(**data)
        return found

    def _requests(self, where: str, params: tuple, order: str) -> list[RequestRecord]:
        rows = self._db.execute(
            f"SELECT * FROM requests WHERE {where} ORDER BY {order}", params
        ).fetchall()
        executions = self._executions_for([row["id"] for row in rows])
        decisions = self._decisions_for([row["id"] for row in rows])
        only_hidden = self._only_hidden([row["id"] for row in rows], executions)
        records = []
        for row in rows:
            if row["id"] in only_hidden:
                continue
            data = dict(row)
            data.pop("position")
            data["think"] = None if data["think"] is None else bool(data["think"])
            data["history"] = json.loads(data["history"])
            records.append(
                RequestRecord(
                    **data,
                    executions=executions.get(row["id"], []),
                    routing=decisions.get(row["id"]),
                )
            )
        return records

    def _only_hidden(self, request_ids: list[str], visible: dict) -> set[str]:
        """Requests whose every answer came from a hidden model: shown nowhere."""
        candidates = [r for r in request_ids if not visible.get(r)]
        if not self.hidden or not candidates:
            return set()
        marks = ",".join("?" * len(candidates))
        hidden = ",".join("?" * len(self.hidden))
        return {
            row["request_id"]
            for row in self._db.execute(
                f"SELECT DISTINCT request_id FROM executions WHERE request_id IN ({marks})"
                f" AND model IN ({hidden})",
                (*candidates, *self.hidden),
            )
        }

    def get_request(self, request_id: str) -> RequestRecord | None:
        found = self._requests("id = ?", (request_id,), "created_at")
        return found[0] if found else None

    def list_requests(
        self, *, limit: int = 50, offset: int = 0, include_batches: bool = False
    ) -> list[RequestRecord]:
        # Follow-up messages are shown inside their conversation, not as separate rows.
        where = "parent_execution_id IS NULL"
        if not include_batches:
            where += " AND batch_id IS NULL"
        return self._requests(where, (limit, offset), "created_at DESC, position LIMIT ? OFFSET ?")

    def requests_since(self, since: datetime | None = None) -> list[RequestRecord]:
        """Every request (single, batch and follow-up) created at or after `since`."""
        if since is None:
            return self._requests("1 = 1", (), "created_at")
        return self._requests("created_at >= ?", (since.isoformat(),), "created_at")

    def batch_requests(self, batch_id: str) -> list[RequestRecord]:
        return self._requests("batch_id = ?", (batch_id,), "position")

    def get_execution(self, execution_id: str) -> tuple[Execution, RequestRecord] | None:
        row = self._db.execute(
            "SELECT request_id FROM executions WHERE id = ?", (execution_id,)
        ).fetchone()
        if row is None:
            return None
        request = self.get_request(row["request_id"])
        execution = next((e for e in request.executions if e.id == execution_id), None)
        return None if execution is None else (execution, request)

    def ancestors(self, execution_id: str) -> list[tuple[RequestRecord, Execution]]:
        """The conversation from its first prompt down to this answer, oldest first."""
        chain: list[tuple[RequestRecord, Execution]] = []
        current: str | None = execution_id
        while current:
            found = self.get_execution(current)
            if found is None:
                break
            execution, request = found
            chain.append((request, execution))
            current = request.parent_execution_id
        return chain[::-1]

    def newest_follow_up(self, execution_id: str) -> tuple[RequestRecord, Execution] | None:
        row = self._db.execute(
            "SELECT id FROM requests WHERE parent_execution_id = ?"
            " ORDER BY created_at DESC LIMIT 1",
            (execution_id,),
        ).fetchone()
        if row is None:
            return None
        request = self.get_request(row["id"])
        return request, request.executions[0]

    def _batch_summaries(self, where: str, params: tuple) -> list[BatchSummary]:
        batches = self._db.execute(
            f"SELECT * FROM batches WHERE {where} ORDER BY created_at DESC", params
        ).fetchall()
        summaries = []
        hidden = tuple(self.hidden)
        not_hidden = f" AND e.model NOT IN ({','.join('?' * len(hidden))})" if hidden else ""
        for batch in batches:
            counts = dict.fromkeys(("queued", "running", "completed", "failed"), 0)
            for row in self._db.execute(
                "SELECT e.status, COUNT(*) AS n FROM executions e JOIN requests r"
                f" ON r.id = e.request_id WHERE r.batch_id = ?{not_hidden} GROUP BY e.status",
                (batch["id"], *hidden),
            ):
                counts[row["status"]] = row["n"]
            request_count = self._db.execute(
                "SELECT COUNT(*) FROM requests WHERE batch_id = ?", (batch["id"],)
            ).fetchone()[0]
            evaluated = self._db.execute(
                "SELECT COUNT(DISTINCT v.execution_id) FROM evaluations v"
                " JOIN executions e ON e.id = v.execution_id"
                f" JOIN requests r ON r.id = e.request_id WHERE r.batch_id = ?{not_hidden}",
                (batch["id"], *hidden),
            ).fetchone()[0]
            summaries.append(
                BatchSummary(
                    id=batch["id"],
                    name=batch["name"],
                    created_at=batch["created_at"],
                    models=[m for m in json.loads(batch["models"]) if m not in self.hidden],
                    request_count=request_count,
                    counts=counts,
                    evaluated_count=evaluated,
                )
            )
        return summaries

    def list_batches(self) -> list[BatchSummary]:
        return self._batch_summaries("1=1", ())

    def get_batch(self, batch_id: str) -> BatchSummary | None:
        found = self._batch_summaries("id = ?", (batch_id,))
        return found[0] if found else None

    # ---- evaluations ----

    def missing_executions(self, execution_ids: list[str]) -> list[str]:
        marks = ",".join("?" * len(execution_ids))
        known = {
            row["id"]
            for row in self._db.execute(
                f"SELECT id FROM executions WHERE id IN ({marks})", execution_ids
            )
        }
        return [i for i in execution_ids if i not in known]

    def add_evaluations(
        self, items: list[EvaluationInput], *, weights: dict[str, float], rubric_version: str
    ) -> list[str]:
        """Store evaluations from any judge. All are stored, or none.

        `weights` and `rubric_version` come from the current rubric; the
        version applies to items that do not state their own.
        """
        ids = []
        with self._transaction():
            for item in items:
                evaluation_id = _id()
                ids.append(evaluation_id)
                self._db.execute(
                    "INSERT INTO evaluations (id, execution_id, judge, rubric_version, status,"
                    " scores, overall_score, summary, strengths, weaknesses, recommendations,"
                    " critical_issues, created_at, task_type, domain)"
                    " VALUES (?,?,?,?,'completed',?,?,?,?,?,?,?,?,?,?)",
                    (
                        evaluation_id,
                        item.execution_id,
                        item.judge,
                        item.rubric_version or rubric_version,
                        item.scores.model_dump_json(),
                        overall_score(item.scores, weights),
                        item.summary,
                        json.dumps(item.strengths),
                        json.dumps(item.weaknesses),
                        json.dumps(item.recommendations),
                        json.dumps(item.critical_issues),
                        _now(),
                        item.task_type,
                        item.domain,
                    ),
                )
        return ids
