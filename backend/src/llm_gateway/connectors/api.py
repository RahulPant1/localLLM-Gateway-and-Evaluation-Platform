"""Endpoints for real-app agent runs from the Playground."""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from llm_gateway.connectors.live import AppUnavailable, LiveAgentRunner
from llm_gateway.providers.base import ModelNotFound
from llm_gateway.schemas import RequestRecord

router = APIRouter(prefix="/api")


class ConnectorInfo(BaseModel):
    name: str
    label: str
    available: bool
    detail: str
    tools: list[str]


class AgentRunCreate(BaseModel):
    prompt: str = Field(min_length=1)
    model: str
    apps: list[str] = Field(min_length=1)
    think: bool = False


def _runner(request: Request) -> LiveAgentRunner:
    return request.app.state.live_agent


@router.get("/connectors")
def list_connectors(request: Request) -> list[ConnectorInfo]:
    """The real apps an agent can use, whether each is set up, and its tools."""
    out = []
    for connector in _runner(request).connectors.values():
        status = connector.status()
        out.append(
            ConnectorInfo(
                name=status.name,
                label=status.label,
                available=status.available,
                detail=status.detail,
                tools=[t["function"]["name"] for t in connector.tools()],
            )
        )
    return out


@router.post("/agent-runs", status_code=201)
async def start_agent_run(body: AgentRunCreate, request: Request) -> RequestRecord:
    """Run a prompt as an agent with read-only tools for the chosen apps.

    Returns the stored request at once; poll GET /api/requests/{id}. The steps
    are in the execution's metrics under "agent" and grow as the run goes.
    """
    if body.model in request.app.state.store.hidden:
        raise HTTPException(409, f"{body.model} is hidden. Restore it in Tests → Models.")
    try:
        return await _runner(request).start(body.prompt, body.model, body.apps, body.think)
    except AppUnavailable as exc:
        raise HTTPException(422, str(exc)) from exc
    except ModelNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
