"""Real apps an agent can use, read-only: local files, Gmail and web APIs."""

from llm_gateway.config import Settings
from llm_gateway.connectors.files import LocalFiles
from llm_gateway.connectors.gmail import Gmail
from llm_gateway.connectors.web import WebApis


def build_connectors(settings: Settings) -> dict:
    connectors = [
        LocalFiles(settings.files_root),
        Gmail(settings.gmail_client_path, settings.gmail_token_path),
        WebApis(settings.web_apis),
    ]
    return {c.name: c for c in connectors}
