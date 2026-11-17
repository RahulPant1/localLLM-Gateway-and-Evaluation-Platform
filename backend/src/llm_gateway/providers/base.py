"""Provider-neutral interface for running models.

Everything outside `providers/` talks to this interface, never to Ollama's
JSON directly. Adding another provider later means writing one more subclass.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ProviderError(Exception):
    """Base class for provider failures. `code` is a stable machine-readable name."""

    code = "provider_error"


class ProviderUnavailable(ProviderError):
    """The provider service could not be reached."""

    code = "provider_unavailable"


class ModelNotFound(ProviderError):
    """The requested model is not installed."""

    code = "model_not_found"


class GenerationFailed(ProviderError):
    """The provider was reached but the generation did not complete."""

    code = "generation_failed"


class GenerationTimeout(GenerationFailed):
    """The generation exceeded the configured time limit."""

    code = "generation_timeout"


class LowMemory(ProviderError):
    """The machine has too little free memory to run a model safely."""

    code = "low_memory"


class ModelInfo(BaseModel):
    name: str
    size_bytes: int
    family: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    # Empty when the provider could not report details for this model.
    capabilities: list[str] = Field(default_factory=list)
    # The most the model architecture supports.
    max_context_length: int | None = None
    # The context window the gateway actually requests for this model.
    num_ctx: int
    details_available: bool = True

    @property
    def supports_thinking(self) -> bool:
        return "thinking" in self.capabilities


class LoadedModel(BaseModel):
    name: str
    size_bytes: int
    vram_bytes: int
    context_length: int | None = None
    expires_at: datetime | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GenerationRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1)
    system: str | None = None
    # Earlier turns of the conversation, oldest first. Empty for a first prompt.
    history: list[ChatMessage] = Field(default_factory=list)
    # None = think if the model supports it. Thinking text is always kept
    # separate from the answer.
    think: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    # None = the model's own default.
    temperature: float | None = Field(default=None, ge=0, le=2)
    # A JSON schema the answer must follow (used by the routing classifier).
    response_schema: dict | None = None


class GenerationResult(BaseModel):
    model: str
    response: str
    thinking: str | None = None
    done_reason: str | None = None
    num_ctx: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    # Durations reported by the provider, in milliseconds.
    load_ms: float | None = None
    prompt_eval_ms: float | None = None
    generation_ms: float | None = None
    provider_total_ms: float | None = None
    tokens_per_second: float | None = None
    warnings: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: str
    # What the model supplied. Kept as returned, so wrong types stay visible.
    arguments: dict = Field(default_factory=dict)
    # Set when the model sent arguments that are not a JSON object.
    malformed: bool = False


class ToolTurn(BaseModel):
    """One model turn in a tool-calling conversation: text, calls, or both."""

    model: str
    content: str = ""
    thinking: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    done_reason: str | None = None
    num_ctx: int
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    generation_ms: float | None = None
    tokens_per_second: float | None = None


# Called with ("thinking" | "content", text) for each piece of a streamed answer.
OnDelta = Callable[[str, str], None]


class ModelProvider(ABC):
    name: str

    @abstractmethod
    async def version(self) -> str:
        """Return the provider version; raises ProviderUnavailable if unreachable."""

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Return every installed model."""

    @abstractmethod
    async def list_loaded(self) -> list[LoadedModel]:
        """Return models currently held in memory."""

    async def unload(self, model: str) -> None:  # noqa: B027 - optional for providers
        """Remove a model from memory, releasing everything its process holds.
        Providers that cannot do this leave it as a no-op."""

    @abstractmethod
    async def generate(
        self, request: GenerationRequest, model: ModelInfo, on_delta: OnDelta | None = None
    ) -> GenerationResult:
        """Run one generation to completion.

        With `on_delta`, the answer is also reported piece by piece while it
        is generated; the returned result is the same either way.
        """

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        model: ModelInfo,
        *,
        think: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ToolTurn:
        """One turn of a conversation in which the model may call `tools`.

        `messages` use the common chat shape (role/content, plus `tool_calls`
        on assistant turns and role "tool" for results). Providers without
        tool support leave this unimplemented.
        """
        raise GenerationFailed(f"{self.name} does not support tool calling")
