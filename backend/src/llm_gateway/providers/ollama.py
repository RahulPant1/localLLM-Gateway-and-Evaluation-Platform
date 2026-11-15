"""Ollama implementation of ModelProvider, over Ollama's HTTP API."""

import asyncio
import json

import httpx2

from llm_gateway.providers.base import (
    GenerationFailed,
    GenerationRequest,
    GenerationResult,
    GenerationTimeout,
    LoadedModel,
    ModelInfo,
    ModelNotFound,
    ModelProvider,
    OnDelta,
    ProviderUnavailable,
)

NS_PER_MS = 1_000_000
# Rough average for English text; used only for an early "prompt too long" warning.
CHARS_PER_TOKEN = 4


def _ms(nanoseconds: int | None) -> float | None:
    return None if nanoseconds is None else round(nanoseconds / NS_PER_MS, 1)


def _modelfile_num_ctx(parameters: str | None) -> int | None:
    """Read `num_ctx` from the Modelfile parameter text Ollama returns."""
    for line in (parameters or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "num_ctx" and parts[1].isdigit():
            return int(parts[1])
    return None


def _error_message(response: httpx2.Response) -> str:
    try:
        return str(response.json().get("error", response.text))
    except (json.JSONDecodeError, AttributeError):
        return response.text


class OllamaProvider(ModelProvider):
    name = "ollama"

    def __init__(
        self,
        client: httpx2.AsyncClient,
        *,
        default_num_ctx: int = 8192,
        generation_timeout_seconds: float = 300.0,
    ) -> None:
        # `client` must be created with the Ollama base URL.
        self._client = client
        self._default_num_ctx = default_num_ctx
        self._generation_timeout = generation_timeout_seconds

    async def _request(
        self, method: str, path: str, *, payload: dict | None = None, timeout: float = 10.0
    ) -> httpx2.Response:
        try:
            return await self._client.request(method, path, json=payload, timeout=timeout)
        except httpx2.TimeoutException as exc:
            raise ProviderUnavailable(f"Ollama did not answer {path} within {timeout:g}s") from exc
        except httpx2.HTTPError as exc:
            raise ProviderUnavailable(f"Cannot reach Ollama: {exc}") from exc

    async def _json(self, method: str, path: str, *, payload: dict | None = None) -> dict:
        response = await self._request(method, path, payload=payload)
        if response.status_code != 200:
            raise ProviderUnavailable(
                f"Ollama returned HTTP {response.status_code} for {path}: "
                f"{_error_message(response)}"
            )
        return response.json()

    async def version(self) -> str:
        return str((await self._json("GET", "/api/version")).get("version", "unknown"))

    def _choose_num_ctx(self, modelfile_num_ctx: int | None, max_context: int | None) -> int:
        """A Modelfile value wins; otherwise the gateway default, capped by the model's limit.

        Using one fixed value per model matters: Ollama reloads a model whenever
        the requested context size changes.
        """
        if modelfile_num_ctx:
            return modelfile_num_ctx
        if max_context:
            return min(self._default_num_ctx, max_context)
        return self._default_num_ctx

    async def _describe(self, tag: dict) -> ModelInfo:
        name = tag["name"]
        details = tag.get("details") or {}
        base = {
            "name": name,
            "size_bytes": tag.get("size", 0),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
        }
        try:
            show = await self._json("POST", "/api/show", payload={"model": name})
        except ProviderUnavailable:
            # One model's details failing must not hide the model itself.
            return ModelInfo(
                **base, num_ctx=self._choose_num_ctx(None, None), details_available=False
            )
        max_context = next(
            (v for k, v in (show.get("model_info") or {}).items() if k.endswith(".context_length")),
            None,
        )
        return ModelInfo(
            **base,
            capabilities=show.get("capabilities") or [],
            max_context_length=max_context,
            num_ctx=self._choose_num_ctx(_modelfile_num_ctx(show.get("parameters")), max_context),
        )

    async def list_models(self) -> list[ModelInfo]:
        tags = (await self._json("GET", "/api/tags")).get("models", [])
        models = await asyncio.gather(*(self._describe(tag) for tag in tags))
        return sorted(models, key=lambda m: m.name.lower())

    async def list_loaded(self) -> list[LoadedModel]:
        return [
            LoadedModel(
                name=entry["name"],
                size_bytes=entry.get("size", 0),
                vram_bytes=entry.get("size_vram", 0),
                context_length=entry.get("context_length"),
                expires_at=entry.get("expires_at"),
            )
            for entry in (await self._json("GET", "/api/ps")).get("models", [])
        ]

    def _chat_payload(self, request: GenerationRequest, model: ModelInfo, stream: bool) -> dict:
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(m.model_dump() for m in request.history)
        messages.append({"role": "user", "content": request.prompt})

        options: dict = {"num_ctx": model.num_ctx}
        if request.max_tokens:
            options["num_predict"] = request.max_tokens
        if request.temperature is not None:
            options["temperature"] = request.temperature
        # No `keep_alive`: Ollama's own unload policy is left untouched.
        payload: dict = {
            "model": model.name,
            "messages": messages,
            "stream": stream,
            "options": options,
        }
        if request.response_schema:
            payload["format"] = request.response_schema
        # Sending `think` to a model without that capability is an error, so
        # only thinking-capable models get it. With it set, Ollama returns the
        # reasoning in its own field instead of mixed into the answer.
        if model.supports_thinking:
            payload["think"] = True if request.think is None else request.think
        return payload

    async def generate(
        self, request: GenerationRequest, model: ModelInfo, on_delta: OnDelta | None = None
    ) -> GenerationResult:
        payload = self._chat_payload(request, model, stream=on_delta is not None)
        try:
            if on_delta is None:
                body, content, thinking = await self._chat_once(payload)
            else:
                # httpx applies its timeout to each read; this limits the whole answer.
                async with asyncio.timeout(self._generation_timeout):
                    body, content, thinking = await self._chat_stream(payload, on_delta)
        except httpx2.ConnectError as exc:
            raise ProviderUnavailable(f"Cannot reach Ollama: {exc}") from exc
        except (httpx2.TimeoutException, TimeoutError) as exc:
            raise GenerationTimeout(
                f"{model.name} did not finish within {self._generation_timeout:g}s"
            ) from exc
        except httpx2.HTTPError as exc:
            raise GenerationFailed(f"Connection to Ollama failed mid-generation: {exc}") from exc

        prompt_tokens = body.get("prompt_eval_count")
        output_tokens = body.get("eval_count")
        eval_ns = body.get("eval_duration")
        return GenerationResult(
            model=model.name,
            response=content,
            thinking=thinking or None,
            done_reason=body.get("done_reason"),
            num_ctx=model.num_ctx,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            load_ms=_ms(body.get("load_duration")),
            prompt_eval_ms=_ms(body.get("prompt_eval_duration")),
            generation_ms=_ms(eval_ns),
            provider_total_ms=_ms(body.get("total_duration")),
            tokens_per_second=(
                round(output_tokens / (eval_ns / 1e9), 1) if output_tokens and eval_ns else None
            ),
            warnings=context_warnings(
                request, model.num_ctx, prompt_tokens, output_tokens, body.get("done_reason")
            ),
        )

    @staticmethod
    def _check_status(status_code: int, message: str) -> None:
        if status_code == 404:
            raise ModelNotFound(message)
        if status_code != 200:
            raise GenerationFailed(f"Ollama returned HTTP {status_code}: {message}")

    async def _chat_once(self, payload: dict) -> tuple[dict, str, str]:
        """One complete answer. Returns Ollama's final body, the answer and the thinking."""
        response = await self._client.post(
            "/api/chat", json=payload, timeout=self._generation_timeout
        )
        self._check_status(response.status_code, _error_message(response))
        try:
            body = response.json()
            message = body["message"]
        except (json.JSONDecodeError, KeyError) as exc:
            raise GenerationFailed("Ollama returned an unreadable response") from exc
        return body, message.get("content", ""), message.get("thinking") or ""

    async def _chat_stream(self, payload: dict, on_delta: OnDelta) -> tuple[dict, str, str]:
        """The same answer, read line by line as Ollama produces it."""
        content: list[str] = []
        thinking: list[str] = []
        final: dict | None = None
        async with self._client.stream(
            "POST", "/api/chat", json=payload, timeout=self._generation_timeout
        ) as response:
            if response.status_code != 200:
                await response.aread()
                self._check_status(response.status_code, _error_message(response))
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise GenerationFailed("Ollama returned an unreadable response") from exc
                if chunk.get("error"):
                    raise GenerationFailed(str(chunk["error"]))
                message = chunk.get("message") or {}
                if message.get("thinking"):
                    thinking.append(message["thinking"])
                    on_delta("thinking", message["thinking"])
                if message.get("content"):
                    content.append(message["content"])
                    on_delta("content", message["content"])
                if chunk.get("done"):
                    final = chunk
        if final is None:
            raise GenerationFailed("Ollama ended the answer without a final message")
        return final, "".join(content), "".join(thinking)


def context_warnings(
    request: GenerationRequest,
    num_ctx: int,
    prompt_tokens: int | None,
    output_tokens: int | None,
    done_reason: str | None,
) -> list[str]:
    """Flag likely truncation. Ollama truncates silently, so these are heuristics."""
    warnings = []
    text_length = (
        len(request.prompt)
        + len(request.system or "")
        + sum(len(m.content) for m in request.history)
    )
    if text_length / CHARS_PER_TOKEN > num_ctx:
        warnings.append("prompt_probably_truncated")
    if done_reason == "length":
        warnings.append("output_cut_off")
    if (prompt_tokens or 0) + (output_tokens or 0) >= num_ctx:
        warnings.append("context_window_full")
    return warnings
