"""Produces a routing decision: quick rules first, the classifier model only when needed."""

from llm_gateway.catalog import ModelCatalog
from llm_gateway.execution import ExecutionQueue
from llm_gateway.providers.base import ModelInfo, ModelProvider, ProviderError
from llm_gateway.routing import classifier, policy, signals
from llm_gateway.routing.registry import Registry
from llm_gateway.routing.types import COMPLEXITIES, RoutingDraft, SignalReport

LABELS = {
    "general_qa": "general Q&A",
    "classification": "classification",
    "extraction": "information extraction",
    "structured_output": "structured output",
    "reasoning": "reasoning and analysis",
    "coding": "coding",
    "medical": "medical-document understanding",
}

# Confidence is built from how the decision was reached. These are starting
# values; Phase 6 compares them with evaluation scores.
BASE_CONFIDENCE = 0.35
CLARITY_WEIGHT = 0.6  # clear rules alone reach 0.95
SAME_MODEL_FLOOR = 0.7  # rules unsure between two categories that pick the same model
LLM_AGREES = 0.85
LLM_DECIDES = 0.6  # rules unsure, classifier chose a different category
LLM_UNAVAILABLE_CAP = 0.5


class Router:
    def __init__(
        self,
        registry: Registry,
        provider: ModelProvider,
        catalog: ModelCatalog,
        queue: ExecutionQueue,
        *,
        use_llm: bool = True,
        clarity_threshold: float = 0.4,
    ) -> None:
        self.registry = registry
        self._provider = provider
        self._catalog = catalog
        self._queue = queue
        self.use_llm = use_llm
        self._clarity_threshold = clarity_threshold

    async def _availability(self) -> tuple[dict[str, ModelInfo] | None, set[str]]:
        try:
            installed = {m.name: m for m in await self._catalog.list()}
            loaded = {m.name for m in await self._provider.list_loaded()}
        except ProviderError:
            return None, set()
        return installed, loaded

    async def route(
        self,
        prompt: str,
        source_material: str | None = None,
        system: str | None = None,
        *,
        use_llm: bool | None = None,
    ) -> RoutingDraft:
        """Decide which model should handle a request. Never raises."""
        report = signals.analyze(prompt, source_material, system)
        installed, loaded = await self._availability()

        def pick(category: str, complexity: str) -> policy.Selection:
            return policy.select(
                self.registry,
                category,
                complexity,
                installed=installed,
                loaded=loaded,
                estimated_tokens=report.estimated_tokens,
            )

        category, complexity = report.category, report.complexity
        confidence = BASE_CONFIDENCE + CLARITY_WEIGHT * report.clarity
        decided_by = "signals"
        llm_info: dict | None = None
        why: list[str] = [self._signal_sentence(report)]

        unsure = report.clarity < self._clarity_threshold
        if unsure and report.runner_up:
            # Being unsure only matters if the two candidate categories lead to different models.
            first = pick(report.category, complexity).model
            second = pick(report.runner_up, complexity).model
            if first and first == second:
                unsure = False
                confidence = max(confidence, SAME_MODEL_FLOOR)
                why.append(
                    f"It could also be {LABELS[report.runner_up]}, but both lead to the same model."
                )

        if unsure:
            wanted = self.use_llm if use_llm is None else use_llm
            llm_info, result = await self._ask_classifier(
                wanted, installed, prompt, source_material
            )
            if result is None:
                decided_by = "signals_fallback"
                confidence = min(confidence, LLM_UNAVAILABLE_CAP)
                why.append(
                    "The quick rules were unsure and the classifier model was not used "
                    f"({llm_info['error']})."
                )
            else:
                decided_by = "llm"
                agrees = result.category == report.category and report.scores
                confidence = LLM_AGREES if agrees else LLM_DECIDES
                category = result.category
                # Take the higher complexity estimate: escalation advice errs on the careful side.
                complexity = max(complexity, result.complexity, key=COMPLEXITIES.index)
                why.append(
                    f"The quick rules were unsure, so {llm_info['model']} classified it as "
                    f"{LABELS[category]}" + (", agreeing with the rules." if agrees else ".")
                )

        selection = pick(category, complexity)
        why.append(self._complexity_sentence(complexity, report))
        why.append(self._selection_sentence(selection, category, complexity))
        if self.registry.error:
            why.append(
                "The model registry file could not be loaded, so no model profiles were used."
            )
            confidence = min(confidence, LLM_UNAVAILABLE_CAP)
        if selection.model is None:
            confidence = 0.0

        return RoutingDraft(
            category=category,
            complexity=complexity,
            selected_model=selection.model,
            fallback_model=selection.fallback,
            confidence=round(confidence, 2),
            explanation=" ".join(why),
            escalate=selection.escalate,
            escalation_reason=selection.escalation_reason,
            decided_by=decided_by,
            candidates=selection.candidates,
            signals=report,
            llm=llm_info,
        )

    async def _ask_classifier(self, wanted, installed, prompt, source_material):
        name = self.registry.classifier_model
        if not wanted:
            return {"error": "classifier switched off"}, None
        if not name:
            return {"error": "no classifier model configured"}, None
        if installed is None or name not in installed:
            return {"model": name, "error": f"{name} is not available"}, None
        try:
            result, elapsed_ms = await classifier.classify(
                self._queue, installed[name], prompt, source_material
            )
        except classifier.ClassifierUnavailable as exc:
            return {"model": name, "error": str(exc)}, None
        return {"model": name, "elapsed_ms": elapsed_ms, **result.model_dump()}, result

    @staticmethod
    def _signal_sentence(report: SignalReport) -> str:
        if not report.scores:
            return "No quick rule matched this request."
        found = ", ".join(report.matched.get(report.category, []))
        return f"The quick rules point to {LABELS[report.category]} ({found})."

    @staticmethod
    def _complexity_sentence(complexity: str, report: SignalReport) -> str:
        reasons = "; ".join(report.complexity_reasons)
        return f"Complexity looks {complexity}" + (f" ({reasons})." if reasons else ".")

    @staticmethod
    def _selection_sentence(selection: policy.Selection, category: str, complexity: str) -> str:
        if selection.model is None:
            return "No local model is available to run it."
        text = (
            f"{selection.model} is the first choice for {complexity}-complexity {LABELS[category]}"
        )
        if selection.fallback:
            text += f", with {selection.fallback} as the fallback"
        text += "."
        if selection.notes:
            text += " Note: " + "; ".join(selection.notes) + "."
        return text
