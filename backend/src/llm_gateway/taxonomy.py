"""Task types and domains a judge assigns to each prompt.

There is no single official standard. The task types follow the task families
that the common public LLM benchmarks are organised around, named in each
entry, so results here can be read next to published ones. The names are
fixed: the database stores them.
"""

from pydantic import BaseModel


class Label(BaseModel):
    name: str
    label: str
    description: str


OTHER = "other"

TASK_TYPES = [
    Label(
        name="knowledge_qa",
        label="Knowledge Q&A",
        description="Factual or conceptual questions answered from general knowledge, "
        "including explanations (benchmark family: MMLU, TriviaQA, Natural Questions).",
    ),
    Label(
        name="reasoning",
        label="Reasoning",
        description="Logic puzzles, multi-step deduction, trade-off and decision analysis "
        "(BIG-Bench Hard, ARC).",
    ),
    Label(
        name="math",
        label="Maths",
        description="Arithmetic, word problems and other quantitative work where a number "
        "or derivation is the answer (GSM8K, MATH).",
    ),
    Label(
        name="coding",
        label="Coding",
        description="Writing, explaining, reviewing or debugging code and queries "
        "(HumanEval, MBPP, SWE-bench).",
    ),
    Label(
        name="summarization",
        label="Summarisation",
        description="Condensing or simplifying a given text (CNN/DailyMail, XSum).",
    ),
    Label(
        name="extraction",
        label="Information extraction",
        description="Pulling named facts, entities, dates or values out of a given text "
        "(CoNLL-2003 NER, SQuAD).",
    ),
    Label(
        name="classification",
        label="Classification",
        description="Assigning labels from a set, including sentiment and spam detection "
        "(GLUE/SST-2, AG News).",
    ),
    Label(
        name="structured_output",
        label="Structured output",
        description="The main difficulty is producing an exact format: JSON, YAML, a table "
        "or a schema (JSON-mode and function-calling tests).",
    ),
    Label(
        name="writing",
        label="Writing",
        description="Composing or rewriting prose: emails, stories, marketing copy, tone "
        "or style changes (the creative-writing category of chat-model leaderboards).",
    ),
    Label(
        name="translation",
        label="Translation",
        description="Converting text between human languages (WMT, FLORES).",
    ),
    Label(name=OTHER, label="Other", description="None of the above fits."),
]

DOMAINS = [
    Label(name="general", label="General", description="Everyday or mixed subject matter."),
    Label(
        name="medical",
        label="Medical",
        description="Clinical notes, lab results, medicines, health questions.",
    ),
    Label(name="legal", label="Legal", description="Contracts, regulation, legal questions."),
    Label(
        name="finance",
        label="Finance",
        description="Accounting, investing, pricing, business numbers.",
    ),
    Label(
        name="science_technology",
        label="Science and technology",
        description="Natural science, engineering, software and IT.",
    ),
    Label(name=OTHER, label="Other", description="A specialist field not listed."),
]

TASK_TYPE_LABELS = {t.name: t.label for t in TASK_TYPES}
DOMAIN_LABELS = {d.name: d.label for d in DOMAINS}


def normalise(value: object, allowed: dict[str, str]) -> str | None:
    """Map a judge's label onto a known name. Unknown labels become "other",
    so one odd label never rejects a whole set of scores."""
    if value is None or not str(value).strip():
        return None
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return name if name in allowed else OTHER


def taxonomy_text() -> str:
    """The lists as they are given to a judge."""
    lines = ["## Task types", ""]
    lines += [f"- `{t.name}`: {t.description}" for t in TASK_TYPES]
    lines += ["", "## Domains", ""]
    lines += [f"- `{d.name}`: {d.description}" for d in DOMAINS]
    return "\n".join(lines)
