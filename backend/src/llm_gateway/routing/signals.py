"""Quick, deterministic classification of a request. No model is called.

Each rule that matches adds its weight to one category. The category with the
highest total wins. Rules are deliberately readable: to change how requests
are classified, edit the tables below and re-run the labelled-prompt test.
"""

import re

from llm_gateway.routing.types import CATEGORIES, SignalReport

CHARS_PER_TOKEN = 4
# Total weight at which evidence for a category counts as strong.
STRONG_SCORE = 3.0
CATEGORY_CAP = 6.0
SPECIFIC_EVIDENCE = 1.5
GENERAL_DISCOUNT = 0.4


def _rule(category: str, weight: float, name: str, pattern: str, where: str = "prompt"):
    return category, weight, name, re.compile(pattern, re.IGNORECASE | re.MULTILINE), where


# `where`: "prompt" = the instruction only; "all" = the instruction plus any source material.
RULES = [
    # ---- coding ----
    _rule("coding", 3.0, "code block", r"```", "all"),
    _rule(
        "coding",
        2.0,
        "code syntax",
        r"\bdef \w+\(|\bfunction \w+\(|\bclass \w+\s*[:({]|#include\s*<|^\s*import \w+|^\s*from \w+ import"
        r"|console\.log|\bSELECT\b.+\bFROM\b|=>\s*[{(]|public static|\bprintf?\(|</\w+>|\bfor \w+ in \w+",
        "all",
    ),
    _rule(
        "coding",
        2.5,
        "coding task",
        r"\b(write|create|implement|build|generate|give me)\b[^.?!\n]{0,60}\b(function|script|program|class|method"
        r"|query|regex|regular expression|unit tests?|test cases?|algorithm|api|endpoint|code|snippet|one-liner|loop)\b"
        r"|\b(debug|refactor|code review|review (this|my|the) (code|function|pull request|pr)|fix (this|the|my)"
        r" (bug|error|code|function)|stack trace|traceback|syntax error|compil(e|er|ation) error|segfault"
        r"|null pointer|time complexity|big[- ]o)\b",
    ),
    _rule(
        "coding",
        2.5,
        "error message",
        r"\b\w*(Error|Exception)\b:|\b(TypeError|ValueError|KeyError|IndexError|NameError|AttributeError|ImportError"
        r"|NullPointerException|SyntaxError|ReferenceError|segmentation fault|undefined is not a function"
        r"|exit code \d+)\b|\b(in|with) my (script|code|program|function|app)\b",
        "all",
    ),
    _rule(
        "coding",
        1.5,
        "programming concept",
        r"\b(linked list|binary (search|tree)|hash ?(map|table)|recursion|recursive|data structures?|sorting algorithm"
        r"|async/await|callbacks?|pointer|array|dictionary|stack overflow|variable|closure|decorator|generics"
        r"|inheritance|unit tests?|pytest|compile|runtime|api call|json parsing|database (index|query|schema))\b",
    ),
    _rule(
        "coding",
        1.5,
        "programming language or tool",
        r"\b(python|javascript|typescript|java|c\+\+|c#|rust|golang|kotlin|swift|php|ruby|sql|bash|shell script"
        r"|powershell|regex|html|css|react|django|flask|fastapi|node\.?js|docker|kubernetes|git|pandas|numpy)\b",
    ),
    # ---- medical (narrow: clinical content, not casual health questions) ----
    _rule(
        "medical",
        3.0,
        "clinical document",
        r"\b(discharge summary|clinical notes?|progress notes?|radiology report|pathology report|lab(oratory)? (results?|report)"
        r"|history of present illness|chief complaint|differential diagnosis|medical records?|case report"
        r"|icd-?10|treatment plan|physical exam(ination)?|vital signs|past medical history)\b",
        "all",
    ),
    _rule(
        "medical",
        2.5,
        "patient description",
        r"\b\d{1,3}[- ]year[- ]old\b|\bpatient (is|was|presents|presented|has|had|with|reports|denies|complains)\b"
        r"|\b(presents|presented) with\b|\bpt\.? (c/o|presents)\b",
        "all",
    ),
    _rule(
        "medical",
        1.5,
        "clinical measurement or dosage",
        r"\b\d+(\.\d+)?\s?(mg|mcg|µg|mmhg|mg/dl|mmol/l|iu|bpm)\b|\b(hba1c|creatinine|hemoglobin|haemoglobin|troponin"
        r"|wbc|platelets?|ecg|ekg|mri|ct scan|x-ray|biopsy|ultrasound|bp \d+/\d+|spo2|egfr)\b",
        "all",
    ),
    _rule(
        "medical",
        1.0,
        "medical vocabulary",
        r"\b(diagnos(is|ed|e)|symptoms?|prescri(bed|ption)|medications?|dosage|contraindicat\w+|side effects?"
        r"|hypertension|diabetes|pneumonia|infection|tumou?r|carcinoma|chronic|acute|antibiotics?|clinical)\b",
        "all",
    ),
    # ---- extraction ----
    _rule(
        "extraction",
        2.5,
        "extraction verb",
        r"\b(extract|pull out|pick out)\b|\b(list|find|identify|get|give me|collect) (all|every|each|the)\b[^.?!\n]{0,40}"
        r"\b(names?|dates?|emails?|addresses|phone numbers?|numbers?|amounts?|prices?|entities|people|companies"
        r"|organi[sz]ations|locations?|places|urls?|links|keywords|key (points|facts|terms)|action items|deadlines"
        r"|skills|ingredients|citations|references)\b|\bnamed entit"
        r"|\bwhat (are|is|were|was) the (key |main )?(action items|deadlines?|dates?|names?|amounts?|findings|figures"
        r"|terms|parties|notice period|price|total)\b",
    ),
    _rule(
        "extraction",
        1.0,
        "refers to given text",
        r"\b(from|in) (the|this) (\w+ )?(following|text|document|email|passage|article|paragraph|transcript|message"
        r"|invoice|resume|cv|contract|report|listing|snippet|clause|below|above)\b"
        r"|\b(mentioned|stated|listed) (in|below|above)\b|\b(the|this) \w+( \w+)? (below|above)\b",
    ),
    # ---- classification ----
    _rule(
        "classification",
        2.5,
        "classification verb",
        r"\b(classify|categori[sz]e|label (this|the|each|these|it)|tag (this|the|each|these)|which (category|class|label|type)"
        r"|assign (a|an|one|the) (category|label|class|tag|priority)|sentiment|is (this|the following|it) (spam|positive|negative"
        r"|toxic|urgent|fake|sarcastic)|spam or (not|ham)|triage)\b",
    ),
    _rule(
        "classification",
        1.5,
        "choice among labels",
        r"\b(positive|negative),? (or|/) (negative|neutral|positive)\b|\bone of( the)?( following)?:?\s*[\w\- ]+,\s*[\w\- ]+"
        r"|\b(categories|labels|classes|options) ?(are)?:\s*\w+|\b(true or false|yes or no)\b"
        r"|\b(whether|if) (this|the|it|each)\b[^.?!\n]{0,60}\bis (a |an )?[\w-]+ or (a |an )?[\w-]+",
    ),
    # ---- structured output ----
    _rule(
        "structured_output",
        3.0,
        "generate structured data",
        r"\b(generate|create|produce|write|output|return|give me|make|build|convert|format|turn)\b[^.?!\n]{0,60}"
        r"\b(json|yaml|csv|xml|toml|markdown table|table|schema|spreadsheet)\b",
    ),
    _rule(
        "structured_output",
        2.0,
        "structured format",
        r"\b(json|yaml|csv|xml|toml)\b|\bmarkdown table\b|\b(as|in) (a|an) table\b|\bkey[- ]value\b|\bjson schema\b",
    ),
    _rule(
        "structured_output",
        1.0,
        "format requirement",
        r"\b(in|using|following|with) (the|this) (following )?(format|structure|schema|template)\b|\b(fields|columns|keys):",
    ),
    # ---- reasoning ----
    _rule(
        "reasoning",
        2.5,
        "reasoning task",
        r"\b(step[- ]by[- ]step|prove|proof|derive|logic puzzle|riddle|trade-?offs?|pros and cons|root causes?|second-order"
        r"|think (through|carefully)|reason (through|about)|what would happen if|implications of|probability"
        r"|argue (for|against)|critique|counter-?argument|deduce|infer|hypothes[ie]s)\b",
    ),
    _rule(
        "reasoning",
        1.5,
        "analysis verb",
        r"\b(analy[sz]e|analysis|compare|contrast|evaluate|assess|justify|weigh|recommend(ation)?|should (i|we|they)"
        r"|which (is|would be|option is) (better|best)|explain why|why (does|do|did|is|are|would|should|might)"
        r"|how many|how long (will|would|does) it take|decide|strategy|plan (for|to))\b",
    ),
    _rule(
        "reasoning",
        2.0,
        "word problem",
        r"(?s)\d.*\d.*\b(how (much|many|long|far|fast|old)|when (does|will|do|did)|what (time|speed|distance|percentage"
        r"|fraction)|at what)\b|\b(how (much|many|long|far|fast|old)|when (does|will))\b.*\d.*\d",
    ),
    # ---- general Q&A ----
    _rule(
        "general_qa",
        2.0,
        "plain question or writing task",
        r"\A\s*(who|what|when|where|which|how|is|are|was|were|does|do|did|can|could|explain|describe|define|tell me"
        r"|summari[sz]e|translate|rewrite|paraphrase|proofread|give me (a|an|some)|suggest|name|write (a|an|me)"
        r" ?(short|brief|polite|formal|friendly|quick)? ?(email|poem|story|essay|paragraph|letter|message|note|speech|bio"
        r"|summary|description|caption|tweet|post|haiku|limerick|joke|title|tagline|cover letter|toast|blog))\b",
    ),
    _rule("general_qa", 1.0, "question mark", r"\?\s*\Z"),
    _rule(
        "general_qa",
        1.5,
        "writing or language task",
        r"\b(summari[sz]e|summary of|translate|rewrite|paraphrase|proofread|grammar|tone|simplify|shorten|in simple terms"
        r"|for a (beginner|child|five[- ]year[- ]old)|eli5)\b",
    ),
]

DEPTH_CUES = re.compile(
    r"\b(step[- ]by[- ]step|in detail|detailed|comprehensive|thorough(ly)?|in[- ]depth|design|architecture|prove|derive"
    r"|optimi[sz]e|trade-?offs?|multi-?step|end-to-end|from scratch|production[- ]ready|edge cases|scalab\w+"
    r"|concurren\w+|distributed|formal(ly)?|rigorous(ly)?|exhaustive)\b",
    re.IGNORECASE,
)
SIMPLE_CUES = re.compile(
    r"\b(one word|single word|briefly|short answer|yes or no|true or false|in one sentence|one-liner|number only"
    r"|just the (answer|number|name)|quick(ly)?|tl;?dr)\b",
    re.IGNORECASE,
)
LIST_ITEM = re.compile(r"^\s*(\d+[.)]|[-*•])\s+\S", re.MULTILINE)


def estimate_tokens(*texts: str | None) -> int:
    return sum(len(t or "") for t in texts) // CHARS_PER_TOKEN


def _complexity(prompt: str, tokens: int, category: str) -> tuple[str, list[str]]:
    points = 0
    reasons = []
    if tokens > 1500:
        points += 2
        reasons.append(f"long input (about {tokens} tokens)")
    elif tokens > 400:
        points += 1
        reasons.append(f"medium-length input (about {tokens} tokens)")
    parts = max(prompt.count("?"), len(LIST_ITEM.findall(prompt)))
    if parts >= 3:
        points += 1
        reasons.append(f"{parts} separate questions or requirements")
    depth = {m.group(0).lower() for m in DEPTH_CUES.finditer(prompt)}
    if depth:
        points += min(2, len(depth))
        reasons.append("asks for depth: " + ", ".join(sorted(depth)[:4]))
    if category == "reasoning":
        points += 1
        reasons.append("reasoning task")
    simple = SIMPLE_CUES.search(prompt)
    if simple:
        points -= 1
        reasons.append(f'asks for a short answer ("{simple.group(0).lower()}")')
    level = "low" if points <= 0 else "medium" if points <= 2 else "high"
    return level, reasons


def analyze(
    prompt: str, source_material: str | None = None, system: str | None = None
) -> SignalReport:
    """Score the request against every rule and estimate its complexity."""
    everything = f"{prompt}\n{source_material or ''}"
    scores = dict.fromkeys(CATEGORIES, 0.0)
    matched: dict[str, list[str]] = {}
    for category, weight, name, pattern, where in RULES:
        if pattern.search(everything if where == "all" else prompt):
            scores[category] = min(CATEGORY_CAP, scores[category] + weight)
            matched.setdefault(category, []).append(name)

    # A question form ("What...", "Explain...") says how the request is phrased, not what
    # it is about. When a specific category has real evidence, the phrasing counts for less.
    specific = max(v for k, v in scores.items() if k != "general_qa")
    if specific >= SPECIFIC_EVIDENCE and scores["general_qa"]:
        scores["general_qa"] = round(scores["general_qa"] * GENERAL_DISCOUNT, 2)

    ranked = sorted(scores.items(), key=lambda item: -item[1])
    (top, top_score), (second, second_score) = ranked[0], ranked[1]
    if top_score == 0:
        # Nothing matched: general Q&A is the least harmful guess, with no confidence in it.
        top, runner_up, clarity = "general_qa", None, 0.0
    else:
        runner_up = second if second_score > 0 else None
        strength = min(1.0, top_score / STRONG_SCORE)
        separation = (top_score - second_score) / top_score
        clarity = round(strength * separation, 3)

    tokens = estimate_tokens(prompt, source_material, system)
    complexity, reasons = _complexity(prompt, tokens, top)
    return SignalReport(
        scores={k: v for k, v in scores.items() if v > 0},
        matched=matched,
        category=top,
        runner_up=runner_up,
        clarity=clarity,
        complexity=complexity,
        complexity_reasons=reasons,
        estimated_tokens=tokens,
    )
