"""
Behavioral Fingerprinting
=========================
Computes behavioral feature vectors from dialogue traces, organized by the
behavioral taxonomy:

  D1 — Communication Style   (8 regex/statistical features)
  D2 — Information Disclosure  (3 regex/statistical features)
  D3 — Clarification Behavior (5 regex/statistical features)
  D4 — Error Reaction         (3 regex/statistical features)

``compute_aggregate_dice_alignment`` implements the aggregate-first
Sørensen–Dice score. In the current
pipeline this Dice score is used only as a **diagnostic** — actual
``human_likeness`` comes from a trained RF discriminator on these regex features only
(``discriminator_<domain>_<agent>_user_<user>.pkl`` under
``outputs/reference_data/``, required at evaluator startup; see
``discriminator.py`` / ``evaluator.py``). Averaging
per-episode Dice scores instead of aggregating first would systematically
under-estimate alignment because Dice is non-linear in its inputs; hence the
aggregate-first pattern is preserved here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Feature dimension definitions
# ---------------------------------------------------------------------------

D1_FEATURES: List[str] = [
    "words_per_turn",
    "short_utterance_rate",
    "politeness_rate",
    "formality_rate",
    "acknowledgment_rate",
    "verbosity_cv",
    "repetition_rate",
    "identity_confusion_rate",
]

D2_FEATURES: List[str] = [
    "front_loading_ratio",
    "identifiers_per_turn",
    "opening_length",
]

D3_FEATURES: List[str] = [
    "uncertainty_rate",
    "certainty_rate",
    "pushback_rate",
    "clarification_question_rate",
    "info_seeking_rate",
]

D4_FEATURES: List[str] = [
    "emotional_expression_rate",
    "accusatory_rate",
    "strategy_pivot_rate",
]

REGEX_FEATURES: List[str] = D1_FEATURES + D2_FEATURES + D3_FEATURES + D4_FEATURES
# Full fingerprint = regex D1–D4 only (19 features).
ALL_FEATURES: List[str] = REGEX_FEATURES

DIMENSION_MAP: Dict[str, List[str]] = {
    "D1": D1_FEATURES,
    "D2": D2_FEATURES,
    "D3": D3_FEATURES,
    "D4": D4_FEATURES,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BehavioralFingerprint:
    """A feature vector representing behavioral characteristics of a dialogue."""

    features: Dict[str, float]

    def to_vector(self, feature_order: List[str]) -> np.ndarray:
        return np.array([self.features.get(k, 0.0) for k in feature_order])

    def to_dict(self) -> dict:
        return self.features.copy()


@dataclass
class HumanBehavioralDistribution:
    """Per-feature mean (and optional std) values from real human dialogues.

    ``std`` is the per-feature across-dialogue standard deviation, cached
    alongside the mean so comparison blocks can report ``mean ± std``. It is
    optional for backward compatibility with older caches; callers that need
    it should re-run ``scripts/compute_human_reference.py``.
    """

    mean: Dict[str, float]
    feature_names: List[str]
    n_dialogues: int
    std: Dict[str, float] = None  # type: ignore[assignment]
    source: Optional[str] = None  # e.g. "tau_bench_human"
    domain: Optional[str] = None  # e.g. "retail", "airline"

    def __post_init__(self) -> None:
        if self.std is None:
            self.std = {}

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload: Dict[str, Any] = {
            "mean": self.mean,
            "feature_names": self.feature_names,
            "n_dialogues": self.n_dialogues,
        }
        if self.std:
            payload["std"] = self.std
        if self.source is not None:
            payload["source"] = self.source
        if self.domain is not None:
            payload["domain"] = self.domain
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HumanBehavioralDistribution":
        with open(path) as f:
            d = json.load(f)
        return cls(
            mean=d["mean"],
            feature_names=d["feature_names"],
            n_dialogues=d["n_dialogues"],
            std=d.get("std") or {},
            source=d.get("source"),
            domain=d.get("domain"),
        )


# ---------------------------------------------------------------------------
# Regex pattern banks
# ---------------------------------------------------------------------------

# D1: Communication Style
POLITENESS_PATTERNS = [
    r"\bplease\b", r"\bpls\b", r"\bplz\b",
    r"\bthanks?\b", r"\bthank you\b", r"\bthx\b", r"\bty\b",
    r"\bcheers\b", r"\bmuch appreciated\b", r"\bappreciate\b", r"\bappreciated\b",
    r"\bgrateful\b",
    r"\bsorry\b", r"\bapologies\b", r"\bmy apologies\b",
    r"\bpardon\b", r"\bexcuse me\b",
    r"\bkindly\b",
    r"\bif you don't mind\b", r"\bif you wouldn't mind\b", r"\bwould you mind\b",
    r"\bif possible\b", r"\bif you can\b",
    r"\bwhen you (?:get|have) (?:a|the) (?:chance|moment|minute)\b",
    r"\bno (?:worries|rush|problem)\b",
]

FORMALITY_MARKERS = [
    r"\u2014",  # em-dash
    r"\u2013",  # en-dash
    r";",       # semicolons in prose
    r"\bfurthermore\b", r"\bmoreover\b", r"\bin addition\b", r"\badditionally\b",
    r"\btherefore\b", r"\bthus\b", r"\bhence\b", r"\bconsequently\b",
    r"\baccordingly\b", r"\bsubsequently\b",
    r"\bhowever\b", r"\bnevertheless\b", r"\bnonetheless\b", r"\bnotwithstanding\b",
    r"\bregarding\b", r"\bwith (?:regard|respect) to\b", r"\bin (?:regard|respect) to\b",
    r"\bpertaining to\b", r"\bconcerning\b",
    r"\bindeed\b", r"\bper se\b", r"\bas such\b",
]

ACKNOWLEDGMENT_PATTERNS = [
    r"^(ok|okay|k|kk|sure|got it|alright|yes|yeah|yea|yep|yup|"
    r"no|nope|nah|right|fine|great|cool|awesome|nice|"
    r"thanks|thank you|thx|ty|cheers|"
    r"understood|noted|copy|copy that|roger|roger that|ack|acknowledged|"
    r"i see|perfect|sounds good|sounds great|works for me|will do|"
    r"gotcha|got you|makes sense)[\.\!\,]*$"
]

IDENTITY_CONFUSION_PATTERNS = [
    r"\bhow (?:may|can) i (?:help|assist)\b",
    r"\blet me (?:check|look into|assist|pull up|find|locate|verify|confirm)\b",
    r"\bi(?:'ll| will) (?:look|check|verify|pull|find|look into) (?:that|this) for you\b",
    r"\bi apologize for (?:the|any) (?:inconvenience|confusion|delay|trouble|wait)\b",
    r"\bthank you for (?:your patience|your time|contacting|reaching out|your inquiry)\b",
    r"\bis there anything (?:else|further) i can (?:help|assist|do)\b",
    r"\bi(?:'d| would) be (?:happy|glad|delighted) to\b",
    r"\bplease (?:hold|wait|allow|give) (?:a moment|one moment|me a moment|while i)\b",
    r"\bone moment please\b", r"\bbear with me\b",
    r"\bour records (?:show|indicate|reflect)\b",
    r"\bfor (?:security|verification) (?:purposes|reasons)\b",
    r"\bper (?:our|company|store) policy\b",
    r"\brest assured\b", r"\bon behalf of\b",
    r"\bi(?:'ve| have) (?:gone ahead and|processed|submitted|escalated)\b",
]

# D3: Clarification Behavior
UNCERTAINTY_PATTERNS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b", r"\bprobably\b",
    r"\bnot sure\b", r"\bi(?:'m| am)? not sure\b", r"\bunsure\b",
    r"\bi think\b", r"\bi guess\b", r"\bi suppose\b", r"\bi believe\b",
    r"\bi(?:'m| am) not (?:entirely |completely |totally )?certain\b",
    r"\b(?:dunno|don't know|do not know|no idea)\b", r"\bidk\b",
    r"\b(?:kinda|kind of|sort of|sorta)\b",
    r"\b(?:might|could|may) be\b",
    r"\bapparently\b", r"\bpresumably\b", r"\bseemingly\b",
    r"\bif i (?:recall|remember) correctly\b", r"\biirc\b",
    r"\bhmm+\b",
]

CERTAINTY_PATTERNS = [
    r"\bdefinitely\b", r"\babsolutely\b", r"\bcertainly\b", r"\bsurely\b",
    r"\bundoubtedly\b", r"\bpositively\b",
    r"\bfor sure\b", r"\bfor certain\b", r"\bno doubt\b", r"\bwithout a doubt\b",
    r"\bexactly\b", r"\bprecisely\b", r"\bclearly\b", r"\bobviously\b",
    r"\bi(?:'m| am) (?:sure|certain|positive|confident)\b",
    r"\bguaranteed\b",
    r"\b100\s*%\b", r"\bhundred percent\b",
]

PUSHBACK_PATTERNS = [
    r"\bare you (?:sure|certain|serious)\b",
    r"\byou (?:already )?asked\b", r"\bi already (?:told|said|mentioned|explained)\b",
    r"\bi (?:just|literally) (?:said|told you|mentioned)\b",
    r"\bthat's not (?:right|correct|what i|true)\b",
    r"\bthat(?:'s| is) (?:wrong|incorrect)\b",
    r"\bthat doesn(?:'t| not) (?:sound|seem|look) right\b",
    r"\bi disagree\b", r"\bi don't (?:think|believe) (?:that's|so)\b",
    r"\bas i (?:said|mentioned|explained|noted)\b",
    r"\blike i (?:said|told you|mentioned)\b",
    r"\bhow many times\b", r"\bdidn(?:'t| not) i (?:just )?(?:say|tell)\b",
    r"\bwe (?:already )?(?:discussed|covered|went over)\b",
    r"\byou(?:'re| are) not listening\b",
    r"\bread what i (?:wrote|said|typed)\b", r"\bscroll up\b",
]

CLARIFICATION_QUESTION_PATTERNS = [
    r"what do you mean", r"what does that mean",
    r"can you (?:clarify|explain|elaborate|rephrase)",
    r"could you (?:clarify|explain|elaborate|rephrase|repeat)",
    r"i(?:'m| am) not sure i understand", r"i don't (?:understand|follow|get it)",
    r"i(?:'m| am) (?:confused|lost)",
    r"what exactly", r"what specifically", r"can you be more specific",
    r"which (?:one|do you mean|should i)",
    r"\bhuh\??", r"\bwhat\??$", r"\bsorry\??$", r"\bpardon\??$", r"\bcome again\b",
    r"can you repeat", r"say that again", r"\bwait,? what\b",
]

INFO_SEEKING_PATTERNS = [
    r"what is the (?:status|update|eta|situation)\b", r"what(?:'s| is) the (?:status|update)\b",
    r"any (?:update|updates|news|info|information)\b",
    r"can you (?:check|confirm|verify|look up)\b",
    r"is there a way to\b", r"is it possible to\b",
    r"how (?:do|can|could|would|should) i\b",
    r"where (?:do|can|should) i\b", r"when (?:will|does|is|can)\b",
    r"why (?:is|does|do|are|am)\b", r"who (?:do|should|can) i\b",
    r"what (?:are|were|is) the\b", r"what about\b",
    r"could you (?:tell|let|show|send|share) me\b",
    r"(?:please )?tell me (?:about|what|how|when|where|why)\b",
    r"do you (?:know|have)\b",
    r"i need (?:to know|the|an)\b",
]

# D4: Error Reaction
EMOTIONAL_EXPRESSION_PATTERNS = [
    r"\bfrustrat(?:ed|ing|ion)\b", r"\bannoyed\b", r"\bannoying\b",
    r"\birritat(?:ed|ing)\b", r"\bfed up\b",
    r"\bangry\b", r"\bmad\b", r"\bfurious\b", r"\bpissed\b",
    r"\bupset\b", r"\bdisappoint(?:ed|ing|ment)\b",
    r"\bstressed\b", r"\bworried\b", r"\banxious\b",
    r"\bexhausted\b", r"\btired of\b", r"\bsick of\b",
    r"\b(?:ugh+|argh+|grr+)\b", r"\boh (?:come on|no|man)\b", r"\bcome on\b",
    r"\bridiculous\b", r"\babsurd\b", r"\binsane\b", r"\boutrageous\b",
    r"\bunbelievable\b", r"\bdisgraceful\b",
    r"\bnever ?mind\b", r"\bforget it\b", r"\bforget (?:that|this)\b",
    r"\bthis is (?:taking|getting|becoming|so|really)\b",
    r"\bseriously\??", r"\bwtf\b", r"\bffs\b",
]

ACCUSATORY_PATTERNS = [
    r"\buseless\b", r"\bunacceptable\b", r"\bincompetent\b", r"\bunprofessional\b",
    r"\bscam\b", r"\bfraud(?:ulent)?\b", r"\brip[- ]?off\b", r"\bripped me off\b",
    r"\b(?:ly|lying|liar|dishonest|deceptive)\b",
    r"\bcheat(?:ed|ing|er)?\b",
    r"\bworst\b", r"\bterrible\b", r"\bhorrific\b", r"\bhorrible\b",
    r"\bawful\b", r"\bappalling\b", r"\bshameful\b", r"\bpathetic\b",
    r"\ba joke\b", r"\btotal joke\b",
    r"\bwaste of (?:time|my time|money)\b",
    r"\byou(?:'re| are) not (?:helping|helpful|listening)\b",
    r"\bdo your job\b",
]

STRATEGY_PIVOT_PATTERNS = [
    r"\binstead\b", r"\brather than\b", r"\bin that case\b",
    r"\bon second thought\b", r"\bactually(?:,| let| i| we)\b",
    r"\blet(?:'s| us) try\b", r"\blet me try\b",
    r"\bhow about\b", r"\bhow about this\b",
    r"\balternatively\b", r"\banother (?:option|way|approach|idea)\b",
    r"\bwhat if (?:we|i|you)\b", r"\bor (?:maybe|perhaps|else)\b",
    r"\bcan we (?:try|do) something (?:else|different)\b",
    r"\bdifferent approach\b", r"\bchange of plans\b", r"\bnew plan\b",
    r"\bscratch that\b", r"\bforget (?:it|that)\b",
]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _rate_from_patterns(
    texts: List[str], patterns: List[str],
) -> float:
    """Fraction of turns matching at least one pattern."""
    if not texts:
        return 0.0
    hits = 0
    for text in texts:
        text_lower = text.lower()
        if any(re.search(p, text_lower) for p in patterns):
            hits += 1
    return hits / len(texts)


class BehavioralFingerprintExtractor:
    """Extracts behavioral fingerprints from dialogue traces.

    Computes 19 regex/statistical features (D1–D4) only.
    """

    def feature_names(self) -> List[str]:
        return list(ALL_FEATURES)

    def regex_feature_names(self) -> List[str]:
        return list(REGEX_FEATURES)

    def compute_fingerprint(
        self,
        trace: List[Dict],
        user_role: str = "user",
        agent_role: str = "assistant",
    ) -> BehavioralFingerprint:
        """Compute regex/statistical features from dialogue (no LLM calls)."""
        user_turns = [
            t for t in trace
            if t.get("role") == user_role and t.get("content")
        ]
        if not user_turns:
            return BehavioralFingerprint(
                features={k: 0.0 for k in ALL_FEATURES}
            )

        user_texts = [t["content"] for t in user_turns]
        user_tokens = [_tokenize(t) for t in user_texts]
        turn_lengths = [len(toks) for toks in user_tokens]

        features: Dict[str, float] = {}

        # === D1: Communication Style ===

        features["words_per_turn"] = float(np.mean(turn_lengths)) if turn_lengths else 0.0

        features["short_utterance_rate"] = (
            float(np.mean([1.0 if length <= 3 else 0.0 for length in turn_lengths]))
            if turn_lengths else 0.0
        )

        features["politeness_rate"] = _rate_from_patterns(user_texts, POLITENESS_PATTERNS)

        features["formality_rate"] = _rate_from_patterns(user_texts, FORMALITY_MARKERS)

        features["acknowledgment_rate"] = 0.0
        if user_texts:
            ack_hits = 0
            for text in user_texts:
                text_stripped = text.strip().lower()
                if any(re.fullmatch(p, text_stripped) for p in ACKNOWLEDGMENT_PATTERNS):
                    ack_hits += 1
            features["acknowledgment_rate"] = ack_hits / len(user_texts)

        if len(turn_lengths) > 1 and np.mean(turn_lengths) > 0:
            features["verbosity_cv"] = float(np.std(turn_lengths) / np.mean(turn_lengths))
        else:
            features["verbosity_cv"] = 0.0

        features["repetition_rate"] = self._compute_repetition(user_tokens)

        features["identity_confusion_rate"] = _rate_from_patterns(
            user_texts, IDENTITY_CONFUSION_PATTERNS
        )

        # === D2: Information Pattern ===

        total_words = sum(turn_lengths)
        if total_words > 0 and len(user_texts) >= 2:
            first_two_words = sum(turn_lengths[:2])
            features["front_loading_ratio"] = first_two_words / total_words
        elif total_words > 0:
            features["front_loading_ratio"] = 1.0
        else:
            features["front_loading_ratio"] = 0.0

        id_pattern = re.compile(
            r"#[A-Z0-9]{5,}"
            r"|[A-Z0-9]{6,}"
            r"|\b\d{7,}\b"
            r"|\S+@\S+\.\S+"
        )
        id_counts = []
        for text in user_texts:
            id_counts.append(len(id_pattern.findall(text)))
        features["identifiers_per_turn"] = float(np.mean(id_counts)) if id_counts else 0.0

        features["opening_length"] = float(turn_lengths[0]) if turn_lengths else 0.0

        # === D3: Clarification Behavior ===

        features["uncertainty_rate"] = _rate_from_patterns(user_texts, UNCERTAINTY_PATTERNS)
        features["certainty_rate"] = _rate_from_patterns(user_texts, CERTAINTY_PATTERNS)
        features["pushback_rate"] = _rate_from_patterns(user_texts, PUSHBACK_PATTERNS)
        features["clarification_question_rate"] = _rate_from_patterns(
            user_texts, CLARIFICATION_QUESTION_PATTERNS
        )
        features["info_seeking_rate"] = _rate_from_patterns(user_texts, INFO_SEEKING_PATTERNS)

        # === D4: Error Reaction ===

        features["emotional_expression_rate"] = _rate_from_patterns(
            user_texts, EMOTIONAL_EXPRESSION_PATTERNS
        )
        features["accusatory_rate"] = _rate_from_patterns(user_texts, ACCUSATORY_PATTERNS)
        features["strategy_pivot_rate"] = _rate_from_patterns(
            user_texts, STRATEGY_PIVOT_PATTERNS
        )

        return BehavioralFingerprint(features=features)

    def _compute_repetition(self, user_tokens: List[List[str]]) -> float:
        """Fraction of turns containing a trigram that appears >3 times across all turns."""
        if not user_tokens:
            return 0.0

        from collections import Counter
        trigram_counts: Counter = Counter()
        for toks in user_tokens:
            for i in range(len(toks) - 2):
                trigram_counts[(toks[i], toks[i + 1], toks[i + 2])] += 1

        repeated_trigrams = {tri for tri, cnt in trigram_counts.items() if cnt > 3}
        if not repeated_trigrams:
            return 0.0

        hits = 0
        for toks in user_tokens:
            turn_trigrams = set()
            for i in range(len(toks) - 2):
                turn_trigrams.add((toks[i], toks[i + 1], toks[i + 2]))
            if turn_trigrams & repeated_trigrams:
                hits += 1
        return hits / len(user_tokens)


# ---------------------------------------------------------------------------
# Dice-based human likeness scoring
# ---------------------------------------------------------------------------

def dice_coefficient(model_value: float, human_value: float) -> float:
    """Sørensen-Dice for a single scalar feature: 2*min(M,H) / (M+H).

    Returns value in [0, 1]. If both are 0, returns 1.0 (perfect match).
    """
    m, h = abs(model_value), abs(human_value)
    denom = m + h
    if denom < 1e-12:
        return 1.0
    return 2.0 * min(m, h) / denom


def aggregate_fingerprints(
    fingerprints: List[BehavioralFingerprint],
    feature_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Per-feature mean across a list of fingerprints.

    Used to collapse a simulator's episodes into one aggregate feature vector
    before Dice comparison against the human mean (aggregate-first).
    """
    if not fingerprints:
        return {}
    names = feature_names if feature_names is not None else REGEX_FEATURES
    return {
        f: float(np.mean([fp.features.get(f, 0.0) for fp in fingerprints]))
        for f in names
    }


def _dim_dice(
    agg: Dict[str, float],
    human_dist: HumanBehavioralDistribution,
    features: List[str],
) -> float:
    if not features:
        return 0.0
    return float(np.mean([
        dice_coefficient(agg.get(f, 0.0), human_dist.mean.get(f, 0.0))
        for f in features
    ]))


def compute_aggregate_dice_alignment(
    fingerprints: List[BehavioralFingerprint],
    human_dist: HumanBehavioralDistribution,
) -> Dict[str, float]:
    """Aggregate-first Sørensen–Dice alignment against the human reference.

    Aggregate-first approach: average each regex feature across the
    simulator's episodes to form one aggregate vector, then compute the
    Sørensen–Dice coefficient against the human mean per dimension.

    Returns a dict on the **[0, 1]** scale (not 0–100):
        {"D1": ..., "D2": ..., "D3": ..., "D4": ..., "overall": ...}
    where ``overall`` is the unweighted mean of D1–D4.
    """
    if not fingerprints:
        return {"D1": 0.0, "D2": 0.0, "D3": 0.0, "D4": 0.0, "overall": 0.0}
    agg = aggregate_fingerprints(fingerprints)
    d1 = _dim_dice(agg, human_dist, D1_FEATURES)
    d2 = _dim_dice(agg, human_dist, D2_FEATURES)
    d3 = _dim_dice(agg, human_dist, D3_FEATURES)
    d4 = _dim_dice(agg, human_dist, D4_FEATURES)
    return {
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "D4": d4,
        "overall": float(np.mean([d1, d2, d3, d4])),
    }


# ---------------------------------------------------------------------------
# Human reference computation from tau_bench_human.json
# ---------------------------------------------------------------------------

def compute_human_distribution(
    dialogues: List[Dict],
    extractor: BehavioralFingerprintExtractor,
    user_role: Optional[str] = None,
    *,
    source: Optional[str] = None,
    domain: Optional[str] = None,
) -> HumanBehavioralDistribution:
    """Compute per-feature means from a list of human conversations.

    Each dialogue should have a ``conversation`` key with a list of
    ``{role, content}`` messages, matching ``tau_bench_human.json`` format.

    ``source`` and ``domain`` are stored in the returned distribution for provenance.
    """
    all_features: List[Dict[str, float]] = []
    regex_names = extractor.regex_feature_names()

    for dialogue in dialogues:
        if "conversation" in dialogue:
            trace = dialogue["conversation"]
        elif "turns" in dialogue:
            trace = dialogue["turns"]
        else:
            trace = dialogue if isinstance(dialogue, list) else []

        if not trace:
            continue

        fp = extractor.compute_fingerprint(trace, user_role="user", agent_role="assistant")
        all_features.append({f: fp.features.get(f, 0.0) for f in regex_names})

    if not all_features:
        raise ValueError("No fingerprints computed. Check dialogue format.")

    mean = {
        f: float(np.mean([fp[f] for fp in all_features]))
        for f in regex_names
    }
    std = {
        f: float(np.std([fp[f] for fp in all_features], ddof=0))
        for f in regex_names
    }

    dist = HumanBehavioralDistribution(
        mean=mean,
        std=std,
        feature_names=regex_names,
        n_dialogues=len(all_features),
        source=source,
        domain=domain,
    )
    print(
        f"Human distribution computed from {len(all_features)} conversations "
        f"({len(regex_names)} regex features)"
    )
    return dist
