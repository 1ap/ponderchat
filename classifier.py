"""
classifier.py - Reasoning Classifier with Gemma 4 + Hybrid Thinking

Uses Gemma 4 E4B 4-bit (~3.5GB) with three thinking modes:
- off:    Fast classification (~100-150ms)
- brief:  Light reasoning via system prompt (~250-400ms)
- full:   Native enable_thinking=True (~500-800ms)

Cascade architecture:
  Tier 1: thinking=off  → if confident, return
  Tier 2: thinking=full → if still uncertain, escalate to user
  Tier 3: User picks between top alternatives

Outputs two independent dimensions:
  tier:   haiku | sonnet | opus
  effort: low | medium | high | xhigh | max
"""

import json
import re
import logging
import hashlib
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# DIMENSIONS
# ============================================================================

class CapabilityTier(str, Enum):
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


class Effort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class ThinkingMode(str, Enum):
    """Local classifier's thinking depth"""
    OFF = "off"      # No reasoning, ~100-150ms
    BRIEF = "brief"  # System-prompted brief reasoning, ~250-400ms
    FULL = "full"    # Native enable_thinking=True, ~500-800ms


MODEL_IDS = {
    CapabilityTier.HAIKU: "claude-haiku-4-5",
    CapabilityTier.SONNET: "claude-sonnet-4-6",
    CapabilityTier.OPUS: "claude-opus-4-7",
}

PRICING = {
    CapabilityTier.HAIKU: (1.0, 5.0),
    CapabilityTier.SONNET: (3.0, 15.0),
    CapabilityTier.OPUS: (5.0, 25.0),
}

EFFORT_TOKEN_ESTIMATES = {
    Effort.LOW: 2000,
    Effort.MEDIUM: 8000,
    Effort.HIGH: 16000,
    Effort.XHIGH: 32000,
    Effort.MAX: 64000,
}

VALID_REASONING_TYPES = {
    "domain_expertise", "multi_step_logic", "synthesis",
    "creative_problem_solving", "nuanced_judgment", "code_or_math",
    "requires_context", "long_output", "agentic",
}


# ============================================================================
# SYSTEM PROMPTS - one per thinking mode
# ============================================================================

# Common output spec
OUTPUT_SPEC = """Output ONLY valid JSON:
{
  "tier": "<haiku|sonnet|opus>",
  "effort": "<low|medium|high|xhigh|max>",
  "confidence": <0.0-1.0>,
  "alternative": null OR {"tier": "...", "effort": "..."},
  "reasoning_types": {<type>: <0-1>},
  "explanation": "<one sentence>",
  "ambiguity": null OR "<one sentence on why this could go either way>"
}"""

# Tier and effort definitions
DIMENSIONS_DOC = """DIMENSION 1 - tier (model capability):
- haiku: Simple Q&A, lookups, classification, routing. Fast/cheap.
- sonnet: Most coding, explanations, agentic work. Balanced.
- opus: Hard reasoning, novel problems, large refactors. Best.

DIMENSION 2 - effort (thinking depth):
- low: Direct answers, no extended reasoning
- medium: Standard reasoning chain (default)
- high: Deeper analysis, multi-step problems
- xhigh: Very hard problems (opus only)
- max: Maximum reasoning, novel research (opus only)

REASONING TYPES (only use these): domain_expertise, multi_step_logic, synthesis, creative_problem_solving, nuanced_judgment, code_or_math, requires_context, long_output, agentic"""

# Calibration instructions
CALIBRATION_DOC = """CRITICAL: Honest uncertainty is more valuable than false confidence.

CALIBRATE YOUR CONFIDENCE:
- 0.95+: unambiguously this tier/effort (e.g., "what's 2+2?" → haiku/low)
- 0.80-0.95: clearly this category, no real alternatives
- 0.60-0.80: probably this, but a reasonable case could be made for adjacent
- 0.40-0.60: genuinely ambiguous - fill "alternative" with the other plausible choice
- <0.40: unclear - pick best guess but flag heavily in "ambiguity"

Examples of high uncertainty (set confidence ~0.5 + provide alternative):
- "Implement this algorithm" - could be sonnet+medium (routine) or opus+high (subtle correctness)
- "Help me think through X" - could be sonnet+high (analysis) or opus+xhigh (deep reasoning)
- "Refactor this" - depends entirely on what "this" is

Your downstream system can ASK THE USER when confidence is low.
Flagging uncertainty is helpful, not a failure.

CONTEXT MATTERS:
If conversation context is provided, classify what the FULL EXCHANGE needs.
"What about caching?" alone → haiku + low
"What about caching?" after "Help me design a distributed system" → opus + high
But genuinely trivial requests stay trivial even in complex threads."""


# Mode-specific system prompts
SYSTEM_PROMPT_OFF = f"""You classify prompts on TWO independent dimensions for the Claude API.

{OUTPUT_SPEC}

{DIMENSIONS_DOC}

{CALIBRATION_DOC}"""


SYSTEM_PROMPT_BRIEF = f"""You classify prompts on TWO independent dimensions for the Claude API.

Keep your thinking brief (LOW depth) - just identify the key signals before classifying.

{OUTPUT_SPEC}

{DIMENSIONS_DOC}

{CALIBRATION_DOC}"""


SYSTEM_PROMPT_FULL = f"""You classify prompts on TWO independent dimensions for the Claude API.

Reason carefully through:
1. What is the prompt actually asking for?
2. What context (if any) changes its complexity?
3. What's the obvious answer? What's the second-most-plausible?
4. Where am I uncertain?

{OUTPUT_SPEC}

{DIMENSIONS_DOC}

{CALIBRATION_DOC}"""


SYSTEM_PROMPTS = {
    ThinkingMode.OFF: SYSTEM_PROMPT_OFF,
    ThinkingMode.BRIEF: SYSTEM_PROMPT_BRIEF,
    ThinkingMode.FULL: SYSTEM_PROMPT_FULL,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ClassificationResult:
    tier: CapabilityTier
    effort: Effort
    confidence: float
    reasoning_types: Dict[str, float]
    explanation: str
    alternative: Optional[Dict[str, str]] = None  # {"tier": "...", "effort": "..."}
    ambiguity: Optional[str] = None
    model_id: str = ""
    estimated_thinking_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    method: str = "mlx"
    thinking_mode: str = "off"
    needs_escalation: bool = False  # Flag for user-facing UI

    def __post_init__(self):
        if not self.model_id:
            self.model_id = MODEL_IDS[self.tier]
        if not self.estimated_thinking_tokens:
            self.estimated_thinking_tokens = EFFORT_TOKEN_ESTIMATES[self.effort]


# ============================================================================
# GEMMA 4 RESPONSE PARSING
# ============================================================================

# Gemma 4 channel format: <|channel>thought\n[reasoning]<channel|>[answer]
# Note: token format has asymmetric pipes
THOUGHT_PATTERN = re.compile(
    r"<\|?channel\|?>thought\s*\n?(.*?)<\|?channel\|?>",
    re.DOTALL,
)


def parse_gemma_response(response: str) -> Tuple[str, str]:
    """Parse Gemma 4 channel-based output. Returns (thought, answer)."""
    thought_match = THOUGHT_PATTERN.search(response)
    thought = thought_match.group(1).strip() if thought_match else ""
    answer = THOUGHT_PATTERN.sub("", response).strip()
    return thought, answer


# ============================================================================
# OUTPUT VALIDATOR
# ============================================================================

class OutputValidator:
    @staticmethod
    def parse_and_validate(text: str) -> Optional[Dict]:
        # Strip Gemma thought channels first
        _, answer = parse_gemma_response(text)
        text_to_parse = answer if answer else text

        # Find JSON block
        json_match = re.search(r'\{.*\}', text_to_parse, re.DOTALL)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

        # Validate tier
        if "tier" not in data or not isinstance(data["tier"], str):
            return None
        tier = data["tier"].lower().strip()
        if tier not in [t.value for t in CapabilityTier]:
            return None
        data["tier"] = tier

        # Validate effort
        if "effort" not in data or not isinstance(data["effort"], str):
            return None
        effort = data["effort"].lower().strip()
        if effort not in [e.value for e in Effort]:
            return None
        data["effort"] = effort

        # Validate confidence
        try:
            data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (ValueError, TypeError):
            data["confidence"] = 0.5

        # Validate reasoning_types
        if not isinstance(data.get("reasoning_types"), dict):
            data["reasoning_types"] = {}
        data["reasoning_types"] = {
            k: max(0.0, min(1.0, float(v)))
            for k, v in data["reasoning_types"].items()
            if k in VALID_REASONING_TYPES and isinstance(v, (int, float))
        }

        # Validate optional alternative
        alt = data.get("alternative")
        if alt is not None:
            if not isinstance(alt, dict) or "tier" not in alt or "effort" not in alt:
                data["alternative"] = None
            elif alt["tier"] not in [t.value for t in CapabilityTier]:
                data["alternative"] = None
            elif alt["effort"] not in [e.value for e in Effort]:
                data["alternative"] = None

        # Validate optional ambiguity
        if not isinstance(data.get("ambiguity"), (str, type(None))):
            data["ambiguity"] = None

        if not isinstance(data.get("explanation"), str):
            data["explanation"] = ""

        return data


# ============================================================================
# GUARDRAILS
# ============================================================================

class Guardrails:
    @staticmethod
    def apply(tier: str, effort: str, confidence: float, reasoning_types: Dict) -> Tuple[str, str]:
        if effort in ("xhigh", "max") and tier != "opus":
            tier = "opus"
        if reasoning_types.get("creative_problem_solving", 0) > 0.85 and tier == "haiku":
            tier = "sonnet"
        if reasoning_types.get("agentic", 0) > 0.7 and tier == "haiku":
            tier = "sonnet"
        if reasoning_types.get("long_output", 0) > 0.7 and tier == "haiku":
            tier = "sonnet"

        # Note: low-confidence upgrade now handled by escalation, not guardrail
        return tier, effort


# ============================================================================
# POLICIES
# ============================================================================

class Policy:
    PRESETS = {
        "auto": None,
        "cheap": {"max_tier": "sonnet", "max_effort": "medium"},
        "balanced": {"downgrade_xhigh_max": True},
        "quality": {"min_tier_for_high_effort": "sonnet"},
        "max": {"force_tier": "opus", "force_effort": "max"},
    }

    @classmethod
    def apply(cls, tier: str, effort: str, policy: str = "auto") -> Tuple[str, str]:
        if policy == "auto" or policy not in cls.PRESETS:
            return tier, effort
        rules = cls.PRESETS[policy]
        if rules is None:
            return tier, effort

        tier_order = ["haiku", "sonnet", "opus"]
        effort_order = ["low", "medium", "high", "xhigh", "max"]

        if "force_tier" in rules:
            tier = rules["force_tier"]
        if "force_effort" in rules:
            effort = rules["force_effort"]
        if "max_tier" in rules:
            if tier_order.index(tier) > tier_order.index(rules["max_tier"]):
                tier = rules["max_tier"]
        if "max_effort" in rules:
            if effort_order.index(effort) > effort_order.index(rules["max_effort"]):
                effort = rules["max_effort"]
        if rules.get("downgrade_xhigh_max") and effort in ("xhigh", "max"):
            effort = "high"
        if "min_tier_for_high_effort" in rules and effort in ("high", "xhigh", "max"):
            min_t = rules["min_tier_for_high_effort"]
            if tier_order.index(tier) < tier_order.index(min_t):
                tier = min_t
        return tier, effort


# ============================================================================
# HEURISTIC FALLBACK
# ============================================================================

class HeuristicClassifier:
    KEYWORDS = {
        "domain_expertise": ["medical", "legal", "physics", "quantum", "specialized"],
        "multi_step_logic": ["then", "therefore", "process", "workflow"],
        "synthesis": ["compare", "contrast", "relationship", "integrate"],
        "creative_problem_solving": ["design", "invent", "novel", "creative"],
        "nuanced_judgment": ["should", "ethical", "dilemma", "tradeoff"],
        "code_or_math": ["code", "function", "algorithm", "debug", "refactor"],
        "requires_context": ["based on", "given the", "according to"],
        "long_output": ["full implementation", "entire", "complete code"],
        "agentic": ["execute", "run tools", "agent", "automate"],
    }

    @classmethod
    def classify(cls, prompt: str) -> Dict:
        prompt_lower = prompt.lower()
        reasoning_types = {}
        for rtype, keywords in cls.KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in prompt_lower) * 0.3
            if score > 0:
                reasoning_types[rtype] = min(1.0, score)

        active = len(reasoning_types)
        length = len(prompt.split())
        has_code = reasoning_types.get("code_or_math", 0) > 0
        has_creative = reasoning_types.get("creative_problem_solving", 0) > 0

        if active == 0 and length < 15:
            tier, effort = "haiku", "low"
        elif active <= 1 and not has_creative:
            tier, effort = "haiku", "medium"
        elif active == 2 or has_code:
            tier, effort = "sonnet", "medium"
        elif active == 3:
            tier, effort = "sonnet", "high"
        elif active >= 4 or has_creative:
            tier, effort = "opus", "high"
        else:
            tier, effort = "sonnet", "medium"

        return {
            "tier": tier, "effort": effort, "confidence": 0.5,
            "reasoning_types": reasoning_types,
            "explanation": "Heuristic fallback (MLX unavailable)",
            "alternative": None, "ambiguity": "Heuristic only - no semantic analysis",
        }


# ============================================================================
# CACHE
# ============================================================================

class Cache:
    def __init__(self, cache_dir: str = ".classifier_cache", ttl_hours: int = 168):
        self.dir = Path(cache_dir)
        self.dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.memory = {}

    def _key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> Optional[Dict]:
        key = self._key(prompt)
        if key in self.memory:
            item = self.memory[key]
            if datetime.now() - item["t"] < self.ttl:
                return item["d"]
        path = self.dir / f"{key}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    item = pickle.load(f)
                if datetime.now() - item["t"] < self.ttl:
                    self.memory[key] = item
                    return item["d"]
                path.unlink()
            except Exception:
                pass
        return None

    def set(self, prompt: str, data: Dict) -> None:
        key = self._key(prompt)
        item = {"t": datetime.now(), "d": data}
        self.memory[key] = item
        try:
            with open(self.dir / f"{key}.pkl", "wb") as f:
                pickle.dump(item, f)
        except Exception:
            pass


# ============================================================================
# MAIN CLASSIFIER WITH HYBRID THINKING CASCADE
# ============================================================================

class ReliableClassifier:
    """
    Reasoning classifier using Gemma 4 E4B 4-bit with hybrid thinking.

    Cascade:
        Tier 1 (fast, ~100ms):     thinking=off
            ↓ if confidence < escalate_to_thinking_below
        Tier 2 (deep, ~500ms):     thinking=full
            ↓ if confidence < escalate_to_user_below
        Tier 3:                     needs_escalation=True (caller asks user)

    Args:
        model_name: HuggingFace/MLX model ID
        policy: Routing policy (auto/cheap/balanced/quality/max)
        escalate_to_thinking_below: Trigger thinking mode below this confidence
        escalate_to_user_below: Trigger user escalation below this confidence
        use_cache: Enable result caching
        verbose: Print debug info
    """

    DEFAULT_MODEL = "mlx-community/gemma-4-E4B-it-4bit"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        policy: str = "auto",
        escalate_to_thinking_below: float = 0.75,
        escalate_to_user_below: float = 0.55,
        use_cache: bool = True,
        cache_dir: str = ".classifier_cache",
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.policy = policy
        self.escalate_to_thinking_below = escalate_to_thinking_below
        self.escalate_to_user_below = escalate_to_user_below
        self.verbose = verbose
        self.cache = Cache(cache_dir) if use_cache else None
        self.validator = OutputValidator()
        self.guardrails = Guardrails()

        self.model = None
        self.tokenizer = None
        self.stats = {
            "total": 0, "cache_hits": 0, "fallbacks": 0,
            "tier1_only": 0, "tier2_used": 0, "tier3_escalations": 0,
        }
        self._load_model()

    def _load_model(self) -> bool:
        try:
            from mlx_lm import load
            if self.verbose:
                print(f"Loading {self.model_name}...")
            self.model, self.tokenizer = load(self.model_name)
            if self.verbose:
                print(f"✓ Model loaded")
            return True
        except Exception as e:
            logger.warning(f"MLX unavailable: {str(e)[:100]}")
            return False

    def classify(
        self,
        prompt: str,
        history: Optional[List[Dict]] = None,
        previous_tier: Optional[str] = None,
        previous_effort: Optional[str] = None,
        force_thinking: Optional[ThinkingMode] = None,
    ) -> ClassificationResult:
        """
        Classify with cascade through thinking modes.

        Args:
            prompt: User prompt
            history: Recent conversation [{role, content}, ...]
            previous_tier: Last turn's tier (for inertia)
            previous_effort: Last turn's effort (for inertia)
            force_thinking: Override cascade, force a specific thinking mode
        """
        if not prompt or not isinstance(prompt, str):
            data = HeuristicClassifier.classify("")
            return self._build(data, method="fallback", thinking_mode=ThinkingMode.OFF)

        self.stats["total"] += 1
        start = time.time()

        # Cache lookup
        cache_key = self._cache_key(prompt, history)
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                self.stats["cache_hits"] += 1
                return self._build(cached, method="cache", cached=True,
                                  thinking_mode=ThinkingMode.OFF,
                                  latency_ms=(time.time() - start) * 1000)

        # No model? Heuristic only
        if self.model is None:
            self.stats["fallbacks"] += 1
            data = HeuristicClassifier.classify(prompt)
            data = self._post_process(data, previous_tier, previous_effort)
            return self._build(data, method="fallback", thinking_mode=ThinkingMode.OFF,
                              latency_ms=(time.time() - start) * 1000)

        # Cascade
        used_thinking = ThinkingMode.OFF
        result_data = None

        # Tier 1: Fast (or forced mode)
        if force_thinking:
            used_thinking = force_thinking
            result_data = self._classify_mlx(prompt, history, force_thinking)
        else:
            used_thinking = ThinkingMode.OFF
            result_data = self._classify_mlx(prompt, history, ThinkingMode.OFF)

            # Escalate to thinking if uncertain
            if result_data and result_data["confidence"] < self.escalate_to_thinking_below:
                if self.verbose:
                    print(f"  ↳ Confidence {result_data['confidence']:.2f} below threshold, escalating to thinking mode")
                used_thinking = ThinkingMode.FULL
                self.stats["tier2_used"] += 1
                thinking_result = self._classify_mlx(prompt, history, ThinkingMode.FULL)
                if thinking_result:
                    result_data = thinking_result
            else:
                self.stats["tier1_only"] += 1

        # Heuristic fallback if MLX returned nothing
        if result_data is None:
            self.stats["fallbacks"] += 1
            result_data = HeuristicClassifier.classify(prompt)
            method = "fallback"
        else:
            method = "mlx"

        # Post-process: inertia, guardrails, policy
        result_data = self._post_process(result_data, previous_tier, previous_effort)

        # Check if user escalation needed
        needs_escalation = result_data["confidence"] < self.escalate_to_user_below
        if needs_escalation:
            self.stats["tier3_escalations"] += 1

        # Cache only confident MLX results
        if self.cache and method == "mlx" and not needs_escalation:
            self.cache.set(cache_key, result_data)

        return self._build(
            result_data, method=method, thinking_mode=used_thinking,
            latency_ms=(time.time() - start) * 1000,
            needs_escalation=needs_escalation,
        )

    def _cache_key(self, prompt: str, history: Optional[List[Dict]]) -> str:
        """Cache key includes recent history snippet"""
        if history:
            ctx = " | ".join(m.get("content", "")[:80] for m in history[-2:])
            return f"{ctx} >>> {prompt}"
        return prompt

    def _post_process(
        self,
        data: Dict,
        previous_tier: Optional[str],
        previous_effort: Optional[str],
    ) -> Dict:
        """Apply inertia, guardrails, policy"""
        # Inertia
        if previous_tier and previous_effort:
            data["tier"], data["effort"] = self._apply_inertia(
                data["tier"], data["effort"],
                previous_tier, previous_effort,
                data["confidence"],
            )

        # Guardrails
        data["tier"], data["effort"] = self.guardrails.apply(
            data["tier"], data["effort"],
            data["confidence"], data["reasoning_types"],
        )

        # Policy
        data["tier"], data["effort"] = Policy.apply(
            data["tier"], data["effort"], self.policy,
        )

        return data

    def _apply_inertia(
        self, tier: str, effort: str,
        prev_tier: str, prev_effort: str, confidence: float,
    ) -> Tuple[str, str]:
        tier_order = ["haiku", "sonnet", "opus"]
        effort_order = ["low", "medium", "high", "xhigh", "max"]

        prev_tier_idx = tier_order.index(prev_tier)
        new_tier_idx = tier_order.index(tier)
        if new_tier_idx < prev_tier_idx - 1 and confidence < 0.8:
            tier = tier_order[prev_tier_idx - 1]

        prev_effort_idx = effort_order.index(prev_effort)
        new_effort_idx = effort_order.index(effort)
        if new_effort_idx < prev_effort_idx - 2 and confidence < 0.8:
            effort = effort_order[prev_effort_idx - 2]

        return tier, effort

    def _classify_mlx(
        self,
        prompt: str,
        history: Optional[List[Dict]],
        thinking_mode: ThinkingMode,
    ) -> Optional[Dict]:
        """Call Gemma 4 with the appropriate thinking mode"""
        try:
            from mlx_lm import generate

            # Build user message with optional context
            if history:
                recent = history[-4:]
                ctx_lines = [f"{m.get('role','')}: {m.get('content','')[:200]}" for m in recent]
                ctx = "\n".join(ctx_lines)
                user_msg = f"Recent conversation:\n{ctx}\n\nNew prompt to classify: \"{prompt}\""
            else:
                user_msg = f'Classify: "{prompt}"'

            messages = [
                {"role": "system", "content": SYSTEM_PROMPTS[thinking_mode]},
                {"role": "user", "content": user_msg},
            ]

            # Apply chat template with enable_thinking for FULL mode
            try:
                template_kwargs = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                # Enable native thinking for FULL mode
                if thinking_mode == ThinkingMode.FULL:
                    template_kwargs["enable_thinking"] = True

                formatted = self.tokenizer.apply_chat_template(messages, **template_kwargs)
            except TypeError:
                # Older tokenizers might not accept enable_thinking
                formatted = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                formatted = f"{SYSTEM_PROMPTS[thinking_mode]}\n\n{user_msg}\n\nResponse:"

            # Higher token budget for thinking mode (it produces reasoning + JSON)
            max_tokens = 800 if thinking_mode == ThinkingMode.FULL else 250

            response = generate(
                self.model, self.tokenizer,
                prompt=formatted, max_tokens=max_tokens, verbose=False,
            )

            return self.validator.parse_and_validate(response)
        except Exception as e:
            logger.error(f"MLX classification error: {e}")
            return None

    def _build(
        self,
        data: Dict,
        method: str = "mlx",
        cached: bool = False,
        latency_ms: float = 0.0,
        thinking_mode: ThinkingMode = ThinkingMode.OFF,
        needs_escalation: bool = False,
    ) -> ClassificationResult:
        return ClassificationResult(
            tier=CapabilityTier(data["tier"]),
            effort=Effort(data["effort"]),
            confidence=data["confidence"],
            reasoning_types=data.get("reasoning_types", {}),
            explanation=data.get("explanation", ""),
            alternative=data.get("alternative"),
            ambiguity=data.get("ambiguity"),
            latency_ms=latency_ms,
            cached=cached,
            method=method,
            thinking_mode=thinking_mode.value,
            needs_escalation=needs_escalation,
        )

    def get_stats(self) -> Dict:
        total = self.stats["total"] or 1
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / total,
            "thinking_escalation_rate": self.stats["tier2_used"] / total,
            "user_escalation_rate": self.stats["tier3_escalations"] / total,
        }


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    classifier = ReliableClassifier(verbose=True, policy="balanced")

    test_prompts = [
        "What's 2+2?",
        "Refactor this Python function",
        "Implement a custom comparator for sorting",  # ambiguous
        "Design a microservices architecture",
        "Prove the Riemann hypothesis",
    ]

    print(f"\n{'PROMPT':<45} {'TIER':<8} {'EFFORT':<8} {'CONF':<6} {'MODE':<6} {'ESC'}")
    print("=" * 90)

    for prompt in test_prompts:
        r = classifier.classify(prompt)
        esc = "⚠️" if r.needs_escalation else ""
        print(f"{prompt[:43]:<45} {r.tier.value:<8} {r.effort.value:<8} "
              f"{r.confidence:.2f}   {r.thinking_mode:<6} {esc}")

    print(f"\nStats: {classifier.get_stats()}")
