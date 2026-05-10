"""
router.py - Smart Claude Router with 2D classification

Uses the new (tier × effort) classification:
- tier: which model (haiku/sonnet/opus)
- effort: thinking depth (low/medium/high/xhigh/max)

These are independent, giving 15 possible combinations instead of 5.
"""

import os
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from classifier import (
    ReliableClassifier, ClassificationResult,
    CapabilityTier, Effort, MODEL_IDS, PRICING
)
from session_manager import SessionManager


@dataclass
class RouterResponse:
    text: str
    classification: ClassificationResult
    model_used: str
    effort_used: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cache_hit_tokens: int
    estimated_cost_usd: float
    full_response: Any = None


class PonderChatRouter:
    """
    Smart Claude API router with 2D routing.

    Args:
        plan: Your Claude plan (free/pro/max_5x/max_20x)
        classifier_model: Local LLM for classification
        policy: Routing policy (auto/cheap/balanced/quality/max)
        force_tier: Override tier selection (haiku/sonnet/opus)
        force_effort: Override effort selection
        enable_prompt_caching: Use Anthropic prompt caching
        max_context_messages: Compress beyond this many turns
        verbose: Show classification decisions
    """

    def __init__(
        self,
        plan: str = "pro",
        classifier_model: str = "Qwen/Qwen-7B-Chat",
        policy: str = "auto",
        force_tier: Optional[str] = None,
        force_effort: Optional[str] = None,
        enable_prompt_caching: bool = True,
        max_context_messages: int = 20,
        verbose: bool = False,
    ):
        self.classifier = ReliableClassifier(
            model_name=classifier_model,
            policy=policy,
            verbose=verbose,
        )
        self.session = SessionManager(plan=plan)
        self.force_tier = force_tier
        self.force_effort = force_effort
        self.enable_prompt_caching = enable_prompt_caching
        self.max_context_messages = max_context_messages
        self.verbose = verbose

        self.conversation_history: List[Dict] = []
        self.last_classification: Optional[ClassificationResult] = None  # for inertia
        self.client = self._init_anthropic()

    def _init_anthropic(self):
        try:
            import anthropic
            return anthropic.Anthropic()
        except ImportError:
            print("⚠️  anthropic SDK not installed. Run: pip install anthropic")
            return None
        except Exception as e:
            print(f"⚠️  Couldn't init Anthropic: {e}")
            return None

    def call(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        use_history: bool = True,
        force_tier: Optional[str] = None,
        force_effort: Optional[str] = None,
    ) -> RouterResponse:
        """
        Call Claude with smart 2D routing.

        Args:
            prompt: User prompt
            system: Optional system prompt
            max_tokens: Max output tokens
            use_history: Include conversation history
            force_tier: Override tier (haiku/sonnet/opus)
            force_effort: Override effort (low/medium/high/xhigh/max)
        """
        # Step 1: Classify with conversation context (or use forced values)
        prev_tier = self.last_classification.tier.value if self.last_classification else None
        prev_effort = self.last_classification.effort.value if self.last_classification else None
        history_for_classifier = self.conversation_history if use_history else None

        if force_tier or force_effort or self.force_tier or self.force_effort:
            classification = self.classifier.classify(
                prompt,
                history=history_for_classifier,
                previous_tier=prev_tier,
                previous_effort=prev_effort,
            )
            tier = force_tier or self.force_tier or classification.tier.value
            effort = force_effort or self.force_effort or classification.effort.value
            classification.tier = CapabilityTier(tier)
            classification.effort = Effort(effort)
            classification.model_id = MODEL_IDS[CapabilityTier(tier)]
        else:
            classification = self.classifier.classify(
                prompt,
                history=history_for_classifier,
                previous_tier=prev_tier,
                previous_effort=prev_effort,
            )

        # Remember for inertia on next turn
        self.last_classification = classification

        if self.verbose:
            transition = ""
            if prev_tier and prev_tier != classification.tier.value:
                transition = f" (was {prev_tier})"
            print(f"🔍 {classification.tier.value.upper()}{transition} + {classification.effort.value} effort "
                  f"({classification.confidence:.0%} confidence, "
                  f"{classification.latency_ms:.0f}ms)")

        # Step 2: Build messages
        messages = self._build_messages(prompt, use_history)
        messages = self._compress_if_needed(messages)

        # Step 3: Call Claude
        if self.client is None:
            return self._mock_response(classification, prompt)

        response = self._call_claude(
            messages=messages,
            tier=classification.tier,
            effort=classification.effort,
            system=system,
            max_tokens=max_tokens,
        )

        # Step 4: Update history
        if use_history:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response["text"]})

        # Step 5: Record usage
        usage = self.session.record_call(
            model=classification.model_id,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            thinking_tokens=response["thinking_tokens"],
            cache_read_tokens=response["cache_hit_tokens"],
            classification=f"{classification.tier.value}/{classification.effort.value}",
        )

        return RouterResponse(
            text=response["text"],
            classification=classification,
            model_used=classification.model_id,
            effort_used=classification.effort.value,
            input_tokens=response["input_tokens"],
            output_tokens=response["output_tokens"],
            thinking_tokens=response["thinking_tokens"],
            cache_hit_tokens=response["cache_hit_tokens"],
            estimated_cost_usd=usage.cost_estimate_usd,
            full_response=response.get("full"),
        )

    def _build_messages(self, prompt: str, use_history: bool) -> List[Dict]:
        if use_history and self.conversation_history:
            return self.conversation_history + [{"role": "user", "content": prompt}]
        return [{"role": "user", "content": prompt}]

    def _compress_if_needed(self, messages: List[Dict]) -> List[Dict]:
        if len(messages) <= self.max_context_messages:
            return messages

        keep_recent = self.max_context_messages - 2
        old = messages[:-keep_recent]
        recent = messages[-keep_recent:]

        if self.verbose:
            print(f"📦 Compressing {len(old)} old messages")

        return [{
            "role": "user",
            "content": f"[Previous context: {len(old)} messages summarized]"
        }] + recent

    def _call_claude(
        self,
        messages: List[Dict],
        tier: CapabilityTier,
        effort: Effort,
        system: Optional[str],
        max_tokens: int,
    ) -> Dict:
        """Make API call with adaptive thinking using effort parameter"""
        kwargs = {
            "model": MODEL_IDS[tier],
            "max_tokens": max_tokens,
            "messages": messages,
        }

        # System prompt with caching for long prompts
        if system:
            if self.enable_prompt_caching and len(system) > 1024:
                kwargs["system"] = [{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = system

        # Adaptive thinking with effort parameter
        # (replaces deprecated budget_tokens)
        if effort != Effort.LOW or tier == CapabilityTier.OPUS:
            kwargs["thinking"] = {
                "type": "adaptive",
                "effort": effort.value,
            }

        try:
            response = self.client.messages.create(**kwargs)

            text_parts = []
            thinking_tokens = 0
            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "thinking":
                        thinking_tokens += len(getattr(block, "thinking", "")) // 4

            text = "".join(text_parts)
            usage = response.usage

            return {
                "text": text,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "thinking_tokens": thinking_tokens or getattr(usage, "thinking_tokens", 0),
                "cache_hit_tokens": getattr(usage, "cache_read_input_tokens", 0),
                "full": response,
            }
        except Exception as e:
            print(f"❌ Claude API error: {e}")
            return {
                "text": f"[Error: {e}]",
                "input_tokens": 0, "output_tokens": 0,
                "thinking_tokens": 0, "cache_hit_tokens": 0,
            }

    def _mock_response(self, classification: ClassificationResult, prompt: str) -> RouterResponse:
        return RouterResponse(
            text=f"[Mock] Would call {classification.model_id} with effort={classification.effort.value}",
            classification=classification,
            model_used=classification.model_id,
            effort_used=classification.effort.value,
            input_tokens=0, output_tokens=0, thinking_tokens=0,
            cache_hit_tokens=0, estimated_cost_usd=0.0,
        )

    def reset_conversation(self):
        self.conversation_history = []
        self.last_classification = None

    def print_session_summary(self):
        self.session.print_summary()
        print("\n💡 TIPS:")
        for tip in self.session.get_optimization_tips():
            print(f"  {tip}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        router = PonderChatRouter(verbose=True)
        result = router.call(prompt)

        print(f"\n📝 Response:\n{result.text}")
        print(f"\n📊 Stats:")
        print(f"  Tier: {result.classification.tier.value}")
        print(f"  Effort: {result.effort_used}")
        print(f"  Model: {result.model_used}")
        print(f"  Tokens: {result.input_tokens}↓ {result.output_tokens}↑ {result.thinking_tokens}🧠")
        print(f"  Cost: ~${result.estimated_cost_usd:.4f}")
    else:
        print("Usage: python router.py 'your prompt'")
