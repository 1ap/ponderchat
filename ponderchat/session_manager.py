"""
session_manager.py - Token Tracking & Plan Limit Management

For long Claude Code sessions. Tracks usage, warns before limits,
and helps optimize token consumption.

Key features:
- Track token usage per session
- Warn before approaching limits
- Suggest optimizations
- Export usage reports
- Persistent state across restarts
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict


# ============================================================================
# PLAN LIMITS (approximate - check current plan)
# ============================================================================

PLAN_LIMITS = {
    "free": {
        "messages_per_5h": 9,
        "context_window": 100_000,
    },
    "pro": {
        "messages_per_5h": 45,
        "context_window": 200_000,
        "weekly_limit": 432_000,  # rough estimate
    },
    "max_5x": {
        "messages_per_5h": 225,
        "context_window": 200_000,
        "weekly_limit": 2_160_000,
    },
    "max_20x": {
        "messages_per_5h": 900,
        "context_window": 200_000,
        "weekly_limit": 8_640_000,
    },
}


@dataclass
class TokenUsage:
    """Track tokens for a single API call"""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    prompt_classification: Optional[str] = None  # The classification level used

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.thinking_tokens

    @property
    def cost_estimate_usd(self) -> float:
        """Rough cost estimate (approximate Claude pricing)"""
        # Rough estimates per 1M tokens
        pricing = {
            "claude-haiku": {"input": 1.0, "output": 5.0},
            "claude-sonnet": {"input": 3.0, "output": 15.0},
            "claude-opus": {"input": 15.0, "output": 75.0},
        }

        # Pick price tier
        if "haiku" in self.model.lower():
            tier = "claude-haiku"
        elif "sonnet" in self.model.lower():
            tier = "claude-sonnet"
        else:
            tier = "claude-opus"

        rates = pricing[tier]
        cost = (self.input_tokens * rates["input"] / 1_000_000 +
                self.output_tokens * rates["output"] / 1_000_000 +
                self.thinking_tokens * rates["output"] / 1_000_000)

        # Cached tokens are 90% cheaper
        if self.cache_read_tokens > 0:
            cost -= self.cache_read_tokens * rates["input"] * 0.9 / 1_000_000

        return cost


class SessionManager:
    """
    Manages token usage across Claude Code sessions.

    Helps you:
    - Stay within plan limits
    - Optimize token usage
    - Track costs
    - Get warnings before hitting limits
    """

    def __init__(
        self,
        plan: str = "pro",
        state_file: str = ".claude_session.json",
        warn_at_percent: float = 0.8,
    ):
        self.plan = plan
        self.limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["pro"])
        self.state_file = Path(state_file)
        self.warn_at_percent = warn_at_percent

        self.usage_history: List[TokenUsage] = []
        self._load_state()

    def _load_state(self):
        """Load previous session state"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                self.usage_history = [TokenUsage(**u) for u in data.get("history", [])]
            except Exception:
                pass

    def _save_state(self):
        """Persist session state"""
        try:
            with open(self.state_file, "w") as f:
                json.dump({
                    "history": [asdict(u) for u in self.usage_history],
                    "saved_at": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: couldn't save state: {e}")

    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        thinking_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        classification: Optional[str] = None,
    ) -> TokenUsage:
        """Record an API call and check limits"""
        usage = TokenUsage(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            prompt_classification=classification,
        )

        self.usage_history.append(usage)
        self._save_state()
        self._check_limits()
        return usage

    def _check_limits(self):
        """Check if approaching limits and warn"""
        recent = self.usage_in_last_hours(5)
        msgs_in_window = len(recent)

        limit = self.limits.get("messages_per_5h", 45)

        if msgs_in_window >= limit * self.warn_at_percent:
            pct = msgs_in_window / limit * 100
            remaining = limit - msgs_in_window
            print(f"\n⚠️  PLAN LIMIT WARNING")
            print(f"   {msgs_in_window}/{limit} messages used in 5h window ({pct:.0f}%)")
            print(f"   {remaining} remaining before limit\n")

    def usage_in_last_hours(self, hours: int) -> List[TokenUsage]:
        """Get usage in last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            u for u in self.usage_history
            if datetime.fromisoformat(u.timestamp) > cutoff
        ]

    def get_session_summary(self) -> Dict:
        """Get current session statistics"""
        recent_5h = self.usage_in_last_hours(5)
        recent_24h = self.usage_in_last_hours(24)
        recent_7d = self.usage_in_last_hours(168)

        total_tokens = sum(u.total_tokens for u in recent_24h)
        total_cost = sum(u.cost_estimate_usd for u in recent_24h)

        # By classification level
        by_level = defaultdict(int)
        for u in recent_24h:
            if u.prompt_classification:
                by_level[u.prompt_classification] += 1

        # By model
        by_model = defaultdict(int)
        for u in recent_24h:
            by_model[u.model.split("-")[1] if "-" in u.model else u.model] += 1

        # Cache savings
        cache_reads = sum(u.cache_read_tokens for u in recent_24h)

        return {
            "plan": self.plan,
            "messages_5h": len(recent_5h),
            "messages_24h": len(recent_24h),
            "messages_7d": len(recent_7d),
            "messages_5h_limit": self.limits.get("messages_per_5h", 45),
            "messages_5h_remaining": max(0, self.limits.get("messages_per_5h", 45) - len(recent_5h)),
            "tokens_24h": total_tokens,
            "estimated_cost_24h_usd": round(total_cost, 2),
            "cache_reads_24h": cache_reads,
            "by_level": dict(by_level),
            "by_model": dict(by_model),
        }

    def print_summary(self):
        """Print human-readable summary"""
        s = self.get_session_summary()
        print("\n" + "=" * 60)
        print(f"📊 SESSION USAGE - Plan: {s['plan'].upper()}")
        print("=" * 60)
        print(f"Last 5 hours:  {s['messages_5h']:3d} / {s['messages_5h_limit']:3d} messages")
        print(f"               {'█' * int(s['messages_5h']/s['messages_5h_limit']*30):<30} {s['messages_5h']/s['messages_5h_limit']*100:.0f}%")
        print(f"Last 24 hours: {s['messages_24h']} messages, {s['tokens_24h']:,} tokens")
        print(f"Last 7 days:   {s['messages_7d']} messages")
        print(f"\n💰 Cost (24h): ~${s['estimated_cost_24h_usd']:.2f}")
        print(f"💾 Cache reads (24h): {s['cache_reads_24h']:,} tokens (saved ~90%)")

        if s['by_level']:
            print(f"\nBy classification level:")
            for level, count in sorted(s['by_level'].items()):
                print(f"  {level:10} {count:3d}")

        if s['by_model']:
            print(f"\nBy model:")
            for model, count in sorted(s['by_model'].items()):
                print(f"  {model:10} {count:3d}")

        print(f"\n✅ {s['messages_5h_remaining']} messages remaining in 5h window")
        print("=" * 60)

    def get_optimization_tips(self) -> List[str]:
        """Get actionable tips to reduce token usage"""
        tips = []
        s = self.get_session_summary()

        # Check level distribution
        by_level = s.get("by_level", {})
        total = sum(by_level.values()) if by_level else 1

        deep_pct = (by_level.get("deep", 0) + by_level.get("expert", 0)) / total
        if deep_pct > 0.5:
            tips.append(
                "💡 Over 50% of prompts use deep/expert thinking. "
                "Consider simpler models for some prompts to save tokens."
            )

        # Check cache usage
        if s["cache_reads_24h"] < 1000 and s["messages_24h"] > 10:
            tips.append(
                "💡 Low cache usage. Use prompt caching for repeated context "
                "(system prompts, large docs) to save 90% on those tokens."
            )

        # Check model distribution
        by_model = s.get("by_model", {})
        opus_pct = by_model.get("opus", 0) / total if total else 0
        if opus_pct > 0.7:
            tips.append(
                "💡 Heavy Opus usage. Many tasks work fine with Sonnet "
                "(5x cheaper). Use classifier to route simpler tasks."
            )

        # Check if approaching limits
        if s["messages_5h"] / s["messages_5h_limit"] > 0.7:
            tips.append(
                f"⚠️  Approaching 5h limit ({s['messages_5h']}/{s['messages_5h_limit']}). "
                "Consider batching or pausing."
            )

        if not tips:
            tips.append("✓ Usage looks healthy - no obvious optimizations needed.")

        return tips

    def reset(self):
        """Reset session history"""
        self.usage_history = []
        self._save_state()


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    sm = SessionManager(plan="pro")

    # Simulate some usage
    print("Simulating session usage...")
    sm.record_call("claude-haiku-4-5", 100, 200, 0, classification="minimal")
    sm.record_call("claude-sonnet-4-6", 500, 1000, 2000, classification="standard")
    sm.record_call("claude-opus-4-7", 2000, 3000, 10000, classification="deep")
    sm.record_call("claude-opus-4-7", 1500, 2500, 8000, classification="deep")

    sm.print_summary()

    print("\n💡 OPTIMIZATION TIPS:")
    for tip in sm.get_optimization_tips():
        print(f"  {tip}")
