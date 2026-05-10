"""
examples.py - Real-world usage examples

Run individual examples:
    python examples.py simple
    python examples.py session
    python examples.py batch
    python examples.py monitor
"""

import sys
from router import PonderChatRouter


def example_simple():
    """Simplest possible usage"""
    print("=" * 60)
    print("EXAMPLE 1: Simple Usage")
    print("=" * 60)

    router = PonderChatRouter(plan="pro", verbose=True)

    result = router.call("What's the capital of France?")
    print(f"\n📝 {result.text}")
    print(f"💰 Cost: ~${result.estimated_cost_usd:.4f}")
    print(f"🎯 Used {result.model_used} ({result.classification.level})")


def example_long_session():
    """Long conversation with auto-management"""
    print("=" * 60)
    print("EXAMPLE 2: Long Session (auto-managed)")
    print("=" * 60)

    router = PonderChatRouter(plan="pro")

    # Series of prompts in a conversation
    prompts = [
        "Help me design a REST API for a todo app",
        "What endpoints would I need?",
        "Add authentication with JWT",
        "How should I structure the database?",
        "Write the user model",
        "Now write the todo model",
        "Add a search endpoint",
        "How do I handle pagination?",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {prompt}")
        result = router.call(prompt)
        print(f"   → {result.classification.level.upper()} ({result.model_used.split('-')[1]}) "
              f"~${result.estimated_cost_usd:.4f}")

    # Show session stats
    print("\n")
    router.print_session_summary()


def example_batch():
    """Batch processing many prompts"""
    print("=" * 60)
    print("EXAMPLE 3: Batch Processing")
    print("=" * 60)

    router = PonderChatRouter(plan="pro")

    prompts = [
        "What's 2+2?",
        "Write a Python function to sort a list",
        "Explain machine learning briefly",
        "Design a microservices architecture",
        "Compare REST vs GraphQL",
        "What's quantum entanglement?",
        "Write code to parse JSON",
        "Analyze the philosophical implications of AI consciousness",
    ]

    total_cost = 0
    for prompt in prompts:
        result = router.call(prompt, use_history=False)  # Don't carry context
        total_cost += result.estimated_cost_usd
        print(f"  '{prompt[:50]:50}' → {result.classification.level:8} "
              f"${result.estimated_cost_usd:.4f}")

    print(f"\n💰 Total: ${total_cost:.4f}")
    router.print_session_summary()


def example_monitor():
    """Just show current usage"""
    print("=" * 60)
    print("EXAMPLE 4: Usage Monitor")
    print("=" * 60)

    router = PonderChatRouter(plan="pro")
    router.print_session_summary()


def example_with_caching():
    """Using prompt caching for long context"""
    print("=" * 60)
    print("EXAMPLE 5: Prompt Caching")
    print("=" * 60)

    # Long system prompt - will be cached automatically
    system = """You are an expert Python developer specializing in:
- Clean code principles (PEP 8, type hints)
- Design patterns (Factory, Singleton, Observer, etc.)
- Performance optimization
- Testing best practices (pytest, mocking)
- Error handling and logging
- Security considerations

[... imagine 2000 more tokens of detailed instructions ...]

When responding, always:
1. Provide working, runnable code
2. Include type hints
3. Add docstrings
4. Suggest tests
5. Consider edge cases
""" * 3  # Make it long enough to trigger caching

    router = PonderChatRouter(plan="pro", enable_prompt_caching=True)

    # First call - creates cache
    result1 = router.call(
        "Write a function to validate email addresses",
        system=system,
    )
    print(f"Call 1: {result1.cache_hit_tokens} cache hits "
          f"(${result1.estimated_cost_usd:.4f})")

    # Subsequent calls - uses cache (90% cheaper)
    result2 = router.call(
        "Write a function to validate phone numbers",
        system=system,
    )
    print(f"Call 2: {result2.cache_hit_tokens} cache hits "
          f"(${result2.estimated_cost_usd:.4f})")

    print(f"\n💡 Cache savings: substantial reduction on system prompt tokens")


def example_force_level():
    """Override classifier for specific use cases"""
    print("=" * 60)
    print("EXAMPLE 6: Force Specific Level")
    print("=" * 60)

    router = PonderChatRouter(plan="pro", verbose=True)

    # Force minimal for known simple tasks
    result = router.call(
        "What is 7 * 9?",
        force_level="minimal",  # Skip classification
    )
    print(f"\n→ Used Haiku, saved on classification time + cost")

    # Force expert for hard tasks
    result = router.call(
        "Explain why this is just a simple question that needs no thinking",
        force_level="expert",  # Override automatic
    )
    print(f"\n→ Forced Opus with max thinking despite simple-looking prompt")


# ============================================================================
# RUN
# ============================================================================

EXAMPLES = {
    "simple": example_simple,
    "session": example_long_session,
    "batch": example_batch,
    "monitor": example_monitor,
    "cache": example_with_caching,
    "force": example_force_level,
}


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in EXAMPLES:
        print("Available examples:")
        for name, func in EXAMPLES.items():
            print(f"  python examples.py {name:10s} - {func.__doc__}")
        sys.exit(1)

    EXAMPLES[sys.argv[1]]()
