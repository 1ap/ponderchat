#!/usr/bin/env python3
"""
smartclaude - Smart Claude CLI with 2D routing

Two independent dimensions:
  --tier   haiku | sonnet | opus       (which model)
  --effort low | medium | high | xhigh | max  (how hard to think)

Usage:
    smartclaude "your question"                   # auto routing
    smartclaude --tier opus --effort max "..."    # force both
    smartclaude --tier sonnet "..."               # force tier, auto effort
    smartclaude --policy cheap "..."              # apply cost policy
    smartclaude                                    # interactive mode
    smartclaude --usage                            # show stats
"""

import argparse
import sys
import os
from pathlib import Path


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"

    @staticmethod
    def disable():
        for attr in dir(C):
            if not attr.startswith("_") and isinstance(getattr(C, attr), str):
                setattr(C, attr, "")


if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    C.disable()


TIER_COLORS = {"haiku": C.GREEN, "sonnet": C.CYAN, "opus": C.MAGENTA}
EFFORT_COLORS = {
    "low": C.GRAY, "medium": C.GREEN, "high": C.YELLOW,
    "xhigh": C.MAGENTA, "max": C.RED,
}


def print_classification(classification):
    tier = classification.tier.value
    effort = classification.effort.value
    tc = TIER_COLORS.get(tier, C.RESET)
    ec = EFFORT_COLORS.get(effort, C.RESET)

    types = ", ".join(classification.reasoning_types.keys()) if classification.reasoning_types else ""

    # Mode indicator
    mode = classification.thinking_mode
    mode_marker = ""
    if mode == "full":
        mode_marker = f" {C.MAGENTA}🧠full{C.RESET}{C.DIM}"
    elif mode == "brief":
        mode_marker = f" {C.YELLOW}🧠brief{C.RESET}{C.DIM}"

    line = (
        f"{C.DIM}┌─ {tc}{tier}{C.RESET}{C.DIM} + {ec}{effort}{C.RESET}{C.DIM} effort"
        f" · {classification.confidence:.0%} conf"
        f"{mode_marker}"
        f" · {classification.method}"
        f" · {classification.latency_ms:.0f}ms{C.RESET}"
    )
    print(line, file=sys.stderr)
    if types:
        print(f"{C.DIM}└─ {types}{C.RESET}", file=sys.stderr)


def escalate_to_user(classification):
    """Ask user to disambiguate when classifier is uncertain"""
    primary_tier = classification.tier.value
    primary_effort = classification.effort.value
    primary_cost = primary_tier + " + " + primary_effort

    alt = classification.alternative
    has_alt = alt and (alt.get("tier") != primary_tier or alt.get("effort") != primary_effort)

    print(f"\n{C.YELLOW}⚠ Routing uncertain{C.RESET} "
          f"({C.DIM}confidence: {classification.confidence:.0%}{C.RESET})",
          file=sys.stderr)

    if classification.ambiguity:
        print(f"{C.DIM}  {classification.ambiguity}{C.RESET}", file=sys.stderr)

    if classification.explanation:
        print(f"{C.DIM}  {classification.explanation}{C.RESET}", file=sys.stderr)

    print("", file=sys.stderr)
    print(f"  {C.BOLD}1.{C.RESET} {primary_cost}{C.DIM}  (classifier's pick){C.RESET}",
          file=sys.stderr)

    if has_alt:
        alt_cost = alt["tier"] + " + " + alt["effort"]
        print(f"  {C.BOLD}2.{C.RESET} {alt_cost}{C.DIM}  (alternative){C.RESET}",
              file=sys.stderr)
        prompt_str = "[1/2/q] "
    else:
        prompt_str = "[Enter to accept, q to quit] "

    print("", file=sys.stderr)

    try:
        choice = input(f"{C.CYAN}Choose: {prompt_str}{C.RESET}").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{C.YELLOW}Cancelled{C.RESET}", file=sys.stderr)
        return None

    if choice in ("q", "quit"):
        return None
    if choice == "2" and has_alt:
        return (alt["tier"], alt["effort"])
    # Default: accept primary
    return (primary_tier, primary_effort)


def print_footer(result):
    tokens = f"{result.input_tokens}↓ {result.output_tokens}↑"
    if result.thinking_tokens:
        tokens += f" {result.thinking_tokens}🧠"
    if result.cache_hit_tokens:
        tokens += f" {result.cache_hit_tokens}💾"

    print(f"{C.DIM}── {tokens} · ${result.estimated_cost_usd:.4f}{C.RESET}",
          file=sys.stderr)


def cmd_oneshot(args, router):
    prompt = args.prompt
    if not sys.stdin.isatty() and not prompt:
        prompt = sys.stdin.read().strip()

    if not prompt:
        print(f"{C.RED}No prompt provided{C.RESET}", file=sys.stderr)
        sys.exit(1)

    system = None
    if args.system_file:
        system = Path(args.system_file).read_text()

    try:
        # First pass - may produce escalation flag
        result = router.call(
            prompt,
            system=system,
            max_tokens=args.max_tokens,
            use_history=not args.no_history,
            force_tier=args.tier,
            force_effort=args.effort,
        )

        # Handle escalation if needed
        if result.classification.needs_escalation and not args.no_escalate:
            choice = escalate_to_user(result.classification)
            if choice is None:
                print(f"{C.YELLOW}Aborted{C.RESET}", file=sys.stderr)
                sys.exit(1)

            user_tier, user_effort = choice
            # Re-call with user's choice if different
            if (user_tier != result.classification.tier.value or
                user_effort != result.classification.effort.value):
                # Pop the previous turn from history (we're redoing it)
                if router.conversation_history:
                    router.conversation_history = router.conversation_history[:-2]
                result = router.call(
                    prompt, system=system, max_tokens=args.max_tokens,
                    use_history=not args.no_history,
                    force_tier=user_tier, force_effort=user_effort,
                )

        if not args.quiet:
            print_classification(result.classification)

        print()
        print(result.text)
        print()

        if not args.quiet:
            print_footer(result)

    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted{C.RESET}", file=sys.stderr)
        sys.exit(130)


def cmd_interactive(args, router):
    print(f"{C.BOLD}smartclaude{C.RESET} {C.DIM}— interactive mode{C.RESET}")
    print(f"{C.DIM}Commands: /usage /reset /quit /tier <name> /effort <name> /policy <name>{C.RESET}\n")

    forced_tier = args.tier
    forced_effort = args.effort

    while True:
        try:
            count = len(router.conversation_history) // 2
            prompt = input(f"{C.CYAN}[{count}] >{C.RESET} ").strip()

            if not prompt:
                continue

            if prompt.startswith("/"):
                parts = prompt.split()
                cmd = parts[0]

                if cmd in ("/quit", "/exit", "/q"):
                    print(f"{C.DIM}Goodbye{C.RESET}")
                    break
                elif cmd == "/usage":
                    router.print_session_summary()
                    continue
                elif cmd == "/reset":
                    router.reset_conversation()
                    print(f"{C.DIM}Conversation reset{C.RESET}")
                    continue
                elif cmd == "/tier":
                    if len(parts) == 2 and parts[1] in ("haiku", "sonnet", "opus", "auto"):
                        forced_tier = None if parts[1] == "auto" else parts[1]
                        print(f"{C.DIM}Tier: {parts[1]}{C.RESET}")
                    else:
                        print(f"{C.DIM}Usage: /tier <haiku|sonnet|opus|auto>{C.RESET}")
                    continue
                elif cmd == "/effort":
                    if len(parts) == 2 and parts[1] in ("low", "medium", "high", "xhigh", "max", "auto"):
                        forced_effort = None if parts[1] == "auto" else parts[1]
                        print(f"{C.DIM}Effort: {parts[1]}{C.RESET}")
                    else:
                        print(f"{C.DIM}Usage: /effort <low|medium|high|xhigh|max|auto>{C.RESET}")
                    continue
                elif cmd == "/policy":
                    if len(parts) == 2 and parts[1] in ("auto", "cheap", "balanced", "quality", "max"):
                        router.classifier.policy = parts[1]
                        print(f"{C.DIM}Policy: {parts[1]}{C.RESET}")
                    else:
                        print(f"{C.DIM}Usage: /policy <auto|cheap|balanced|quality|max>{C.RESET}")
                    continue
                elif cmd == "/help":
                    print(f"""
{C.BOLD}Commands:{C.RESET}
  /usage              Show usage stats
  /reset              Clear conversation
  /tier <name>        Force tier (haiku/sonnet/opus/auto)
  /effort <name>      Force effort (low/medium/high/xhigh/max/auto)
  /policy <name>      Set policy (auto/cheap/balanced/quality/max)
  /quit               Exit
""")
                    continue
                else:
                    print(f"{C.YELLOW}Unknown command. Type /help{C.RESET}")
                    continue

            result = router.call(
                prompt,
                max_tokens=args.max_tokens,
                force_tier=forced_tier,
                force_effort=forced_effort,
            )

            # Handle escalation in interactive mode
            if result.classification.needs_escalation and not args.no_escalate:
                choice = escalate_to_user(result.classification)
                if choice is None:
                    print(f"{C.YELLOW}Skipped{C.RESET}", file=sys.stderr)
                    continue
                user_tier, user_effort = choice
                if (user_tier != result.classification.tier.value or
                    user_effort != result.classification.effort.value):
                    if router.conversation_history:
                        router.conversation_history = router.conversation_history[:-2]
                    result = router.call(
                        prompt, max_tokens=args.max_tokens,
                        force_tier=user_tier, force_effort=user_effort,
                    )

            print_classification(result.classification)
            print()
            print(result.text)
            print()
            print_footer(result)

        except KeyboardInterrupt:
            print(f"\n{C.DIM}(Ctrl+D or /quit to exit){C.RESET}")
            continue
        except EOFError:
            print(f"\n{C.DIM}Goodbye{C.RESET}")
            break


def main():
    parser = argparse.ArgumentParser(
        prog="smartclaude",
        description="Smart Claude CLI with 2D routing (tier × effort)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Two independent dimensions:
  --tier         which Claude model
  --effort       how hard to think

Examples:
  smartclaude "What's 2+2?"                    Auto: haiku + low
  smartclaude "Hard math problem"              Auto: opus + high
  smartclaude --tier sonnet "any prompt"       Force sonnet, auto effort
  smartclaude --tier opus --effort max "..."   Force both
  smartclaude --policy cheap "..."             Apply cost policy
  smartclaude --policy quality "..."           Apply quality policy

Policies:
  auto      Use classifier output as-is (default)
  cheap     Cap at sonnet, max effort medium
  balanced  Downgrade xhigh/max to high
  quality   Min sonnet for high+ effort
  max       Always opus + max effort

Plans: free, pro, max_5x, max_20x
""",
    )

    parser.add_argument("prompt", nargs="?", help="Your prompt (omit for interactive)")
    parser.add_argument("--tier", choices=["haiku", "sonnet", "opus"],
                       help="Force model tier")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                       help="Force effort level")
    parser.add_argument("--policy", default="auto",
                       choices=["auto", "cheap", "balanced", "quality", "max"],
                       help="Routing policy (default: auto)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens")
    parser.add_argument("--system-file", help="File with system prompt")
    parser.add_argument("--session", default="default", help="Named session")
    parser.add_argument("--plan", default=os.environ.get("SMARTCLAUDE_PLAN", "pro"),
                       choices=["free", "pro", "max_5x", "max_20x"],
                       help="Your Claude plan")
    parser.add_argument("--no-history", action="store_true", help="Don't use history")
    parser.add_argument("--no-escalate", action="store_true",
                       help="Skip user escalation on uncertain classifications")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress info output")
    parser.add_argument("--usage", action="store_true", help="Show usage stats and exit")
    parser.add_argument("--reset", action="store_true", help="Reset session and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show internal decisions")

    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from router import SmartClaudeRouter
    except ImportError as e:
        print(f"{C.RED}Error: {e}{C.RESET}", file=sys.stderr)
        sys.exit(1)

    session_dir = Path.home() / ".smartclaude" / args.session
    session_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(session_dir)

    router = SmartClaudeRouter(
        plan=args.plan,
        policy=args.policy,
        force_tier=args.tier,
        force_effort=args.effort,
        verbose=args.verbose,
    )

    if args.reset:
        confirm = input(f"{C.YELLOW}Reset session? [y/N] {C.RESET}").strip().lower()
        if confirm == "y":
            router.session.reset()
            router.reset_conversation()
            print(f"{C.GREEN}Reset{C.RESET}")
    elif args.usage:
        router.print_session_summary()
    elif args.prompt or not sys.stdin.isatty():
        cmd_oneshot(args, router)
    else:
        cmd_interactive(args, router)


if __name__ == "__main__":
    main()
