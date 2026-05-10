# smartclaude

CLI that wraps Claude with smart routing — picks the right model and reasoning depth per prompt, falls back to a thinking-mode local model on uncertain cases, and asks you when it's genuinely ambiguous.

## How it actually works

```
Your prompt
    ↓
┌─ Tier 1: Gemma 4 E4B, thinking off  (~100-150ms)
│  └─ if confidence < 0.75
├─ Tier 2: Gemma 4 E4B, thinking on   (~500-800ms)
│  └─ if confidence < 0.55
└─ Tier 3: Ask you                     (interactive disambiguation)
    ↓
Routes to Claude with chosen tier × effort
```

One model loaded (~3.5GB), three reliability levels. Average latency stays low because most prompts are confident on Tier 1.

## Two routing dimensions

Every Claude API call has two independent decisions:

**Tier (which model)** — haiku · sonnet · opus
**Effort (how hard to think)** — low · medium · high · xhigh · max

That's 15 combinations. The classifier picks both per prompt.

## Install

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the CLI
./install.sh

# 3. Set API key
export ANTHROPIC_API_KEY="your-key-here"

# 4. First run will download Gemma 4 E4B (~3.5GB)
smartclaude "What's 2+2?"
```

## Usage

### Auto routing (default)

```bash
smartclaude "Refactor this function"
# → sonnet + medium (auto-picked)
```

### See the classifier work

```bash
$ smartclaude "Implement a custom comparator for sorting"
┌─ sonnet + medium effort · 58% conf · mlx · 92ms
└─ code_or_math, multi_step_logic
  ↳ Confidence 0.58 below threshold, escalating to thinking mode
┌─ sonnet + high effort · 72% conf · 🧠full · mlx · 612ms
└─ code_or_math, nuanced_judgment

⚠ Routing uncertain (confidence: 72%)
  Custom comparator may have subtle correctness issues - depends on complexity

  1. sonnet + high  (classifier's pick)
  2. opus + high    (alternative)

Choose: [1/2/q] _
```

### Force a specific tier or effort

```bash
smartclaude --tier opus --effort max "complex problem"   # Force both
smartclaude --tier sonnet "any prompt"                   # Force tier, auto effort
smartclaude --effort high "any prompt"                   # Force effort, auto tier
smartclaude --no-escalate "..."                          # Skip user escalation
```

### Apply a policy

```bash
smartclaude --policy cheap "..."     # Cap at sonnet, max effort medium
smartclaude --policy balanced "..."  # Downgrade xhigh/max → high
smartclaude --policy quality "..."   # Min sonnet for high+ effort
smartclaude --policy max "..."       # Always opus + max effort
```

### Interactive mode

```bash
$ smartclaude
[0] > Help me design a database schema
┌─ sonnet + high effort · 88% conf · mlx · 95ms

[1] > Add user authentication
[2] > /tier opus
[3] > /effort max
[4] > Now think deeply about consistency guarantees
[5] > /usage
[6] > /quit
```

Commands: `/usage` `/reset` `/tier <name>` `/effort <name>` `/policy <name>` `/quit`

## Why Gemma 4 E4B

Released April 2026, designed specifically for agentic JSON workflows:

- **Native thinking mode** via `enable_thinking=True` chat template kwarg
- **Channel-based output** for clean reasoning/answer separation
- **3.5GB at 4-bit** quantization — fits easily on 32GB Mac
- **Apache 2.0** license — clean for any use
- **Strong instruction following** — reliable JSON output

The hybrid thinking lets us trade latency for accuracy on demand. Easy prompts: 100ms. Hard prompts: 500ms with proper reasoning. Genuinely ambiguous: ask the user.

## Multi-turn conversations

Each turn is a fresh classification with two important behaviors:

**Context-aware** — classifier sees recent conversation, not just new prompt:

```
Turn 1: "Help me design a distributed trading system"  → opus + high
Turn 2: "What about caching?"                          → opus + high
        (recognized as deep follow-up in design thread)
Turn 3: "Fix this typo"                                → haiku + low
        (genuinely trivial, drops out)
```

**Tier inertia** — won't aggressively bounce models. Drops at most one tier per turn unless confidence is high. Prevents a brief follow-up from collapsing out of a complex thread.

## CLI flags

```
Routing:
  --tier              haiku | sonnet | opus
  --effort            low | medium | high | xhigh | max
  --policy            auto | cheap | balanced | quality | max

Behavior:
  --max-tokens        Max output tokens (default: 4096)
  --system-file       File with system prompt (cached if >1024 tokens)
  --session           Named session (default: "default")
  --plan              free | pro | max_5x | max_20x
  --no-history        Don't use conversation history
  --no-escalate       Skip user escalation on uncertain prompts
  --quiet, -q         Suppress info output
  --verbose, -v       Show internal decisions

Utility:
  --usage             Show usage stats and exit
  --reset             Reset session and exit
```

## Configuration

Tune the cascade thresholds in `classifier.py`:

```python
ReliableClassifier(
    escalate_to_thinking_below=0.75,  # Trigger thinking-mode retry
    escalate_to_user_below=0.55,      # Trigger user escalation
)
```

Lower = more conservative (asks user more often). Higher = more aggressive (trusts itself more).

## Plans (post May 6, 2026 doubling)

| Plan | 5h Limit |
|------|----------|
| free | ~9 messages |
| pro | ~90 messages |
| max_5x | ~450 messages |
| max_20x | ~1800 messages |

## Privacy

Data flow:
1. Your prompt → local Gemma 4 on your machine (no network)
2. Your prompt → Anthropic's Claude API (HTTPS, per their privacy policy)
3. Conversation history → `~/.smartclaude/<session>/` (local)
4. Classification cache → `~/.classifier_cache/` (local)

No telemetry. Nothing else sent anywhere.

## Limitations

- **Classifier can misclassify.** Cascade and escalation reduce this but don't eliminate it.
- **Cost estimates are approximate.** Reconcile with Anthropic billing for actuals.
- **First run downloads ~3.5GB** (Gemma 4 E4B 4-bit) from Hugging Face.
- **Inertia can persist a wrong tier choice** through a thread. Use `/tier auto` to reset.

## Disclaimers

**This is an unofficial tool. Not affiliated with, endorsed by, or sponsored by Anthropic or Google.** Uses the official Claude API following Anthropic's terms of service. Uses Gemma 4 under its Apache 2.0 license.

**Trade-off: Speed vs. Optimality** — ponderchat uses a local classifier for fast decisions, but Gemma 4 isn't perfect. The cascade (thinking mode + user escalation) reduces errors, but doesn't eliminate them. In production systems where you absolutely need the optimal model every time, consider:
- Testing the classifier on your workloads first
- Using escalation mode (`--no-escalate` disabled) to catch uncertain cases
- Tuning thresholds in `classifier.py` for your use case
- Monitoring routing decisions over time

For individual use and cost optimization, this works great. For critical systems, validate the approach first.

Provided as-is, no warranty. Use at your own risk.

## Files

```
classifier.py        Gemma 4 classifier with hybrid thinking cascade
session_manager.py   Token tracking & plan limits
router.py           Smart Claude API integration with adaptive thinking
smartclaude         CLI entry point
install.sh          Installer
requirements.txt    pip dependencies
```

## License

MIT. See [LICENSE](LICENSE) file.
