# Contributing to ponderchat

Thanks for interest in contributing! Here's how you can help.

## Getting Started

1. Clone the repo and set up the environment:
   ```bash
   git clone https://github.com/yourusername/ponderchat.git
   cd ponderchat
   make setup
   source .venv/bin/activate
   ```

2. Make your changes and test them:
   ```bash
   make dev  # Install with dev dependencies
   ```

## What to Contribute

### High Priority
- **Classifier improvements** — Better heuristics for model/effort selection (see `classifier.py`)
- **Threshold tuning** — Data-driven adjustments to `escalate_to_thinking_below` and `escalate_to_user_below`
- **Tests** — Unit and integration tests
- **Edge cases** — Report or fix handling of unusual inputs

### Always Welcome
- Bug reports and fixes
- Documentation improvements
- Example use cases
- Performance optimizations

## Known Limitations

- Classifier is not perfect; see [README](README.md#limitations) for details
- Best effort to route correctly, but no guarantees in production
- Threshold tuning is empirical, not theoretical

## Testing Your Changes

```bash
# Run the classifier on a test prompt
ponderchat "Your test prompt here" --verbose

# Test with different policies
ponderchat --policy cheap "Test"
ponderchat --policy quality "Test"
```

## Submitting Changes

1. Fork the repo
2. Create a branch: `git checkout -b fix/your-fix`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push and open a PR

## Code Style

- Follow PEP 8
- Use descriptive variable names
- Add comments for non-obvious logic
- Keep functions focused and small

## Questions?

Open an issue to discuss larger changes before investing time.

Thanks for contributing!
