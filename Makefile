.PHONY: setup install dev clean help

help:
	@echo "Available commands:"
	@echo "  make setup       - Create venv and install dependencies with uv"
	@echo "  make install     - Install dependencies (venv must exist)"
	@echo "  make dev         - Install with dev dependencies"
	@echo "  make clean       - Remove venv and cache"

setup:
	@echo "Setting up project with uv..."
	uv sync
	@echo "✓ Setup complete. Activate with: source .venv/bin/activate"

install:
	uv sync

dev:
	uv sync --all-extras

clean:
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
