.PHONY: dev test lint format typecheck

dev:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/ scripts/ tools/

format:
	uv run ruff format src/ tests/ scripts/ tools/

typecheck:
	uv run mypy src/
