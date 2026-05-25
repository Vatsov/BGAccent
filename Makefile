.PHONY: dev test lint format typecheck

dev:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/
