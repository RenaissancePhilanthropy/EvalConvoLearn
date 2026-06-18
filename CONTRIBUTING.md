# Contributing to EvalConvoLearn

## Setup

```bash
git clone https://github.com/BaptisteMP/EvalConvoLearn.git
cd EvalConvoLearn
uv sync --extra dev
pre-commit install
cp .env-example .env  # fill in API keys
```

Requires Python ≥ 3.11.

## Development workflow

1. Fork the repo and create a branch from `main` (`git checkout -b my-feature`).
2. Make your changes, keeping `src/evalconvolearn/` as the package root.
3. Run checks before pushing:
   ```bash
   ruff check --fix .
   ruff format .
   pyright
   pytest
   ```
4. Open a pull request against `main` with a clear description of what changed and why.

Pre-commit hooks (ruff, pyright, basic file hygiene) run automatically on each commit.

## Testing

```bash
pytest                          # unit tests
pytest -m integration           # integration tests (requires API keys)
pytest -m "not integration"     # skip integration tests
```

Add tests under `tests/unit/` for new logic and `tests/integration/` for end-to-end flows that call external APIs.

## Code style

- **Formatter / linter**: `ruff` (line length 120, double quotes)
- **Type checker**: `pyright` in `standard` mode — annotate all public functions
- No bare `except:`, no unused imports, no `# type: ignore` without a comment explaining why

## Adding a benchmark

New benchmarks should subclass the appropriate base class in `src/evalconvolearn/benchmarks/` and follow the existing four-family structure (placement test, learning from conversation, multi-conversation, dataset-fitted).

## Reporting issues

Open a GitHub issue with a minimal reproducible example and the versions of `evalconvolearn` and Python you are using.
