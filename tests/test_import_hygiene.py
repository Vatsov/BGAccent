"""Guard against eager imports of optional heavy deps in the test suite.

Importing ``torch`` / ``onnx`` / ``onnxruntime`` / ``spacy`` — or
``scripts.train_stress_model``, which imports torch — at module scope makes
``pytest`` fail at *collection* in a clean ``uv sync --dev`` install that lacks
the optional extras, instead of skipping. These deps must be gated behind
``pytest.importorskip`` inside a fixture (see ``test_train_model.py`` /
``test_neural_predictor.py``). This test fails if the convention is broken so
the collection-time regression cannot recur.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).parent

# Top-level packages that are optional extras, not part of the dev group.
FORBIDDEN_TOP_LEVEL = {"torch", "onnx", "onnxruntime", "spacy", "stanza"}


def _module_level_offenders(tree: ast.Module) -> list[str]:
    """Return forbidden imports found directly in the module body.

    Imports nested inside functions/fixtures are intentionally ignored — that
    is the sanctioned ``pytest.importorskip``-then-import pattern.
    """
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_TOP_LEVEL:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] in FORBIDDEN_TOP_LEVEL or module.startswith(
                "scripts.train_stress_model"
            ):
                offenders.append(module)
            elif module == "scripts" and any(
                alias.name == "train_stress_model" for alias in node.names
            ):
                offenders.append("scripts.train_stress_model")
    return offenders


def test_no_eager_optional_imports_in_tests() -> None:
    violations: dict[str, list[str]] = {}
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders = _module_level_offenders(tree)
        if offenders:
            violations[path.name] = offenders
    assert violations == {}
