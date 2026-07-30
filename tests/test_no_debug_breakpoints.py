import ast
import re
from pathlib import Path

import pytest
import torch

from models.ts2vec.fsnet_ import SamePadConv


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = (
    ROOT / "main.py",
    ROOT / "data",
    ROOT / "exp",
    ROOT / "models",
    ROOT / "utils",
    ROOT / "scripts",
)


def _production_files():
    for path in PRODUCTION_PATHS:
        if path.is_file():
            yield path
            continue
        yield from path.rglob("*.py")
        yield from path.rglob("*.sh")


def test_production_contains_no_interactive_debugger_or_bare_except() -> None:
    forbidden = (
        "pdb." + "set_trace",
        "break" + "point(",
    )
    failures = []
    for path in _production_files():
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                failures.append(f"{path.relative_to(ROOT)}: {token}")
        if re.search(r"(?m)^\s*except\s*:", source):
            failures.append(f"{path.relative_to(ROOT)}: bare except")
        if re.search(r"(?m)^\s*except\s+Exception\s*:", source):
            failures.append(
                f"{path.relative_to(ROOT)}: broad Exception catch"
            )
        if path.suffix == ".py":
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names]
                    if "pdb" in names:
                        failures.append(
                            f"{path.relative_to(ROOT)}: imports pdb"
                        )
    assert failures == []


def test_fsnet_invalid_shape_raises_contextual_error() -> None:
    layer = SamePadConv(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        device=torch.device("cpu"),
    )
    invalid = torch.zeros(1, 2, 5)

    with pytest.raises((RuntimeError, ValueError)) as caught:
        layer(invalid)

    message = str(caught.value)
    assert "SamePadConv" in message
    assert "input_shape=(1, 2, 5)" in message

