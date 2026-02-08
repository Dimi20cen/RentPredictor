#!/usr/bin/env bash
set -euo pipefail

echo "[lint] python compile check"
python -m compileall app.py src scripts tests

echo "[typecheck] lightweight import/type smoke"
python - <<'PY'
import importlib
for mod in [
    "app",
    "src.ml_pipeline",
    "scripts.train",
    "scripts.evaluate",
    "scripts.predict",
]:
    importlib.import_module(mod)
print("imports-ok")
PY

echo "[test] unittest"
python -m unittest discover -s tests -p "test_*.py" -v

echo "[docs] presence checks"
test -f README.md
test -f docs/changes.md
echo "docs-ok"

echo "[gate] complete"
