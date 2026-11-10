"""Make a test's own directory importable, so sibling helpers still resolve.

`tests/` mirrors the package: extractions / unsupervised_model /
supervised_model, with the cross-cutting files at the top. Several tests import
a fixture builder from a sibling (`from test_transfer_wiring import
write_summary`) rather than duplicating it, which worked by accident while
every file sat in one flat directory and pytest's rootdir insertion covered it.

Adding each test directory to sys.path keeps that working and is also what lets
a helper live beside the tests that use it instead of in a shared module nobody
can find. It is `tests/conftest.py` rather than one per directory because the
rule is the same for all of them.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for d in [HERE, *sorted(p for p in HERE.iterdir() if p.is_dir()
                        and not p.name.startswith((".", "__")))]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
