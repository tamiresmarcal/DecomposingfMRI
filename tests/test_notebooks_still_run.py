"""Every notebook's imports and path defaults, executed.

WHY THIS FILE EXISTS. The reorganisation moved every table and figure module
into the package, and the unit tests stayed green while three things were
broken in ways only running the code shows:

  * `cli.py` called `attach_confounds` without importing it -- caught by the
    fixture chain, not by any test.
  * `contextualization/correlations.py` and `figures/states.py` built their
    default paths from `HERE`, which went away with the sys.path bootstrap the
    move replaced. Both raised NameError at import of the argparse block.
  * `hads.Settings.cache_dir` defaulted to `Path(__file__).parent / "cache"`,
    which followed the module into the package -- so the slow tables would
    have been cached into site-packages.

Each of those is in the SETUP of a notebook: the imports, and the defaults
evaluated when a module loads. That is the cheapest half of "do the notebooks
still run" and it needs no data, so it belongs in the suite. The other half --
do they produce the right figures -- needs the real tree and is run by hand.

What this does NOT check: that a notebook's analysis cells work. A green run
here means the imports resolve and no default path points somewhere absurd.
"""
import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(REPO.joinpath("notebooks").glob("*.ipynb"))


def code_cells(path):
    nb = json.loads(path.read_text())
    return ["".join(c["source"]) for c in nb["cells"]
            if c["cell_type"] == "code"]


def imported_modules(src):
    """Every `fmri_decomposition...` module a cell imports, via the AST.

    The AST rather than a regex: `from a import b as c` and `import a.b.c`
    are different nodes and a regex over both is how one form gets missed.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:                     # a cell using IPython magics
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            out += [node.module]
            out += [f"{node.module}.{a.name}" for a in node.names]
    return [m for m in out if m.startswith("fmri_decomposition")]


def test_there_are_notebooks_to_check():
    """So a glob that silently matches nothing cannot pass this file."""
    assert len(NOTEBOOKS) >= 6, [p.name for p in NOTEBOOKS]


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_every_module_a_notebook_imports_is_importable(nb):
    """The whole class of break the reorganisation could cause.

    A notebook naming a module that moved, or a module whose own imports did
    not survive the move, fails here instead of in front of her.
    """
    import importlib

    wanted = {m for src in code_cells(nb) for m in imported_modules(src)}
    assert wanted, f"{nb.name} imports nothing from the package -- has its " \
                   f"setup cell been rewritten by hand?"
    for mod in sorted(wanted):
        try:
            importlib.import_module(mod)
        except ImportError:
            # `from x import name` yields "x.name", which is not a module
            # when `name` is a function. Importing the parent is the check.
            importlib.import_module(mod.rsplit(".", 1)[0])


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_no_notebook_still_inserts_the_old_sys_path(nb):
    """`sys.path.insert(... "notebooks")` was how these modules used to be
    importable. A leftover one hides a real ImportError behind a path that
    happens to exist on one machine."""
    for src in code_cells(nb):
        assert 'REPO / "notebooks"' not in src, nb.name
        assert 'str(REPO / "tools")' not in src, nb.name


class TestNoDefaultPathPointsIntoThePackage:
    """Three defaults were `Path(__file__).parent / <something>` and followed
    their module into the package when it moved. A package directory is
    read-only in an installed environment and is never where data goes."""

    def _not_in_package(self, p, what):
        pkg = REPO / "fmri_decomposition"
        assert pkg not in Path(p).resolve().parents, f"{what} -> {p}"

    def test_the_hads_cache(self):
        from fmri_decomposition.supervised_model import hads as ht
        s = object.__new__(ht.Study)
        s.cfg = ht.Settings(root=Path("/o"))
        self._not_in_package(s.cache_root, "hads cache")
        assert "version=v0" in str(s.cache_root), s.cache_root

    def test_the_reference_maps(self):
        from fmri_decomposition.io import reference_maps_dir
        self._not_in_package(reference_maps_dir(), "reference maps")

    def test_the_reference_maps_follow_their_env_var(self, monkeypatch):
        from fmri_decomposition.io import reference_maps_dir
        monkeypatch.setenv("FMRIDECOMP_REFMAPS", "/project/x/refmaps")
        assert reference_maps_dir() == Path("/project/x/refmaps")

    def test_the_contextualisation_output(self):
        from fmri_decomposition.io import tree_root
        p = tree_root("/o", "contextualization")
        self._not_in_package(p, "contextualization results")
        assert "version=v0" in str(p), p

    def test_both_contextualisation_entry_points_parse(self):
        """Their argparse defaults are evaluated at parse time, and `--help`
        is the cheapest way to make that happen. This is the exact failure:
        both raised NameError here and nothing noticed."""
        from fmri_decomposition.figures import states
        from fmri_decomposition.unsupervised_model.contextualization \
            import correlations
        for mod in (correlations, states):
            with pytest.raises(SystemExit) as e:      # --help exits 0
                mod.main(["--help"])
            assert e.value.code == 0, mod.__name__
