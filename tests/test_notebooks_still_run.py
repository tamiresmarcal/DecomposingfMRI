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
def test_the_setup_cell_runs_the_way_a_kernel_runs_it(nb):
    """From `notebooks/`, with the package NOT already importable.

    THIS IS THE TEST THAT WAS MISSING. A Jupyter kernel starts with
    sys.path[0] = the notebook's own directory, so `import fmri_decomposition`
    fails from notebooks/ unless the cell puts the repo root on the path --
    and the package is not pip-installed inside the analysis image. That line
    was removed along with the `notebooks/` entry beside it, which was the
    actual hack, and every notebook broke at cell 1 while
    `test_every_module_a_notebook_imports_is_importable` stayed green: that
    one runs from the repo root, where the import works for free.

    So this runs the real cell, in a subprocess, with cwd=notebooks/ and the
    repo root removed from the inherited path -- the kernel's conditions.
    """
    import subprocess
    import sys
    import textwrap

    src = code_cells(nb)[0]
    # Everything up to the first non-import, non-assignment line: the setup,
    # not the analysis, which would need the real output tree.
    keep, lines = [], src.splitlines()
    for ln in lines:
        if ln.startswith(("import ", "from ", "#", " ", ")", "]", "}")) or \
                not ln.strip() or "=" in ln.split("#")[0]:
            keep.append(ln)
        else:
            break
    setup = "\n".join(keep)
    assert "import" in setup, f"{nb.name}: no imports found in cell 1"

    out = subprocess.run(
        [sys.executable, "-c", setup],
        cwd=str(REPO / "notebooks"), capture_output=True, text=True,
        # PYTHONPATH emptied AND -c run from notebooks/, so the only way the
        # package is found is the cell's own bootstrap.
        env={**{k: v for k, v in __import__("os").environ.items()
                if k != "PYTHONPATH"},
             "MPLBACKEND": "Agg"})
    assert "No module named 'fmri_decomposition'" not in out.stderr, (
        f"{nb.name} cannot import the package from notebooks/ -- its setup "
        f"cell needs the repo root on sys.path:\n{out.stderr[-600:]}")
    # Anything else that goes wrong here is about data or a missing optional
    # dependency, which this test is not about.
    if out.returncode != 0:
        bad = ("ModuleNotFoundError: No module named 'fmri_decomposition",
               "ImportError: cannot import name")
        for b in bad:
            assert b not in out.stderr, f"{nb.name}:\n{out.stderr[-600:]}"


@pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
def test_every_notebook_puts_the_repo_root_on_the_path(nb):
    """Explicitly, because the subprocess test above can only prove the
    import worked -- not that it worked for the right reason. An installed
    copy of the package in the test environment would hide a missing
    bootstrap."""
    first = code_cells(nb)[0]
    assert "sys.path.insert(0, str(REPO))" in first, nb.name


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


class TestTheReferenceMapsRefuseByName:
    """They are an input she may keep outside the repo -- shared between
    projects -- so the one thing that must not happen is a FileNotFoundError
    on some .nii.gz deep inside a loader, which says nothing about a 285 MB
    download or an environment variable."""

    def test_the_refusal_names_the_path_the_variable_and_the_fetcher(
            self, tmp_path, monkeypatch):
        from fmri_decomposition.io import require_reference_maps
        monkeypatch.setenv("FMRIDECOMP_REFMAPS", str(tmp_path / "nope"))
        with pytest.raises(SystemExit) as e:
            require_reference_maps()
        msg = str(e.value)
        assert str(tmp_path / "nope") in msg
        assert "FMRIDECOMP_REFMAPS" in msg
        assert "fetch_reference_maps.sh" in msg
        assert "slurm/env.sh" in msg

    def test_it_says_when_the_variable_is_not_even_set(self, monkeypatch):
        from fmri_decomposition.io import require_reference_maps
        monkeypatch.delenv("FMRIDECOMP_REFMAPS", raising=False)
        with pytest.raises(SystemExit) as e:
            require_reference_maps()
        assert "not set" in str(e.value)

    def test_an_empty_directory_is_not_a_directory_of_maps(self, tmp_path,
                                                           monkeypatch):
        from fmri_decomposition.io import require_reference_maps
        d = tmp_path / "reference_atlases"
        d.mkdir()
        monkeypatch.setenv("FMRIDECOMP_REFMAPS", str(d))
        with pytest.raises(SystemExit):
            require_reference_maps()
        (d / "MNI152.nii.gz").write_bytes(b"")
        assert require_reference_maps() == d


class TestPipingToHeadIsNotAnError:
    """`fmri-decomp status | head -20` is how a long report is read, and
    `head` closes the pipe as soon as it has its lines. The default is a
    six-frame traceback that looks like the command failed."""

    def test_the_cli_handles_a_closed_stdout(self):
        import subprocess
        import sys

        out = subprocess.run(
            f"{sys.executable} -m fmri_decomposition.cli --help | head -2",
            shell=True, capture_output=True, text=True, cwd=str(REPO))
        assert "BrokenPipeError" not in out.stderr, out.stderr
        assert "Exception ignored" not in out.stderr, out.stderr
        assert out.stderr.strip() == "", out.stderr

    def test_the_handler_is_in_main_not_in_one_command(self):
        """Every command can be piped, so it belongs at the top level. It was
        `status` that hit it first, which is not a reason to fix it there."""
        src = (REPO / "fmri_decomposition" / "cli.py").read_text()
        i = src.index("def main(")
        assert "except BrokenPipeError" in src[i:]


class TestNoNotebookBuildsATreePathByHand:
    """Five sites across three notebooks read the pre-split locations after
    the reorganisation, and found nothing without complaining.

    `root / "meta" / "cohorts"`, `ROOT / "transitions"`, `ROOT /
    f"{tree}_selection"` -- each one is an assertion about where a tree
    starts, written where nobody would look for it. The whole point of
    io.tree_root is that the shape is stated once.
    """

    STEMS = ("activation", "dfc", "latents", "transitions", "static_fc",
             "censor", "bstm_selection", "fcm_selection",
             "transferred_models", "contextualization")

    @pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
    def test_it_goes_through_io(self, nb):
        import re

        bad = []
        for i, src in enumerate(code_cells(nb)):
            for ln in src.splitlines():
                code = ln.split("#")[0]
                if "tree_root" in code or "meta_dir" in code:
                    continue
                for stem in self.STEMS:
                    # `<something> / "<stem>"` or `/ f"{x}_selection"`.
                    if re.search(rf'/\s*f?"{stem}"', code) or \
                            re.search(r'/\s*f"\{\w+\}_selection"', code):
                        bad.append(f"cell{i}: {ln.strip()}")
                        break
        assert not bad, (
            f"{nb.name} builds a tree path by hand; use io.tree_root or "
            f"io.meta_dir:\n  " + "\n  ".join(bad))

    @pytest.mark.parametrize("nb", NOTEBOOKS, ids=lambda p: p.stem)
    def test_nor_the_old_flat_meta(self, nb):
        for i, src in enumerate(code_cells(nb)):
            for ln in src.splitlines():
                code = ln.split("#")[0]
                if "meta_dir" in code:
                    continue
                assert '/ "meta"' not in code, f"{nb.name} cell{i}: {ln.strip()}"
