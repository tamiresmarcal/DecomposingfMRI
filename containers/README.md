# Containers

| image | stages | why separate |
|---|---|---|
| `fmri_decomp.sif` (exists) | 2, 3 — extract, dfc | stable, and every cohort re-run depends on it |
| `fmri_decomp_stage45.sif` (`stage45.def`) | 4, 5 — state sets, transitions, brms | adds Stan, hmmlearn, umap, decoding tools |

Building Stan into the stage 2/3 image would mean rebuilding the thing all the
existing outputs were produced with. Two images, one boundary.

## Build

Login node — compute nodes have no outbound network.

```bash
cd /project/6008063/tamires/DecomposingfMRI
module load apptainer
apptainer build --fakeroot \
    /project/6008063/tamires/singularity/fmri_decomp_stage45.sif \
    containers/stage45.def
```

20–40 minutes. If `--fakeroot` is refused, build somewhere you have root and
copy the `.sif` across.

**The finished `.sif` is about 1 GB**, not the ~6 GB this file used to claim. A
.sif is compressed squashfs, so the image is a few times smaller than the tree
inside it, and a build that looks "too small to contain CmdStan" is the normal
size. Checking the size is not a way to tell what an image holds — ask it:

```bash
apptainer exec "$FMRIDECOMP_SIF45" sh -c 'ls /opt/cmdstan; python3 -c "import lightgbm, hmmlearn, umap"'
```

**Build from the repository root.** `%files` copies `containers/stage45.def`
into the image and its source path resolves against the working directory, so
building from inside `containers/` fails at the copy.

### An image does not update itself

The definition is a file in git; the image is a build of it at one moment. Three
commits changed the definition after the image on `nibi` was built, and the
newest of them added `lightgbm` — so `select` died for a package the definition
had carried for two days.

    2026-09-10  definition created, no lightgbm
    2026-09-30  image built  <-- the .sif on disk
    2026-10-01  four build failures fixed
    2026-10-01  numba/numpy pin conflict fixed
    2026-10-02  lightgbm==4.5.0 added

Nothing connected the two, so nothing could notice. Now the image carries its
own provenance:

    /opt/stage45.def            the definition it was built from
    /opt/def_sha256             that file's sha256
    /opt/image_provenance.txt   build time, python version, full pip freeze

and every stage 4–5 sbatch compares `/opt/def_sha256` against the definition in
the checkout, printing a NOTE when they differ. A note and not a failure: most
changes to the definition do not affect a given run, and refusing to start would
be worse than saying so. Both files are written at the END of `%post`, so they
exist only in an image that built all the way through.

To see what an image actually has:

```bash
apptainer exec "$FMRIDECOMP_SIF45" cat /opt/image_provenance.txt | head -20
```

Pin the apptainer version. A bare `module load apptainer` opens the
interactive `mii` menu, and picking the newest entry swaps `StdEnv/2023` for
`StdEnv/2026`, reloading a dozen modules as a side effect.

### What the base image actually is

`rocker/r-ver:4.4.1` is **Ubuntu 22.04 jammy, Python 3.10.12** — confirmed from
a build log, not assumed. The `%post` section prints `/etc/os-release` and
`python3 --version` so the next person does not have to assume either.

### Things this definition is careful about

Each of these failed a build. Only the first is cheap to discover; the rest
fail after minutes of downloading:

* **No python minor version is named.** The build uses `python3`, prints the
  version and asserts `>= 3.10`. jammy's `python3` is 3.10, and `python3.11` is
  not in its default repos — so the original `python3.11` pin could not have
  worked here, and would break differently again if rocker moves to noble.
* **`libtiff-dev`, not `libtiff5-dev`.** Both exist on jammy; only the former
  exists on noble.
* **`numba` pins `numpy`, not the reverse.** `numba==0.60.0` requires
  `numpy<2.1`, which contradicted the pinned `numpy==2.1.3` and failed the
  build with `ResolutionImpossible` after the whole apt stage. `numba==0.61.2`
  allows `numpy<2.3`; `scipy==1.14.1` caps it the same way, so 2.1.3 sits
  inside both. **Before changing any of numpy, scipy or numba, resolve the set
  first** — no container build needed:

  ```bash
  pip install --dry-run --ignore-installed --python-version 3.10       --only-binary=:all: --target /tmp/pipcheck -r <the pin list>
  ```

  That reproduces the build's resolution in seconds and reproduced this exact
  failure.
* **The pins are a file in the image** (`/opt/requirements.txt`), not a command
  line, so the set can be audited later — and so the optional install can be
  constrained with `-c` against it. "Installing neuromaps/nimare last means
  they cannot move numpy" was a hope about pip's upgrade behaviour; `-c` makes
  it enforced.
* **The R install is a heredoc file, not `Rscript -e`.** A backslash-newline
  inside a single-quoted shell string is a literal backslash, not a
  continuation, so a multi-line `-e '...'` hands R a script with stray
  backslashes in it.
* **CmdStan goes in `/opt`, not `~/.cmdstan`.** Apptainer runs as the invoking
  user and `/root` is mode 700, so a CmdStan under `/root` is unreadable by
  everyone the image was built for. `CMDSTAN` is then resolved by command
  substitution — `export CMDSTAN=/opt/cmdstan/cmdstan-*` does not glob.

## Use

```bash
export FMRIDECOMP_SIF45=/project/6008063/tamires/singularity/fmri_decomp_stage45.sif
apptainer exec --cleanenv --bind /project,/scratch,/home --pwd $PWD \
    $FMRIDECOMP_SIF45 python3 -m fmri_decomposition.cli states fit --help
```

`--cleanenv` is not optional. The host sets
`PYTHONPATH=/cvmfs/soft.computecanada.ca/custom/python/site-packages`, which
without it precedes `/opt/venv` on `sys.path` and can shadow a pinned package
with a host build for a different Python.

## What is pinned, and why

Every Python version is pinned. The ad-hoc `pip install umap-learn` into
`/tmp/pylibs` produced a numpy 1.x / 2.x ABI clash that printed a traceback on
every import; an unpinned image reproduces that at random six months from now.
`numpy==2.1.3` is the anchor — `numba`, `hmmlearn` and `umap-learn` are the
packages that constrain it.

`neuromaps` and `nimare` are installed last and allowed to fail without
failing the build: they pull heavy, fast-moving dependency trees, and the rest
of stage 4/5 does not depend on them. If the build log shows them missing, the
decoding panel needs them installed separately.

## Verifying

`apptainer test` runs the `%test` block, which imports every Python module and
checks the four R packages:

```bash
apptainer test /project/6008063/tamires/singularity/fmri_decomp_stage45.sif
```

Do this before submitting anything. A missing `hmmlearn` discovered inside an
array job costs an allocation.

It **exits non-zero** when a required module, an R package or CmdStan is
missing — a gate that only prints is not a gate. `neuromaps` and `nimare` are
reported and never fatal.

## shap, and the two stage-4/5 definitions

`containers/stage45.def` has no `shap`. Ridge does not need it — its SHAP is
exact arithmetic, coefficient × standardised value, computed with numpy — so
every ridge |SHAP| panel draws in the image we run today. A boosted tree's
needs `shap.TreeExplainer`.

`containers/stage45_shap.def` is that file plus one pinned line, and builds to
its own `.sif` so the current one keeps working:

```bash
cd /project/6008063/tamires/DecomposingfMRI
module load apptainer/1.3.5
apptainer build --fakeroot \
    /project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif \
    containers/stage45_shap.def
apptainer test /project/6008063/tamires/singularity/fmri_decomp_stage45_shap.sif
```

Then point `FMRIDECOMP_SIF45` at it in `slurm/env.sh`. When nothing uses the
old `.sif`, delete it **and** `containers/stage45.def` — a definition no image
was built from is one nobody will keep true. The sbatch scripts compare the
image's `def_sha256` against **every** `containers/*.def`, so both are quiet
while both exist.

### Installing a package into this image from the notebook

There IS a `pip install` that works from a notebook cell. Two that do not,
first, because both look like they should:

```
pip install shap          ->  OSError: Read-only file system: '/opt/venv/...'
pip install --user shap   ->  ERROR: Can not perform a '--user' install.
                              User site-packages are not visible in this
                              virtualenv.
```

The first is the `.sif`: `/opt/venv` is inside it and a SIF is read-only. It
appears to work under `apptainer --writable-tmpfs`, and the install is then
discarded when the container exits — which is the "sometimes it works".

The second is **not** `PYTHONNOUSERSITE`. `/opt/venv` is a *venv*, and a venv
has `site.ENABLE_USER_SITE = False` whatever that variable says — checked, not
assumed:

```
$ PYTHONNOUSERSITE=1 venv/bin/python -c "import site; print(site.ENABLE_USER_SITE)"
False
$ venv/bin/python -c "import site; print(site.ENABLE_USER_SITE)"
False            # same, with the variable unset
```

so `~/.local` is never on the path here and `--user` is refused either way.
Unsetting `PYTHONNOUSERSITE` changes nothing about this.

**What works, in one notebook cell:**

```python
PYLIBS = "/project/6008063/tamires/pylibs"
%pip install --no-cache-dir --no-deps --target=$PYLIBS \
    shap==0.46.0 slicer==0.0.8 cloudpickle==3.1.0

import sys
if PYLIBS not in sys.path:
    sys.path.append(PYLIBS)        # APPEND: see below
import shap
```

`--target` writes to a bind-mounted host directory, which is writable, and is
the one pip mode that neither needs `/opt/venv` nor `~/.local`. Nothing has to
be relaunched and the directory survives the kernel, so later sessions need
only the `sys.path.append` — which notebook 13's setup cell does for you if you
set `EXTRA_PYLIBS`.

Two details that are the whole safety of it:

* **`--no-deps`.** Without it pip puts its own numpy, scipy and scikit-learn in
  that directory. An ABI mismatch against the ones the container was built with
  is a segfault, not an ImportError. `shap` needs `slicer` and `cloudpickle`
  on top of what the image already has; those are the two named above.
* **`append`, not `insert(0, ...)`.** Appended, the directory is searched
  *after* the container's own packages, so nothing in it can shadow them. If
  something really is missing you get a named ImportError, which is the failure
  you want.
