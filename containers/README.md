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

20–40 minutes, ~6 GB, mostly CmdStan. If `--fakeroot` is refused, build
somewhere you have root and copy the `.sif` across.

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
