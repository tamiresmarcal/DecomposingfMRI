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
