# Singularity workflows

## Build and run with Singularity

The HPC workflow uses Singularity's Docker transport. It pulls the CUDA image from Docker Hub and converts it to a SIF locally.

```bash
./docker/setup_singularity.sh
```

This checks the cluster runtime, converts the CUDA base image in place, and validates `innovation-lab.sif`. Set `FORCE=1` to replace an existing SIF. Set `IMAGE_URI` to use another compatible CUDA image. Singularity's Docker bootstrap is the only Docker dependency.

On ROSIE, load the site's Singularity module if needed and use an interactive teaching-node shell or command:

```bash
./docker/run_singularity.sh shell
./docker/run_singularity.sh uv run python main.py
```

The runner binds the repository to `/workspace`, binds ROSIE's `/data` when it exists, and passes through NVIDIA GPUs with `--nv` by default. Use `USE_GPU=0` for CPU-only work. For non-interactive work, submit the optional Slurm wrapper:

```bash
sbatch docker/run_singularity.sbatch
sbatch docker/run_singularity.sbatch uv run your_script.py
```

The GPU, memory, and CPU requests in the wrapper are examples; adjust them for the ROSIE allocation.

The SIF contains the shared Python/uv environment and CUDA libraries. It binds the repository at `/workspace` and ROSIE's `/data` when available.

## Problem-specific launchers

Each launcher selects its corresponding development config and exposes it inside the container as `INNOVATION_LAB_CONFIG`.

```bash
./docker/problem1/run.sh shell
./docker/problem2/run.sh shell
./docker/problem3/run.sh shell
```

To run a command for a problem:

```bash
./docker/problem1/run.sh uv run python main.py
./docker/problem2/run.sh uv run jupyter lab --ip=0.0.0.0
./docker/problem3/run.sh uv run python main.py
```

## Load and render the CSI digital twins

`data/csi_digital_twin_v1` contains the tray-inspection and fill-station USD twins. Isaac Sim is not included in the general project SIF, so provide either an Isaac Sim `python.sh` or an Isaac Sim SIF.

With an Isaac Sim installation:

```bash
ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh \
  ./docker/problem1/render_twin.sh

ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh \
  ./docker/problem2/render_twin.sh

TWIN_CAMERA=CapSealCam TWIN_FRAME=1 \
ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh \
  ./docker/problem2/render_twin.sh
```

With an Isaac Sim container:

```bash
ISAAC_SIM_SIF=/path/to/isaac-sim.sif ./docker/problem1/render_twin.sh
ISAAC_SIM_SIF=/path/to/isaac-sim.sif ./docker/problem2/render_twin.sh
```

The container launcher invokes `/isaac-sim/python.sh` through `/bin/bash` to
support SIFs whose bundled launcher is not directly executable. Override either
path when using a custom image:

```bash
ISAAC_SIM_PYTHON_IN_CONTAINER=/opt/isaac-sim/python.sh \
ISAAC_SIM_SHELL_IN_CONTAINER=/bin/bash \
  ISAAC_SIM_SIF=/path/to/isaac-sim.sif ./docker/problem1/render_twin.sh
```

## Build an Isaac Sim SIF

The Isaac Sim image is separate from the project SIF. NVIDIA publishes the
container at `nvcr.io/nvidia/isaac-sim:6.0.0`; pulling it requires NGC access.

```bash
export NGC_API_KEY=your_ngc_key
./docker/isaac_sim/build_sif.sh
```

The helper submits `build_sif.sbatch` to Slurm, keeping the large image
extraction off the login node. The build uses `docker/isaac_sim/Singularity.def`
to normalize `/isaac-sim` permissions for Rosie’s non-root runtime user. Build output is written to
`logs/isaac-sim-build-<jobid>.out` and `.err`. The job places Singularity's
temporary unpack directory and OCI layer cache under
`docker/.cache/isaac-sim-*` rather than `/tmp`. It uses Singularity's
`--fakeroot --fix-perms` options so the non-root Rosie build can run the
permission-normalization step and make `/isaac-sim` readable to the runtime
user. Rosie must have an `/etc/subuid` mapping for this mode. If it does not,
ask the administrator to add one or use `ISAAC_SIM_BUILD_MODE=remote` with a
configured Singularity remote builder. The remote builder requires a separate
Singularity token; `NGC_API_KEY` only authenticates the NVIDIA registry:

```bash
singularity remote login
FORCE=1 ./docker/isaac_sim/build_sif.sh
```

The build defaults to `ISAAC_SIM_BUILD_MODE=remote` when no mode is supplied.
Set `ISAAC_SIM_BUILD_MODE=fakeroot` explicitly on a Rosie installation with
subuid/subgid mappings. `ISAAC_SIM_BUILD_MODE=auto` remains available when
you want the script to select fakeroot only if the mapping exists.

`ISAAC_SIM_BUILD_MODE=direct` is only a diagnostic fallback because it cannot
apply the permission fix. Rebuild an existing image with `FORCE=1`.

To submit the job directly:

```bash
sbatch docker/isaac_sim/build_sif.sbatch
```
```

## Rosie teaching-node submission

The Isaac Sim launcher targets Rosie teaching nodes, which provide T4 GPUs. It
requests `gpu:t4:1` and launches the SIF through `srun`.

```bash
./docker/isaac_sim/submit.sh
```

`run.sh` requests one T4 GPU on Rosie’s `teaching` partition with `srun` when
called outside an allocation, then launches the container on that node.
`submit.sh` remains as a compatibility wrapper around `run.sh`.

For a container-only shell test inside the allocated T4 node:

```bash
ALLOW_UNSUPPORTED_GPU=1 \
  ./docker/isaac_sim/submit.sh shell
```

The fill animation uses frames 1–686 at 30 fps. Problem 1 uses `CapSealCam`
and applies its lens correction automatically. Problem 2 defaults to `FillCam`
at frame 686; set `TWIN_FRAME` or `TWIN_CAMERA` to change the render.

NVIDIA TAO Toolkit and Isaac Sim are distributed through NVIDIA-specific launchables and may require NGC credentials. Keep credentials in `.env`, never in Git.
