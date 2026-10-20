# ROSIE Docker and Isaac Sim workflows

This project has two separate runtime paths:

- `run_singularity.sh` runs the general project Python environment in the project SIF.
- `run_csi_twin.sh` runs the shared standalone Isaac Sim 6.1.0 installation directly on a GPU node.

The Isaac Sim path no longer builds, pulls, or runs an Isaac Sim container.

## General project environment

For a complete Linux setup for Problems 1, 2, and 3, run this from the
repository root. It logs each stage to `logs/setup-<timestamp>.log`, installs
the project and Cosmos environments, checks/downloads model assets, and
installs Isaac Sim under `models/`:

```bash
./setup_linux.sh
```

If Cosmos3-Nano is not already present, authenticate with Hugging Face and
rerun the setup or download it into `models/Cosmos3-Nano` as described in
`fanuc_er4ia_sim/SETUP.md`.

Build the project SIF once:

```bash
./docker/setup_singularity.sh
```

Run an interactive shell or a project command:

```bash
./docker/run_singularity.sh shell
./docker/run_singularity.sh uv run python main.py
```

Problem launchers select their config automatically:

```bash
./docker/problem1/run.sh shell
./docker/problem2/run.sh uv run python main.py
```

The general SIF is not used to provide Isaac Sim.

## Shared Isaac Sim installation

Install Isaac Sim 6.1.0 once from the repository root:

```bash
./scripts/install_isaac_sim_standalone.sh
test -x models/isaacsim_6.1/kit/python/bin/python3
```

From a ROSIE management node, submit the render as a Slurm batch job:

```bash
./docker/problem2/render_twin.sh
```

The `.sh` launcher submits itself with `sbatch` when run outside an allocation,
requests `gpu:t4:1`, waits for completion, and writes logs under `logs/`. From
inside an existing allocation it runs directly. The output is written to
`outputs/problem2_fillcam.png`.

To submit explicitly and preserve all logs:

```bash
sbatch --wait --partition=teaching --gres=gpu:t4:1 \
  --cpus-per-gpu=8 --ntasks=1 --chdir="$PWD" \
  --output=logs/problem2-%j.out --error=logs/problem2-%j.err \
  docker/problem2/render_twin.sh
```

Problem 1 renders the tray-inspection camera:

```bash
./docker/problem1/render_twin.sh
```

Problem 2 defaults to `FillCam` at frame 686. Override the camera, frame,
subframes, or output path with environment variables:

```bash
TWIN_CAMERA=FillCam TWIN_FRAME=301 TWIN_SUBFRAMES=16 \
  ./docker/problem2/render_twin.sh
```

Problem 3 runs the FANUC ER-4iA cube-stacking simulation from the in-repository
`fanuc_er4ia_sim` starter. It runs capture, Cosmos perception, host
planning, physical execution, report validation, and video encoding in one
T4 allocation by default:

```bash
./docker/problem3/run.sh
```

The launcher automatically selects the available GPU count from the Slurm
allocation or teaching partition and writes each run to a timestamped directory
under `output/runs/`. Override the count or output directory when needed:

```bash
PROBLEM3_GPUS=2 ./docker/problem3/run.sh output/runs/stack_01
```

Cosmos3-Nano can shard across multiple T4s. The default `device_map=auto` then
spreads the model across every GPU Slurm exposes to the job. Set
`PROBLEM3_GPUS` only when you want to override automatic selection:

```bash
export COSMOS="$HOME/.venvs/cosmos3/bin/python"
export COSMOS3_NANO_PATH="$PWD/models/Cosmos3-Nano"
PROBLEM3_GPUS=2 ./docker/problem3/run.sh output/runs/stack_01
```

Problem 3 submits an `sbatch` job automatically when launched from a
management node. Its output is written to `logs/problem3-<jobid>.out` and
`.err`.

Use `PROBLEM3_GPUS=3` or `4` if those resources are available. Do not pass
`--gpu 0` to `perceive.py`, because that intentionally masks the other GPUs.
If you already opened a one-GPU interactive allocation, exit it and rerun the
command from the management node; `PROBLEM3_GPUS` cannot expand an allocation
that Slurm has already granted. For an explicit batch submission:

```bash
PROBLEM3_GPUS=2 sbatch --wait --partition=teaching --gres=gpu:t4:2 \
  --cpus-per-gpu=8 --ntasks=1 --chdir="$PWD" \
  --output=logs/problem3-%j.out --error=logs/problem3-%j.err \
  docker/problem3/run.sh output/runs/stack_01
```

When using a batch submission, do not set `CUDA_VISIBLE_DEVICES=0`. The loader
preserves Slurm's GPU visibility and uses Accelerate's automatic device map.
Set `COSMOS_DEVICE_MAP` only for debugging or an intentional custom mapping.

The starter is included at `fanuc_er4ia_sim/`. Override its location only when
using a separate checkout:

```bash
FANUC_SIM_ROOT=/path/to/other/fanuc_er4ia_sim \
HOST_PYTHON="$PWD/.venv/bin/python" \
COSMOS=/path/to/cosmos-environment/bin/python \
  ./docker/problem3/run.sh output/runs/stack_01
```

Run the starter's 33 host-side regression tests without a GPU or Cosmos:

```bash
./docker/problem3/run.sh --host-tests
```

The starter's `SETUP.md` documents the required Cosmos3-Nano model snapshot,
FANUC asset resolution, and success checks in `execution_report.json`.

Use another shared installation explicitly:

```bash
ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh \
  sbatch --wait ./docker/problem2/render_twin.sh
```

Inside an existing GPU allocation, the low-level launcher runs a custom Isaac
Sim Python command directly:

```bash
./docker/isaac_sim/run.sh \
  data/csi_digital_twin_v1/tools/render_example.py \
  --camera FillCam \
  --stage data/csi_digital_twin_v1/stations/fill_station/fill_station.usda \
  --frame 686 \
  --out outputs/custom_fill.png
```

Submit a custom Isaac Sim Python job:

```bash
sbatch --wait docker/isaac_sim/submit.sh \
  data/csi_digital_twin_v1/tools/render_example.py \
  --camera FillCam \
  --stage data/csi_digital_twin_v1/stations/fill_station/fill_station.usda \
  --frame 686 \
  --out outputs/custom_fill.png
```

If `kit/python/bin/python3` is missing, the standalone extraction is
incomplete. Rerun `scripts/install_isaac_sim_standalone.sh`; it resumes the
archive download and completes extraction.
