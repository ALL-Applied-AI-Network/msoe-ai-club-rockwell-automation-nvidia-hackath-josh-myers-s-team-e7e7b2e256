# Setup checklist for people and coding agents

Run commands from the extracted `fanuc_er4ia_sim/` directory. Replace all
`/path/to/...` placeholders with actual machine paths. The starter has no media
or saved run to replay. It was validated on Linux with Isaac Sim 6.0 and an RTX
A6000; Windows and smaller GPUs have not been benchmarked.

## 1. Host setup

Prerequisites: Python with `venv` and pip. Host Python 3.12 was used for validation.

```bash
./bootstrap.sh
```

This creates/reuses `.venv`, installs `requirements.txt`, generates procedural
metal/environment maps in `textures/`, and runs 33 tests. No reference images
are needed. Detailed finger meshes use silver materials without photo maps.
These generated maps are local outputs, not bundled image files.

## 2. Isaac Sim

Install Isaac Sim separately, or reuse an existing installation:

```bash
export ISAAC=/path/to/isaacsim
mkdir -p ../output/runs
"$ISAAC/python.sh" build_scene.py --headless --save ../output/runs/scene.usd
```

The scene loads the FANUC ER-4iA robot USD from Isaac's asset root. Network access
or an available asset cache is needed on first use. Robot USD and its referenced
files are not bundled. An optional local asset/URI override is supported:

```bash
export FANUC_ROBOT_USD=/path/to/er4ia.usd
```

Preserve the asset's dependencies, original joint names and hierarchy.
To open the scene interactively, run `"$ISAAC/python.sh" build_scene.py`.

## 3. Cosmos3-Nano

Use a separate `uv` environment with CUDA-enabled PyTorch, compatible
Transformers, and NVIDIA's `transformers-cosmos3` package/framework. Obtain
Cosmos3-Nano model weights separately; these large dependencies are not in the
starter.

```bash
export COSMOS=/path/to/cosmos-environment/bin/python
uv venv "$(dirname "$(dirname "$COSMOS")")"
uv pip install --python "$COSMOS" \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu129
uv pip install --python "$COSMOS" \
  -r requirements-cosmos.txt 'transformers>=4.57,<5' accelerate safetensors
export COSMOS3_NANO_PATH=/path/to/Cosmos3-Nano/snapshot
```

To download the gated repository into a shared project location, first accept
NVIDIA's model license on Hugging Face and authenticate with a read token:

```bash
uv pip install --python "$COSMOS" -U huggingface_hub
"$COSMOS" -c 'from huggingface_hub import login; login()'
"$COSMOS" -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="nvidia/Cosmos3-Nano", local_dir="../models/Cosmos3-Nano")'
export COSMOS3_NANO_PATH="$PWD/../models/Cosmos3-Nano"
```

Alternatively, use `hf auth login` and `hf download` if the standalone `hf`
command is installed. Do not put the token itself in the repository or `.env`.

The snapshot must contain `config.json` and the model/processor files.
If the framework package is not installed, set its checkout location:

```bash
export COSMOS3_ROOT=/path/to/cosmos-framework
```

That directory must contain `packages/transformers-cosmos3/transformers_cosmos3/`.
With no explicit snapshot, model discovery uses the Hugging Face cache;
`HF_HOME` or `HF_HUB_CACHE` can select its location.

Check dependencies and model loading before a complete run:

```bash
"$COSMOS" -c 'import torch, transformers, scipy, cv2; print("CUDA:", torch.cuda.is_available())'
"$COSMOS" -c 'from perceive import load_model; model, processor = load_model(); print("Model ready")'
```

CUDA must be available. The second command also checks framework imports and
weights, and exits after loading to release GPU memory.

## 4. SAM 3.1 cube labels

SAM 3.1 is an optional alternate teacher for the YOLO dataset workflow. It
segments instances from text prompts ("yellow cube", "blue cube", and "red
cube") and turns each mask into a normalized bounding box. Its outputs are
pseudo-labels, not ground truth: inspect each generated `sam3_preview.png`
before exporting labels. This does not change the Cosmos runtime used by the
robot sorting pipeline.

SAM 3.1 requires the official Meta SAM 3 code, Python 3.12+, PyTorch 2.7+ and a
CUDA 12.6+ compatible GPU. The official setup currently recommends PyTorch
2.10.0 with CUDA 12.8. Use a separate environment from Isaac Sim and Cosmos.
The teaching T4 failed during SAM grounding; a V100 completed the 30-image
inference run. The checkpoint is several gigabytes.

```bash
UV_PYTHON_INSTALL_DIR="$PWD/../models/.python" uv python install 3.12.11
SAM3_BASE_PYTHON="$PWD/../models/.python/cpython-3.12.11-linux-x86_64-gnu/bin/python"
"$SAM3_BASE_PYTHON" -m venv ../models/.venv-sam3-hpc
SAM3_PYTHON="$PWD/../models/.venv-sam3-hpc/bin/python"
git clone https://github.com/facebookresearch/sam3.git ../models/sam3
uv pip install --python "$SAM3_PYTHON" torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$SAM3_PYTHON" -e ../models/sam3 \
  numpy==1.26.4 scipy==1.15.3 opencv-python-headless==4.11.0.86 \
  pycocotools==2.0.8 einops psutil
```

The weights are gated. Request and accept access to the
[facebook/sam3.1 Hugging Face repository](https://huggingface.co/facebook/sam3.1),
then authenticate in the SAM environment and download its checkpoint:

```bash
uv pip install --python "$SAM3_PYTHON" huggingface_hub
"$PWD/../models/.venv-sam3-hpc/bin/hf" auth login
"$PWD/../models/.venv-sam3-hpc/bin/hf" download facebook/sam3.1 \
  sam3.1_multiplex.pt --local-dir ../models/SAM3-1
```

Do not put an access token in the repository.

Run SAM once over all captured views; it loads the model once and writes
`sam3_detections.json` plus a colored-mask review image for each view:

```bash
srun --partition=dgx --gpus=1 --ntasks=1 --cpus-per-task=8 \
  --mem=64G --time=02:00:00 \
  "$SAM3_PYTHON" ../scripts/label_with_sam3.py \
  --checkpoint ../models/SAM3-1/sam3.1_multiplex.pt \
  --runs ../output/runs/cube_views_gt_20261020_1435/train/view_* \
         ../output/runs/cube_views_gt_20261020_1435/val/view_* \
  --device cuda \
  --threshold 0.5
```

The loader currently reports four missing `convs.3` neck parameters for this
checkpoint. Treat SAM results as pseudo-labels and compare them with the
simulator annotations before using them for training.

After reviewing the per-view previews, feed the labels to the existing exporter:

```bash
uv run python ../scripts/build_yolo_dataset.py \
  --runs ../output/runs/cube_views_gt_20261020_1435/train/view_* \
         ../output/runs/cube_views_gt_20261020_1435/val/view_* \
  --detections-name sam3_detections.json \
  --output ../artifacts/cube_yolo_sam3 \
  --validation-group val
```

SAM output files are separate from Cosmos `detections.json` and simulator
ground-truth labels. The script refuses to overwrite existing outputs unless
`--overwrite` is passed.

## 5. Run and verify

```bash
./run_sort_pipeline.sh ../output/runs/stack_01
```

Use an empty run directory. Without an argument, the launcher creates a
timestamped directory. `HOST_PYTHON` optionally overrides `.venv/bin/python`.

| Stage | Interpreter | Entry point |
| --- | --- | --- |
| Calibrated 4K table capture | Isaac | `capture_scene.py` |
| Cube semantics and measured pose reporting | Cosmos | `perceive.py --stack-cubes` |
| Plan and evaluation | Host | `plan_sort.py` |
| Physical grasps and wrist control | Isaac | `sort_scene.py` |
| Locally generated review videos | Host | `encode_video.py` |

The run produces its own images, videos and JSON reports. Read
`execution_report.json` and require `task_complete: true`, six verified pickups
and six verified placements. Inspect `sort_plan.json` for skips and
`perception_evaluation.json` for position/colour/yaw errors. Raw model replies
and image-measured pose inputs are saved under `cube_pose_crops/`.
Square yaw errors use 90° symmetry. Ground truth evaluates results only.

To create a set of varied, repeatable images without restarting Isaac for each
one, capture multiple cube layouts into one directory:

```bash
"$ISAAC/python.sh" capture_scene.py \
  --out ../output/runs/cube_views_01 \
  --views 30 \
    --validation-views 6 \
    --seed 42
  ```

  This writes 24 training candidates under `train/` and six held-out views under
  `val/`. Each view includes `world_camera.png`, `camera.json`, and
  `ground_truth.json`; the seed changes the four additional cube positions and
  all six cube rotations while keeping cubes spaced from one another and the
  fixed props. Use the same seed and view count to reproduce the set. The arm
  and camera stay fixed; these captures vary object arrangement and orientation,
  not lighting or camera angle.

  For Cosmos pseudo-labels, run perception on each view directory; it writes the
  `detections.json` expected by the YOLO dataset exporter:

  ```bash
  "$COSMOS" perceive.py \
    --runs ../output/runs/cube_views_01/train/view_* \
           ../output/runs/cube_views_01/val/view_* \
    --dataset-labels
  ```

  Cosmos loads once and processes all 30 views sequentially, without the extra
  robot-pose auditing passes (which are not needed for bounding-box labels).
  Export the reviewed pseudo-labels in YOLO format, explicitly holding out the
  separate capture group:

  ```bash
  uv run python ../scripts/build_yolo_dataset.py \
    --runs ../output/runs/cube_views_01/train/view_* \
           ../output/runs/cube_views_01/val/view_* \
    --output ../artifacts/cube_yolo_v1 \
    --validation-group val
  ```

  Since the camera, lighting and props remain fixed, the held-out views estimate
  generalization to new object layouts in this scene only; they are not a
  substitute for captures under meaningfully different conditions.

To rerun physical validation without rendering, use a fresh captured/planned run:

```bash
"$ISAAC/python.sh" sort_scene.py --run ../output/runs/new_planned_run --physics-only
```

## Troubleshooting

- Missing launcher: check the interpreter paths and executable permissions.
- Missing model/framework: verify the snapshot and package/checkout paths.
- Missing robot: check Isaac's asset root or `FANUC_ROBOT_USD` and dependencies.
- No eligible picks: inspect skips and raw model replies; all six poses must pass.
- Incomplete stack: inspect the execution report and simulator logs.
- Changed geometry/layout/camera: recapture and replan in an empty directory.

Agents should report missing dependencies and completed checks precisely;
host tests do not prove model or simulator execution. Read `AGENTS.md` for
editing constraints and success criteria.
