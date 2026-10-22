# Rockwell Automation + NVIDIA Innovation Lab · Fall 2026

This is your team's repository for the four-week physical AI hackathon run by Rockwell Automation, NVIDIA, the MSOE AI Club and UWM's Connected Systems Institute. Build everything here. Sponsors can follow your work as it progresses, and your final submission links straight to this repository.

**Full event guide** (datasets, compute, rules, judging, prizes): <https://msoe-maic.com/library/?article=Rockwell-NVIDIA-Fall-2026>
**Your team workspace** (team, repository access, submissions): <https://dashboard.all-ai-network.org/j/L36HEH>
**Questions:** [join the hackathon Slack](https://join.slack.com/t/rockwellnvidi-wq82211/shared_invite/zt-4bp05nsn9-UW3afFtM9wjzWtZiQoJBHg) and ask in the channel that matches your question. The event guide's "Getting help" section says who to ping for what.

## Key dates

| When | What |
|---|---|
| Thu Oct 8, 4:00–6:00 PM | Kickoff at Rockwell Automation HQ, 1201 S 2nd St, Milwaukee |
| **Wed Oct 14, 11:59 PM CT** | **Team plan due:** [submit it here](https://dashboard.all-ai-network.org/s/V28NHK) |
| **Sun Nov 1, 11:59 PM CT** | **Final submission due:** [submit it here](https://dashboard.all-ai-network.org/s/V28NHK) |
| Thu Nov 5, 5:00–8:00 PM | Finals at MSOE Diercks Hall: live presentations and prizes |

## First week

- [ ] Everyone on the team has an ALL AI Network account and has joined this team in the [workspace](https://dashboard.all-ai-network.org/j/L36HEH).
- [ ] Everyone has connected GitHub and accepted the repository invitation (check your GitHub email, or open the workspace and press the check button).
- [ ] Pick your dataset. We recommend one strong solution around a single dataset.
- [ ] Download the data and make sure everyone who needs it can open it.
- [ ] Read the compute guide and decide what you'll need.
- [ ] Join Slack, open `#hackathon-announcements`, and find your dataset's channel.
- [ ] Submit the [team plan](https://dashboard.all-ai-network.org/s/V28NHK) by Wednesday, October 14. It's a short resource request, not a graded paper, and any teammate can update it before the deadline.

## The three datasets

1. **[Final Vial Inspection](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Vial-Inspection):** find missing caps, crooked caps, damaged vials and unexpected empty positions after the robot fills a fixture.
2. **[Fluid Mixing Station 3](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Fluid-Mixing):** use synchronized camera views to detect fill and color problems caused by recipes and fluid carryover.
3. **[Grasping and Sorting](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Grasping-Sorting):** use a digital twin and multiple cameras to make a standalone FANUC arm grasp and sort objects more reliably.

## Read before your team plan

- **[Compute: Brev, DGX Spark and build.nvidia.com](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Compute).** Brev credits are spent whenever an instance is running, so stop instances you aren't using.
- **[Collect and test your own data](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Data-Collection):** visits, extra data, digital twins, and booking a supervised test on the real equipment.
- **[Tools and resources](https://msoe-maic.com/library?article=Rockwell-NVIDIA-Fall-2026-Resources):** AI coding tools, vision-language models, NVIDIA Cosmos and more. Start with a simple baseline.

## Final submission (Sun Nov 1)

- A video of about five minutes
- A one-page project summary
- A link to this repository
- References and credits for data, models, code and tools

## Working in this repository

```bash
git clone <this repository's URL from the green "Code" button>
```

- Commit early and push often. Mentors and judges see what's here.
- **Don't commit large datasets or model weights.** Keep them out with `.gitignore` and note in this README where to download them.
- Before the finals, replace this README with your project: the problem, your approach, results, how to run it, and credits.

## Development environment

The repository uses `uv` for Python dependencies and Singularity for the general project environment on ROSIE. Isaac Sim and Cosmos3-Nano are stored under the ignored `models/` directory.

```text
configs/                 Shared and problem-specific YAML configuration
data/problem1/           Final vial inspection data (large assets stay untracked)
data/problem2/           Fluid mixing data
data/problem3/           Grasping and sorting data
docker/                  Project SIF and shared Isaac Sim launch scripts
notebooks/               Exploratory analysis
scripts/                 Validation and automation scripts
artifacts/               Generated datasets and experiment artifacts (ignored)
models/                  Isaac Sim and Cosmos checkpoints (ignored)
outputs/                 Predictions, renders, and reports (ignored)
logs/                    Training and runtime logs (ignored)
```

Run the one-time setup from the repository root:

```bash
./setup_linux.sh
```

On Windows PowerShell:

```powershell
.\setup_windows.ps1
```

The Linux setup script creates the required environments, installs Isaac Sim under `models/isaacsim_6.1`, prepares the Cosmos3 environment and model path, and runs validation checks. Hugging Face authentication is required before Cosmos3-Nano weights can be downloaded. On ROSIE, the general project SIF remains optional and is built separately with `docker/setup_singularity.sh`.

Useful commands after setup:

```bash
uv run python main.py
uv run jupyter lab
./docker/problem1/run.sh shell
```

On ROSIE, use `docker/problem1/`, `docker/problem2/`, or `docker/problem3/` for problem-specific project runs. Problem 1 and 2 render the CSI twin with the shared Isaac Sim installation; Problem 3 runs the in-repository FANUC ER-4iA simulation from `fanuc_er4ia_sim/`.

## Lightweight cube detector

The sorting workflow can train YOLO26n using simulator-projected ground truth
or reviewed teacher labels. The default Problem 3 model schema covers
`yellow_cube`, `blue_cube`, `red_cube`, `cap`, and `vial`. Projected cap/vial
boxes are geometric extents rather than occlusion-aware masks, so review them
alongside all other labels.

Collect separate capture groups to measure transfer to a held-out camera and
lighting domain. Multi-view captures randomize cube placement/orientation and
studio-light intensity; `--camera` selects the viewpoint.

```bash
ISAAC=/path/to/isaac-sim/python.sh
"$ISAAC" fanuc_er4ia_sim/capture_scene.py \
  --out output/runs/table_train --camera table --views 24 --seed 42 \
  --group-name table_train --lighting-scale-range 0.65 0.95
"$ISAAC" fanuc_er4ia_sim/capture_scene.py \
  --out output/runs/side_holdout --camera side --views 12 --seed 1000 \
  --group-name unseen_side --lighting-scale-range 1.1 1.4
```

Build and inspect a group-held-out dataset before training:

```bash
uv run python scripts/run_vision_pipeline.py \
  --runs output/runs/table_train/table_train/view_* \
         output/runs/side_holdout/unseen_side/view_* \
  --validation-group unseen_side \
  --skip-training
```

Review `data/problem3/dataset/previews/` and `manifest.json`. The manifest
records class counts, teachers, groups, and any reusable instance masks. Run
the pipeline without `--skip-training` for an end-to-end train, or fine-tune
the reviewed dataset explicitly:

```bash
uv run --extra yolo python scripts/train_yolo_student.py \
  --data data/problem3/dataset/data.yaml \
  --device 0 \
  --imgsz 1280 \
  --epochs 100 \
  --batch 4
```

For non-simulator images, the exporter supports draft teacher boxes and SAM
3.1 labels, with configurable classes. To save masks from SAM, pass
`--class-names yellow_cube blue_cube red_cube cap vial`; customize prompts with
repeated `--class-prompt CLASS=TEXT` flags. Binary instance masks are saved as
PNG files and copied into `data/problem3/dataset/masks/`, indexed by
`mask_index.json`. See the dedicated [SAM 3.1 setup and labeling
instructions](fanuc_er4ia_sim/SETUP.md#4-sam-31-instance-masks).

The larger training image size preserves detail for small cubes; tune it and
the batch size for the target GPU. A held-out camera/light group is a stronger
generalization check than a random image split, but this simulation still does
not substitute for real-camera evaluation. The detector predicts 2D classes
and boxes; it does not replace calibrated geometry for world-frame grasping.

To render with the standalone Isaac Sim 6.1.0 install on ROSIE, run this once
from the repository root. The archive is large and the extraction can take
several minutes on the shared filesystem:

```bash
./scripts/install_isaac_sim_standalone.sh
test -x models/isaacsim_6.1/kit/python/bin/python3
```

Then submit the render as a Slurm batch job from a management node:

```bash
./docker/problem2/render_twin.sh
ls -lh outputs/problem2_fillcam.png
```

The launcher requests a teaching-node T4 allocation with `sbatch`, waits for
completion, and writes the job logs under `logs/`:

```bash
./docker/problem2/render_twin.sh
ls -lh outputs/problem2_fillcam.png
```

If `python.sh` reports that `kit/python/bin/python3` is missing, the install
was interrupted or incomplete. Wait for the installer to finish, then rerun
the installer; it resumes the download and safely completes extraction.

**Can't push or see this repository?** Open the [team workspace](https://dashboard.all-ai-network.org/j/L36HEH) and press the GitHub check button. If it still doesn't work, ping Brett Storoe in `#hackathon-general` on Slack.
