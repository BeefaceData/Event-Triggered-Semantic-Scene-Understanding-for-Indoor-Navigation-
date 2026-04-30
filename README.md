# Event-Triggered Semantic Scene Understanding for Indoor Navigation

This repository contains the final project code, report, slides, experiment outputs, and demo artifacts for a geometry-first indoor navigation system with selective semantic reasoning.

The core idea is simple:

- use geometry as the default navigation signal
- measure when geometry becomes uncertain
- invoke semantic reasoning only when uncertainty or scene cues justify the extra cost

The project compares three policy families:

- `geometry_only`
- `always_semantic`
- `event_triggered`

and evaluates them across:

- Stage 1 frame-level HM3D analysis
- Stage 2 closed-loop embodied Habitat evaluation
- a recorded real-video demonstration

## Main Claim

The strongest supported claim of the project is not that semantics universally improves indoor navigation. The supported claim is narrower and more useful:

- uncertainty-triggered semantic reasoning can be integrated into a geometry-first navigation stack in a selective, interpretable, and experimentally grounded way
- `event_triggered + BLIP` is the strongest practical semantic policy in the current embodied setting
- `geometry_only` remains a very strong raw movement baseline

## Repository Layout

```text
Event-Triggered_Semantic_Scene_Understanding_for_Indoor_Navigation/
  README.md
  requirements.txt
  src/                  # Core pipeline, evaluation scripts, utilities
  app/                  # Assistive demo web app + FastAPI backend
  report/               # Final paper/report source
  slides/               # Beamer slide deck
  outputs/              # Evaluation results, demo videos, plots
  images/               # Shared project figures
  datasets/             # Local HM3D dataset location used by scripts
  recorded_video/       # Source recorded footage for the real-video demo
  notebooks/            # Notebook workspace
  docs/                 # Presentation planning and supporting notes
```

## Important Folders

### `src/`

Main implementation and experiment scripts.

Key files:

- `pipeline.py`: shared frame-level perception pipeline
- `geometry.py`: depth scoring, obstacle fusion, region scoring
- `trigger.py`: uncertainty and cue-based trigger logic
- `semantics.py`: BLIP / SmolVLM semantic querying logic
- `models.py`: model loading
- `evaluate_hm3d.py`: Stage 1 frame-level evaluation
- `threshold_calibration.py`: trigger-threshold sweep
- `backend_comparison.py`: BLIP / SmolVLM / Qwen backend screening
- `prompt_ablation.py`: prompt-stability comparison
- `run_closed_loop_hm3d.py`: single-scene embodied rollout
- `evaluate_closed_loop_hm3d.py`: Stage 2 closed-loop evaluation
- `run_recorded_video_demo.py`: recorded real-video demo pipeline
- `compare_recorded_demo_policies.py`: policy comparison on recorded video

### `report/`

Final report source.

Main entry files:

- `report/main_full.tex`: full draft version
- `report/main_ieee.tex`: IEEE-style version
- `report/main_cvpr.tex`: CVPR-style version

Main sections:

- `report/sections/introduction.tex`
- `report/sections/related_work.tex`
- `report/sections/method.tex`
- `report/sections/stage1_experimental_setup.tex`
- `report/sections/stage1_results.tex`
- `report/sections/stage2_method.tex`
- `report/sections/stage2_experimental_setup.tex`
- `report/sections/stage2_results.tex`
- `report/sections/recorded_real_video_demo_results.tex`
- `report/sections/discussion.tex`
- `report/sections/limitations.tex`
- `report/sections/conclusion.tex`

### `slides/`

Presentation deck source:

- `slides/main.tex`

### `app/`

Prototype assistive guidance app:

- browser camera frontend
- FastAPI backend
- speech and vibration-oriented guidance layer

See [app/README.md](/home/cmu/Event-Triggered_Semantic_Scene_Understanding_for_Indoor_Navigation/app/README.md:1) for app-specific usage.

### `outputs/`

Experiment outputs and demo artifacts.

Important subfolders:

- `outputs/val/calibration/`: threshold calibration artifacts
- `outputs/val/backend_comparison/`: Stage 1 backend screening
- `outputs/val/closed_loop/`: Stage 2 evaluation outputs
- `outputs/plots/`: summary plots used for the report and slides
- `outputs/demo/`: recorded real-video policy comparisons

## Core Method

The shared pipeline is geometry-first:

1. detect obstacles
2. estimate monocular depth
3. fuse obstacle and depth cues
4. score left / center / right regions
5. compute uncertainty and cue-based trigger signals
6. invoke semantics only when needed
7. output a final direction decision

The trigger combines:

- relative separability
- entropy
- center-blocked state
- OCR / signage semantic cues

The calibrated operating point used in the report is:

- `tau_delta = 0.08`
- `tau_H = 1.03`

## Main Experiment Stages

### Stage 1: Frame-Level HM3D Evaluation

Purpose:

- validate trigger behavior
- compare selective vs always-on semantic invocation
- compare semantic backends before embodied control

Main scripts:

- `python src/threshold_calibration.py ...`
- `python src/backend_comparison.py ...`
- `python src/evaluate_hm3d.py ...`

### Stage 2: Closed-Loop HM3D Evaluation

Purpose:

- test whether Stage 1 policy distinctions remain meaningful inside an action loop
- compare movement behavior, semantic usage, latency, and recovery behavior

Main scripts:

- `python src/run_closed_loop_hm3d.py ...`
- `python src/evaluate_closed_loop_hm3d.py ...`

### Recorded Real-Video Demo

Purpose:

- verify that the same pipeline runs end to end outside simulation
- inspect OCR/signage cues and qualitative policy differences on real footage

Main scripts:

- `python src/run_recorded_video_demo.py ...`
- `python src/compare_recorded_demo_policies.py ...`

## Typical Setup

### 1. Create environment

```bash
conda create -n etssin python=3.10 -y
conda activate etssin
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Habitat-Sim

Example:

```bash
conda install -c conda-forge -c aihabitat habitat-sim
```

### 4. Confirm dataset paths

The evaluation scripts assume local HM3D assets are available. Check:

- `datasets/hm3d/`
- `minival_datasets/`
- `train_dataset/`
- `val_dataset/`

and review `src/config.py` if you need to adjust paths.

## Common Commands

### Stage 1 evaluation

```bash
python src/evaluate_hm3d.py --split val --max-scenes 5 --frames-per-scene 20 --trajectory-length 15
```

### Trigger calibration

```bash
python src/threshold_calibration.py --split val --max-scenes 100 --frames-per-scene 30
```

### Backend comparison

```bash
python src/backend_comparison.py --split val --max-scenes 5 --frames-per-scene 20 --trajectory-length 15
```

### Stage 2 single rollout

```bash
python src/run_closed_loop_hm3d.py --split val --steps 40 --save-video
```

### Stage 2 full evaluation

```bash
python src/evaluate_closed_loop_hm3d.py --split val --max-scenes 5 --steps 40
```

### Recorded-video demo

```bash
python src/run_recorded_video_demo.py --semantic-policy event_triggered --semantic-backend blip
```

### Recorded-demo policy comparison

```bash
python src/compare_recorded_demo_policies.py
```

## Main Output Artifacts

Examples of report-relevant outputs:

- `outputs/val/calibration/threshold_calibration_heatmap.png`
- `outputs/val/backend_comparison/backend_comparison.png`
- `outputs/plots/stage2_policy_comparison_summary.svg`
- `report/figures/stage2_policy_comparison_summary.png`
- `report/figures/recorded_demo_policy_comparison.png`

Important report files:

- `report/main_full.tex`
- `report/main_ieee.tex`
- `report/main_cvpr.tex`

Important slide file:

- `slides/main.tex`

## Current Status

The repository now contains:

- completed Stage 1 and Stage 2 evaluation code
- final report sections with figures and tables
- a Beamer slide deck
- recorded demo outputs
- a prototype assistive guidance app

The project should currently be understood as:

- a strong prototype research system
- not a certified real-world assistive navigation product
- not a full goal-conditioned navigation benchmark yet

## Known Scope Limits

The current results support:

- frame-level trigger validation
- embodied movement and latency comparison
- systems-level real-video demonstration

They do not yet establish:

- task-level goal success
- SPL / full ObjectNav success metrics
- robust real-world deployment claims

## Recommended Entry Points

If you are reading the project for the first time:

1. start with this README
2. read `report/sections/method.tex`
3. read `report/sections/stage1_results.tex`
4. read `report/sections/stage2_results.tex`
5. open `slides/main.tex` for the presentation view

If you want to run the app:

1. read [app/README.md](/home/cmu/Event-Triggered_Semantic_Scene_Understanding_for_Indoor_Navigation/app/README.md:1)

If you want to inspect the evaluation artifacts:

1. browse `outputs/val/`
2. browse `outputs/demo/`
3. browse `report/figures/`

## Notes

Two practical points:

- `report/main_cvpr.tex` requires the CVPR style files, including `cvpr.sty`
- the mobile/browser camera app may require HTTPS or `localhost` for camera access, as described in `app/README.md`
