# Project Documentation

This repository implements the **Degradation-Aware Speech Enhancement using Residual Expert Networks** project.

## Overview
The system is built around a pretrained wav2vec2 encoder with four specialized residual experts targeting distinct degradation types. Augmented datasets are validated for realism using distribution, perceptual, and embedding similarity metrics. Training follows a staged procedure with freezing/unfreezing of the backbone and expert fusion strategies.

## Directory Structure and File Roles

```
Project_implementation/
├─ data/
│  ├─ augmentation.py        # routines to create synthetic degradations per expert
│  ├─ validation.py          # functions to check realism of generated data
├─ models/
│  ├─ backbone.py            # wrapper around wav2vec2 encoder
│  ├─ expert.py              # residual expert definitions and expert set
│  ├─ decoder.py             # waveform reconstruction module
│  ├─ fusion.py              # routing/gating logic for experts
│  ├─ training.py            # training loop, losses, and staging logic
├─ experiments/
│  ├─ run_experiments.py     # entry script for running model comparisons/ablations
├─ utils/
│  ├─ metrics.py             # evaluation metric wrappers (PESQ, STOI, SI-SDR, etc.)
│  ├─ audio_utils.py         # I/O helpers for loading and saving audio
├─ docs/
│  ├─ project_overview.md    # this documentation file
├─ README.md                 # top‑level instructions and setup
```

### Notable Points

* **data/augmentation.py** contains four functions corresponding to each expert's augmentation strategy (noise, reverberation, device distortion, physiological weakness).  Each will later implement parameterized transformations using libraries like `pydub`, `scipy`, or custom DSP code.

* **data/validation.py** provides objective and perceptual validation utilities implementing distribution matching, DNSMOS/NISQA scoring, and embedding similarity via wav2vec2.

* **models/backbone.py** handles loading the wav2vec2 encoder, freezing parameters, and unfreezing layers during Stage 2 training.

* **models/expert.py** defines a `ResidualExpert` and a container `ExpertSet` for either hard selection or weighted combinations.

* **models/decoder.py** reconstructs waveform from latent features with optional STFT consistency, using `ConvTranspose1d` layers.

* **models/fusion.py** includes a trivial hard router and a simple gating network for soft expert weighting.

* **models/training.py** orchestrates the training process with loss functions (waveform L1, STFT, SI-SDR) and a `Trainer` class with epoch/validation loops.

* **experiments/run_experiments.py** is a placeholder for running the main comparisons and ablation studies; it will load configurations and instantiate the appropriate models and datasets.

* **utils/metrics.py** and **utils/audio_utils.py** hold generic evaluation and I/O functions to keep other modules clean.

* **docs/project_overview.md** consolidates this high-level blueprint along with reference to each file for developers and readers.

* **README.md** (to be created) will include setup instructions, dependencies, and running guidelines.

## Future Work
The following additional artifacts are planned:

* Exact folder structure is already defined.
* A detailed training pipeline with configuration and scripts.
* Architecture diagram (to be added as `workflow_diagram.html` or a PDF in `docs/`).
* Dataset augmentation code with real implementations and validation scripts.
* Week-by-week implementation roadmap (could be added as another markdown file).

Feel free to explore each module and extend the placeholders with concrete implementations as the research progresses.
