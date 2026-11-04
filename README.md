# REED

**Rectify then Debias: From Reference-Anchored Features to Shift-Informed Supervision in Multi-Modal Test-Time Adaptation**

REED adapts an audio-visual classifier to an unlabeled stream. Three statistical cues compare each modality with a clean reference. When a modality shifts, REED aligns its token mean with the reference and applies radial spectral correction to corrupted video. It updates the clean modality’s stable adapter and the corrupted modality’s plastic adapter using different supervision. The plastic adapter learns from a set of candidate classes weighted by prediction history.

See [Method](docs/METHOD.md) for the algorithm and [Configuration](docs/CONFIGURATION.md) for input requirements, dataset and checkpoint downloads, and benchmark commands.

## Quick start

From the repository root, run:

```sh
python -m pip install torch
python -m examples.minimal
```

The example runs one online update on CPU using synthetic audio and video tokens, a frozen classification head and a supplied video-corruption decision. For evaluation on Kinetics50-C, follow the [setup and data preparation instructions](docs/CONFIGURATION.md).

## Core modules

| Component | Implementation |
| --- | --- |
| Three-cue modality diagnosis | `reed/diagnosis.py` · `JointRouter.predict` |
| Feature statistics and percentile calibration | `reed/statistics.py` |
| Temporal shift detection | `reed/diagnosis.py` · `GlobalChangeClock` and `reed/boundary.py` · `OnlineBoundary` |
| Video spectral correction | `reed/spectrum.py` · `correct` |
| Token mean alignment, candidate targets and adapter updates | `reed/adaptation.py` · `OnlineAdapters.step` |
| Prediction on an unlabeled batch | `reed/runtime.py` · `REED.predict_and_update` |

## Repository layout

```text
REED/
├── reed/
│   ├── __init__.py
│   ├── diagnosis.py
│   ├── statistics.py
│   ├── boundary.py
│   ├── spectrum.py
│   ├── adaptation.py
│   └── runtime.py
├── assets/                 # Fitted KS50 reference and calibration statistics
├── examples/minimal.py
├── docs/
│   ├── METHOD.md
│   └── CONFIGURATION.md
├── evaluate.py             # Kinetics50-C evaluation entry point
├── requirements.txt
└── README.md
```

## Use with an audio-visual stream

Supply a frozen KS50 CAV-MAE classifier and batches in temporal order. The audio input is a normalized filterbank tensor and the video input is a normalized RGB frame tensor. Both tensors must be on the model’s device.

```python
from reed.runtime import REED

state = REED(model, assets="assets", gain_max=gain_max, beta=0.9)
for audio, video in stream:
    probabilities = state.predict_and_update(audio, video)
    predicted_classes = probabilities.argmax(dim=1)
```

Each call returns the predictions made before that batch’s parameter update. Reuse the state across corruption conditions for continual evaluation. Create a new state at each condition for episodic evaluation. The bundled calibration tables support batches of 64 and the final batch of 34 in each KS50-C condition.
