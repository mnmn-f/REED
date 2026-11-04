# Configuration and benchmark preparation

## Setup

Use Python 3.10 or later and install compatible PyTorch, torchvision and torchaudio builds for your device. From this directory run:

```sh
pip install -r requirements.txt
git clone https://github.com/he4cs/DASP.git third_party/benchmark
git -C third_party/benchmark checkout f43db39556d9bf14e96b83ea783bb41194a6e97b
```

## Data and pretrained weights

| Resource | Source | Local destination |
| --- | --- | --- |
| Kinetics50 evaluation clips and extracted frames | Download `Kinetics50.zip` from the [benchmark folder](https://drive.google.com/drive/folders/1SWkNwTqI08xbNJgz-YU2TwWHPn5Q4z5b) or [Baidu mirror](https://pan.baidu.com/s/1Xo3IxQyd_fkzMVofDWKYVw?pwd=fnha). | `data/Kinetics50/` |
| Background recordings for audio corruptions | Download `NoisyAudios.zip` from the same benchmark folder. | Extract the recordings under `third_party/benchmark/preprocess/noisy_audios/`. |
| Frozen 50-class action classifier | Download the [Kinetics50 fine-tuned CAV-MAE checkpoint](https://drive.google.com/file/d/1m38uCAfwL--RP6rWtOvGee4i2SfAzbjl/view). | Save as `checkpoints/cav-mae-ft-ks50.pth`. |
| Clean-reference statistics | Included in this release under `assets/`. | The evaluator reads these files directly. |

After extracting the archives, set the following paths **from the REED project root**:

```sh
export DATA_ROOT="$PWD/data"
export KS50_CHECKPOINT="$PWD/checkpoints/cav-mae-ft-ks50.pth"
```

## Prepare Kinetics50-C

Kinetics50-C follows the audio-visual corruption evaluation protocol used by [READ](https://github.com/XLearning-SCU/2024-ICLR-READ). This release uses the corruption generators, class and clip manifests, and data loader from the pinned [benchmark implementation](https://github.com/he4cs/DASP/tree/f43db39556d9bf14e96b83ea783bb41194a6e97b). The visual generator reads its frost textures from that checkout.

Install FFmpeg and ImageMagick (including the MagickWand library) for preprocessing. Run the commands below to generate all video and audio corruptions at severity 5:

```sh
python -m pip install opencv-python scikit-image Wand pydub tqdm
python third_party/benchmark/preprocess/make_corruptions_image.py --corruption all --severity 5 \
  --data_path "$DATA_ROOT/Kinetics50/image_mulframe_val256_k=50" \
  --save_path "$DATA_ROOT/Kinetics50/image_mulframe_val256_k=50-C"
python third_party/benchmark/preprocess/make_corruptions_audio.py --corruption all --severity 5 \
  --data_path "$DATA_ROOT/Kinetics50/audio_val256_k=50" \
  --save_path "$DATA_ROOT/Kinetics50/audio_val256_k=50-C" \
  --noise_path third_party/benchmark/preprocess/noisy_audios
```

The loader uses the following layout:

```text
data/Kinetics50/
├── audio_val256_k=50/
├── audio_val256_k=50-C/<corruption>/severity_5/
├── image_mulframe_val256_k=50/
└── image_mulframe_val256_k=50-C/<corruption>/severity_5/
```

## Evaluation

Set GAIN_MAX to the desired upper bound on the radial amplitude gain in Eq. (6).

```sh
python evaluate.py --benchmark third_party/benchmark --data "$DATA_ROOT" \
  --checkpoint "$KS50_CHECKPOINT" --scenario S2 --protocol continual \
  --severity 5 --gain-max "$GAIN_MAX"
```

S1 follows the official interleaved order, S2 selects video corruptions, and S3 selects audio corruptions. Episodic evaluation starts a new adapter and optimizer for each condition. Continual evaluation carries them across conditions. Each batch produces predictions before one Adam update. The runner reports accuracy per corruption and the arithmetic mean across conditions.

The reference and calibration splits each contain 700 clean audio-visual pairs from the Kinetics-400 training split restricted to KS50 classes. The reference precision matrix is used to measure mean deviation. Token reanchoring subtracts the batch-to-reference mean shift from the affected modality.

