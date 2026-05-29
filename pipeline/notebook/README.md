# Training notebook — Kaggle (free P100 GPU)

This folder holds the Kaggle notebook that fine-tunes the vehicle classifier each month. **Kaggle gives every account 30 hours/week of P100 GPU time for free** — more than enough for a monthly retrain (~30-45 min) and any number of experiments.

## Setup (one time)

1. Create a free Kaggle account at https://www.kaggle.com.
2. Verify your phone (required to enable GPU access).
3. Get an API token from your account settings — drop `kaggle.json` into `~/.kaggle/` on your laptop and as `KAGGLE_USERNAME`/`KAGGLE_KEY` secrets in your GitHub repo.
4. Create a notebook called `vov-vehicle-classifier`. Paste in `train.ipynb` from this folder.
5. In the notebook settings, set:
   - Accelerator: GPU P100
   - Internet: ON (so it can fetch the dataset from your repo / Hugging Face)
   - Persistence: Output

## Monthly retrain flow

Either:

- **Manual (5 min):** Open the Kaggle notebook → "Run all" → wait ~30 min → click "Save Version (Quick Save)". Then trigger the `publish-model.yml` GitHub Action with a version number. That's it.

- **Automated (zero clicks/month):** Use the [Kaggle API to push & run kernels](https://github.com/Kaggle/kaggle-api#push-a-new-kernel-version). A monthly cron job in GitHub Actions does `kaggle kernels push` → `kaggle kernels status` (poll) → `kaggle kernels output` → trigger `publish-model.yml`.

## What the notebook does

1. Clones the pipeline repo, reads `pipeline/data/models.json` to know the label space.
2. Downloads accepted photos from the staged dataset (Wikimedia-fetched + customer corrections).
3. Builds train / val / test splits stratified by class.
4. Loads MobileNetV3-Small pretrained on ImageNet (via timm).
5. Fine-tunes the head + last 2 blocks for ~30 epochs with augmentation tuned to valet-stand conditions (variable lighting, perspective, partial occlusion).
6. Evaluates on a hold-out test set, reports per-year accuracy.
7. Exports ONNX (opset 17) for browser inference.
8. Saves `vehicle-classifier-v{version}.onnx`, `vehicle-classifier-v{version}.labels.json`, and `metadata.json` to `/kaggle/working/`.

The GitHub Action picks those up and creates a Release.

## Why not Colab / HF Spaces / etc

| Option | Free GPU? | Stable? | Notes |
|---|---|---|---|
| **Kaggle** | ✓ P100 / T4 / TPU, 30 hr/week | ✓ | Wins on stability + time budget |
| Google Colab Free | ✓ T4 (occasionally) | △ disconnects, timeouts | Fine for experimentation, not CI |
| Colab Pro | $10/mo, A100 | ✓ | If you want fully automated retrains |
| HF Spaces Free | CPU only | ✓ | Too slow for training |
| HF Spaces Pro | $9/mo, A10G | ✓ | Same as Colab Pro tradeoff |
| Lambda Labs spot | $0.50-2/hr | ✓ | Pay-per-use, ~$1-2 per retrain |

Kaggle is the cleanest $0 path for our monthly cadence.
