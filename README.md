# Global Wheat Detection with Source-Aware Validation

An object-detection project for the [Kaggle Global Wheat Detection competition](https://www.kaggle.com/competitions/global-wheat-detection). The project studies how a wheat-head detector generalises across seven acquisition sources and builds an auditable inference pipeline around YOLO11m.

The main engineering work is not limited to calling a detector. It includes annotation auditing, source-aware validation, a handwritten competition evaluator, memory-safe batched inference, coordinate validation, intermediate prediction storage, model fusion, non-maximum suppression, and submission checks.

## Highlights

- Audited 3,373 labelled images and 147,793 wheat-head bounding boxes.
- Kept 49 images without annotation rows outside supervised training instead of treating them as confirmed negatives.
- Used seven-fold Leave-One-Source-Out validation to reduce leakage between visually related acquisition sources.
- Trained YOLO11m at the original 1024 by 1024 resolution.
- Implemented the competition IoU, one-to-one matching, and multi-threshold image score in Python.
- Used explicit small inference batches and per-image NPZ files to avoid unnecessary GPU memory growth.
- Combined seven fold-specific models with weighted box fusion and final NMS.
- Added structural validation for the generated Kaggle submission.

## Results

The repository contains results from several evaluation stages. These metrics use different protocols and should not be compared as if they were the same score.

| Evaluation | Metric | Result |
| --- | --- | ---: |
| Fold 3 with `ethz_1` held out | Precision | 0.878 |
| Fold 3 with `ethz_1` held out | Recall | 0.840 |
| Fold 3 with `ethz_1` held out | Ultralytics mAP at IoU 0.50 | 0.873 |
| Fold 3 with `ethz_1` held out | Ultralytics mAP at IoU 0.50 to 0.95 | 0.342 |
| Seven-fold LOSO validation | Competition mAP at IoU 0.50 to 0.75, macro average | 0.5030 |
| Seven-fold LOSO validation | Competition mAP at IoU 0.50 to 0.75, image-weighted average | 0.4480 |
| Final Kaggle submission | Public leaderboard | 0.6560 |
| Final Kaggle submission | Private leaderboard | 0.5465 |

The final seven-model inference run reduced 7,221 raw candidate boxes to 870 final boxes across ten test images. The recorded run used a 1024-pixel input, test-time augmentation, a pre-confidence threshold of 0.03, WBF IoU of 0.55, final confidence threshold of 0.22, and NMS IoU of 0.70.

## Dataset

The original competition data has the following structure:

| Item | Value |
| --- | ---: |
| Labelled training images | 3,373 |
| Bounding boxes | 147,793 |
| Acquisition sources | 7 |
| Images without annotation rows | 49 |
| Competition test images | 10 |
| Image size | 1024 by 1024 |

The seven sources are `arvalis_1`, `arvalis_2`, `arvalis_3`, `ethz_1`, `inrae_1`, `rres_1`, and `usask_1`.

## Method

### Data audit and conversion

The preparation process groups row-level annotations by image, compares CSV identifiers with files on disk, checks for invalid or out-of-bounds boxes, and converts `[x, y, width, height]` annotations to the formats required by YOLO training and evaluation.

### Source-aware validation

Each validation fold contains one complete acquisition source, while the remaining six sources are used for training. This design avoids putting images with the same acquisition style in both training and validation sets.

### Model training

The recorded experiments use YOLO11m with COCO-pretrained weights, a 1024-pixel input, batch size 2, and 30 epochs per fold. The training configuration uses horizontal flipping, HSV perturbation, rotation, translation, and scale augmentation while disabling Mosaic and MixUp for the dense small-object setting.

### Handwritten numerical evaluation

`scripts/evaluate_kaggle_fold.py` implements:

1. CSV parsing and conversion from `[x, y, width, height]` to corner coordinates.
2. Intersection over union without calling a ready-made metric implementation.
3. Confidence-ordered greedy matching with one target matched at most once.
4. `TP / (TP + FP + FN)` at IoU thresholds from 0.50 to 0.75.
5. Per-image and overall score aggregation.
6. JSON summaries and per-image CSV diagnostics.

### Inference and ensembling

The inference pipeline processes small batches, normalises and clips predicted coordinates, removes invalid boxes, sorts predictions by confidence, and stores one compressed NPZ file per fold and image. The final ensemble uses fold-specific model weights, test-time augmentation, weighted box fusion, confidence filtering, and final NMS before writing the Kaggle submission.

## Repository Structure

```text
global-wheat-detection/
|-- scripts/
|   `-- evaluate_kaggle_fold.py
|-- final_package_extract/
|   |-- gwhd/input/notebook/
|   |   `-- global-detection.ipynb
|   |-- results/
|   |   |-- _fold_infer.py
|   |   |-- fold_predictions/fold0...fold6/
|   |   |-- test_images.json
|   |   `-- submission.csv
|   `-- submissions/
|       |-- gwhd_wbf_report_optimized.json
|       `-- submission.csv
|-- report_output/
|   |-- 全球小麦穗检测项目双语报告_最终版.pdf
|   `-- 全球小麦穗检测项目双语报告_最终版.docx
|-- source_text_utf8.txt
`-- README.md
```

The report-generation helpers and extracted DOCX resources in the repository are supporting materials rather than detector source code.

## Environment

Python 3.10 or a compatible version is recommended. Install the PyTorch build appropriate for your CPU or CUDA environment first, following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/), then install the remaining packages.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas opencv-python ultralytics
```

For Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Required External Files

The raw competition dataset, fold split files, and trained model weights are not part of the compact repository package. To run the fold evaluator locally, arrange the required files as follows:

```text
global-wheat-detection/
|-- train.csv
|-- train/
|   |-- <image_id>.jpg
|   `-- ...
|-- splits/
|   |-- fold_0_val.txt
|   |-- ...
|   `-- fold_6_val.txt
`-- runs/
    `-- <experiment>/weights/best.pt
```

Download the images and `train.csv` from the Kaggle competition page. Model weights must be trained locally or supplied separately.

## Evaluate One Fold

After preparing the dataset, validation IDs, and a trained checkpoint, run:

```powershell
python scripts/evaluate_kaggle_fold.py `
  --fold 3 `
  --model runs/yolo11m_1024_loso_fold3/weights/best.pt `
  --conf 0.05 `
  --iou 0.60 `
  --batch 2 `
  --device 0
```

On Bash, replace the PowerShell backticks with backslashes.

The evaluator writes results under:

```text
runs/kaggle_eval_fold3/
|-- summary_conf*_iou*.json
`-- per_image_conf*_iou*.csv
```

Use `--device cpu` when CUDA is unavailable.

## Kaggle Notebook

`final_package_extract/gwhd/input/notebook/global-detection.ipynb` is a compact submission notebook that:

- loads a YOLO checkpoint;
- installs Ultralytics from mounted wheel files when internet access is disabled;
- discovers mounted test images;
- converts predictions to the competition format;
- writes `submission.csv`; and
- validates headers, token counts, confidence values, coordinates, and box dimensions.

Update `WEIGHTS_PATH` and the mounted dataset paths before running it on Kaggle.

The notebook stored in this package is a compact single-model submission notebook. The final seven-model WBF experiment is documented by the project report, prediction artifacts, final CSV, and `gwhd_wbf_report_optimized.json`; the full training code and model checkpoints are not included in this repository snapshot.

## Reproducibility Notes

- The recorded random seed is 42.
- The fold evaluator imports Ultralytics only inside `main`, so its numerical functions can be tested independently.
- Predictions are sorted by confidence before greedy matching.
- The evaluator uses IoU thresholds `0.50, 0.55, 0.60, 0.65, 0.70, 0.75`.
- The final JSON report records model paths, model weights, inference settings, counts, elapsed time, and validation errors.
- The ten competition test images do not have public ground truth in this package. Their output can be checked structurally, but not assigned a local accuracy score.

## Limitations

- The detector architecture, pretrained weights, optimiser, and CUDA tensor operations are provided by Ultralytics and PyTorch.
- The repository snapshot does not include the raw dataset, seven trained checkpoints, or the complete training pipeline.
- The final leaderboard results cannot be reproduced from the included files alone without the missing checkpoints and competition data.
- Performance differs substantially by acquisition source, especially in dense scenes and under strict localisation thresholds.

## Acknowledgements

- [Global Wheat Detection competition](https://www.kaggle.com/competitions/global-wheat-detection)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)

