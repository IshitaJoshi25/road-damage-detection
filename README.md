# Road Damage Detection and Severity Assessment

Detects four road-damage types in road images with a lightweight YOLO detector
trained on the **India subset of RDD2022**, and serves it through a Streamlit app.
Each detected defect also gets a Low / Medium / High severity: from a small
classifier trained on crops you label yourself, or, until that exists, from a
clearly marked rule-based estimate.

| ID | RDD2022 code | Class name           |
|----|--------------|----------------------|
| 0  | D00          | `longitudinal_crack` |
| 1  | D10          | `transverse_crack`   |
| 2  | D20          | `alligator_crack`    |
| 3  | D40          | `pothole`            |

> **Severity limitation.** RDD2022 provides road-damage detection annotations, but
> Low/Medium/High severity labels are not part of the original dataset. Severity
> therefore comes from our own labeling plus a trained classifier. Until that
> classifier exists, the app shows a rule-based estimate marked "(estimate)"
> (see [section 10](#10-severity-assessment)).

---

## 1. Architecture

```
 RDD2022 India download (JPG + Pascal VOC XML)
            │
            ▼
 src/inspect_dataset.py ──► reports/class_distribution.{csv,png}
            │
            ▼
 src/convert_voc_to_yolo.py ──► data/interim/labels/*.txt + manifest.csv
            │                     reports/conversion_issues.csv
            ▼
 src/split_dataset.py ──► data/yolo/{images,labels}/{train,val,test}
            │               data/yolo/data.yaml, split_summary.json
            ▼
 src/train_detector.py ──► runs/detect/<run>/weights/best.pt
            │               └─ copied to models/road_damage_yolo.pt
            ▼
 src/evaluate_detector.py ──► reports/evaluation/ (metrics, PR curves, confusion matrix)
            │
            ▼
 src/inference.py  (load_detector / predict_image / draw_predictions / analyze_image)
      │        │
      │        └──► src/severity.py  trained classifier (models/severity_classifier.pt)
      │                               └─ or rule-based estimate if no classifier exists yet
      ▼
 app.py (Streamlit)

 Severity labeling loop:
 src/crop_defects.py ──► data/severity_crops/*.jpg + metadata.csv
            ▼
 label_severity.py (Streamlit, rules in docs/severity_rubric.md) ──► labels.csv
            ▼
 src/train_severity.py ──► models/severity_classifier.pt + reports/severity/
```

Shared helpers live in `src/common.py` (config, paths, seeding, device selection)
and `src/voc_dataset.py` (recursive file discovery, image/XML matching, VOC parsing).

## 2. Project structure

```
road-damage-detection/
├── app.py                  Streamlit UI
├── label_severity.py       Streamlit tool for labeling severity crops
├── config.yaml             All paths and hyper-parameters
├── requirements.txt
├── data/
│   ├── raw/                Put the RDD2022 India download here (not committed)
│   ├── interim/            Converted labels + manifest (generated)
│   ├── yolo/               Final YOLO dataset + data.yaml (generated)
│   └── severity_crops/     Defect crops, metadata.csv and your labels.csv
├── src/
│   ├── common.py, voc_dataset.py
│   ├── inspect_dataset.py, convert_voc_to_yolo.py, split_dataset.py
│   ├── train_detector.py, evaluate_detector.py
│   ├── inference.py, crop_defects.py, severity.py, train_severity.py
├── docs/severity_rubric.md Severity labeling rules (adapted from ASTM D6433)
├── colab/colab_train.ipynb Self-contained Google Colab training notebook
├── tools/                  build_colab_notebook.py (regenerates the notebook)
├── reports/                Generated reports (not committed)
├── runs/                   Ultralytics training runs (not committed)
└── models/                 Exported weights (not committed)
```

All paths in `config.yaml` are resolved relative to the project root, so the
scripts work from any working directory on Windows and Linux.

## 3. Dataset source

RDD2022 (Road Damage Dataset 2022), Arya et al., Crowdsensing-based Road Damage
Detection Challenge. Paper: <https://arxiv.org/abs/2209.08538>. The data is
distributed via the authors' GitHub (<https://github.com/sekilab/RoadDamageDetector>)
and figshare. Download **RDD2022_India.zip** yourself and check its license terms.
The scripts never download the dataset.

## 4. Expected dataset structure

Extract the archive anywhere inside `data/raw/RDD2022_India/`. The official
archive usually looks like this:

```
data/raw/RDD2022_India/
└── India/
    ├── train/
    │   ├── images/                India_000001.jpg ...
    │   └── annotations/xmls/      India_000001.xml ...
    └── test/
        └── images/                (no annotations)
```

You don't have to match this layout exactly. Images (`.jpg/.jpeg/.png`) and
`.xml` files are found **recursively** and matched by filename stem. Only
images that have a matching XML file are used. The unlabeled `test/images`
folder is reported as "unmatched" and ignored, because an image without an XML
file cannot be treated as "no damage". Our own test split is carved out of the
annotated training data.

## 5. Installation

Python 3.10+.

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

# NVIDIA GPU only: install a CUDA build of PyTorch FIRST, e.g. for CUDA 12.1
# (pick the right command for your driver at https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Check whether PyTorch can see your GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

`device: auto` in `config.yaml` uses GPU 0 when CUDA is available and the CPU otherwise.

## 6. Dataset preparation

After the RDD2022 India download is extracted into `data/raw/RDD2022_India/`:

### Step 1: Inspect

```bash
python src/inspect_dataset.py
```

Prints the number of images, XML files, matched pairs and unmatched files, the
folders they were found in, the per-class instance counts and image-size
statistics. Saves `reports/class_distribution.csv` and `.png`. Unreadable XML
files are listed in `reports/inspection_problems.txt`, and the script keeps going.

### Step 1b (optional): Check the annotations visually

```bash
python src/visualize_annotations.py                        # 40 random boxes per class
python src/visualize_annotations.py --classes D40 --n 80   # 80 potholes
```

This writes files to `reports/annotation_samples/`:
- `<class>/sheet.jpg`: a contact sheet with one tile per sampled box.
- `<class>/<image>.jpg`: the full image with all of its XML boxes drawn.
- `box_size_stats.csv`: box sizes per class.
- `review.csv`: fill in the `verdict` column (`ok` / `wrong_class` / `bad_box` / `not_damage`).

The script also warns about photos with an EXIF rotation tag, since their boxes
can appear shifted. If Step 2 has already run, it also checks that every
converted YOLO box, drawn as a thin white box, matches its XML box.

### Step 1c (optional): Add your own labelled images (training only)

Put extra images in class folders under a folder listed in `dataset.extra_dirs`
(default `data/raw/road_damage_assesment`), then create the Pascal VOC XML files:

```bash
# close-ups where the defect fills the whole photo: one full-image box per image
python src/annotate_boxes.py --folder data/raw/road_damage_assesment/dataset/alligator_cracks --class alligator_crack --whole-image
# road scenes: draw boxes in a window (drag = box, 1-4 = class, u = undo, Enter = next, s = skip, q = quit)
python src/annotate_boxes.py --folder data/raw/road_damage_assesment/dataset/potholes --class pothole
```

The tool writes the XML files next to the images. It automatically skips images
under 100 px and exact duplicates of images already labelled. Images without an
XML file, such as ones you skip, are not used. The converter adds these images
to the **training split only** (`source=extra` in the manifest), so validation
and test remain pure RDD2022 India and identical with or without the extra data.
In Colab, zip the folder, put it on Google Drive and set `EXTRA_DATASET`.

### Step 2: Convert annotations

```bash
python src/convert_voc_to_yolo.py
```

Writes one YOLO label file per annotated image (`class cx cy w h`, normalised to
[0, 1]). Boxes outside the image are clipped with a warning. Swapped min/max
coordinates are fixed with a warning. Zero-area boxes are dropped with an error
entry. Classes other than D00/D10/D20/D40 (e.g. D43, D44, D50) are ignored and
counted. Every issue is written to `reports/conversion_issues.csv`. Use
`--force` to redo the conversion.

### Step 3: Split dataset

```bash
python src/split_dataset.py
```

Creates a 70 / 15 / 15 train / val / test split (`split:` in `config.yaml`, seed 42),
copies the images and labels into `data/yolo/` and generates `data/yolo/data.yaml`.
It refuses to overwrite an existing dataset unless you pass `--force`.

**Data leakage.** RDD2022 India filenames (`India_000123.jpg`) carry no
reliable sequence, trip or GPS information, so the default `strategy: random`
is a reproducible per-image random split. Two alternatives are available:

- `strategy: block` groups consecutive image numbers (`block_size`) so that
  neighbouring frames, if they are from the same drive, stay in the same split.
  This is a heuristic.
- `strategy: regex` groups by the first capture group of `group_regex` on the
  filename. Use this if you get real sequence or location identifiers.

With a grouping strategy the ratios apply to groups, so image proportions are
approximate. `split_summary.json` records the strategy, seed and per-split counts.

## 7. Training

```bash
python src/train_detector.py
```

This starts from pretrained `yolo11n.pt` (Ultralytics downloads the official
weights on first use; it does not train from scratch) and uses the `training:`
settings in `config.yaml`. Command-line overrides:

```bash
python src/train_detector.py --model yolo11s.pt              # larger model
python src/train_detector.py --epochs 150 --patience 30
python src/train_detector.py --imgsz 800 --batch 8           # larger images, smaller batch
python src/train_detector.py --device cpu --epochs 1         # quick CPU check
python src/train_detector.py --batch -1                      # Ultralytics auto-batch (GPU)
```

To change the default model permanently, edit `model.name` in `config.yaml`.
If you run out of GPU memory, lower `--batch` or `--imgsz`.

Outputs go to `runs/detect/<run_name>/`: Ultralytics curves, `results.csv`,
`weights/best.pt`, and `run_metadata.json`. The metadata file records the model,
image size, epochs, batch, seed, device, library versions and the dataset split
summary. `best.pt` is also copied to `models/road_damage_yolo.pt`. Weights are
never committed to Git (`.gitignore`).

### Training without a local GPU (Google Colab)

`colab/colab_train.ipynb` is a self-contained notebook that already includes
`config.yaml` and every file in `src/`. Upload `RDD2022_India.zip` (or the
extracted `RDD2022_India/` folder) to *My Drive/road-damage/*, open the notebook in Colab, choose a T4 GPU and select
**Run all**. It saves `models/road_damage_yolo.pt`, `reports/` and `runs/` to
*My Drive/road-damage/outputs/* while it trains. If the session disconnects, set
`RESUME = True` and run it again, and training continues from the last epoch
(`python src/train_detector.py --resume` does the same locally).
`tools/build_colab_notebook.py` regenerates the notebook after code changes.

## 8. Evaluation

```bash
python src/evaluate_detector.py                 # test split, models/road_damage_yolo.pt
python src/evaluate_detector.py --split val --weights runs/detect/rdd2022_india/weights/best.pt
```

Prints precision, recall, mAP50 and mAP50-95, overall and per class. Everything
is saved under `reports/evaluation/`:

- `metrics.json` and `per_class_metrics.csv`
- the confusion matrix and PR / P / R / F1 curves that Ultralytics generates
- `examples/`: annotated predictions on sample test images

mAP is computed with the standard low confidence threshold (`evaluation.confidence: 0.001`).

## 9. Streamlit usage

```bash
streamlit run app.py
```

The app uses the **MobileNetV2 classifier**
(`models/road_damage_mobilenetv2.keras.zip`, configured under `classifier` in
`config.yaml`).

1. Upload a JPG, JPEG, PNG or WEBP photo of a road surface.
2. Click **Detect Damage**.

The page shows:
- the uploaded image next to the result image, which has a coloured banner (downloadable);
- summary cards for the **defect type**, the **confidence** and the **severity** (a rule-based estimate);
- a probability bar for each of the four damage types.

When the top probability is below `classifier.min_confidence` (default 0.6),
the app warns that the model is unsure and names the runner-up class. The
sidebar describes the model and the four damage types. The model file path
can be changed under **Advanced**.

The YOLO detector scripts (`train_detector.py`, `evaluate_detector.py`,
`inference.py`) remain in the project, but the app no longer uses them.

### MobileNetV2 classifier (whole-image)

The app can also use a Keras MobileNetV2 classifier, configured in the
`classifier` section of `config.yaml` (default path
`models/road_damage_mobilenetv2.keras.zip`). It is the default in the
sidebar's **Model** selector. The classifier reports the damage type of the
whole photo with probabilities for all four classes; it does **not** draw
bounding boxes. Keras runs on the PyTorch backend, so TensorFlow is not needed.
The output order `[alligator_crack, longitudinal_crack, pothole, transverse_crack]`
(alphabetical training-folder order) and the `[-1, 1]` pixel scaling were
confirmed on the labeled category folders. For whole-image predictions, the
rule-based severity uses only dark-pixel density, because the defect's size is
unknown.

## 10. Severity assessment

RDD2022 provides road-damage detection annotations, but Low/Medium/High
severity labels are not part of the original dataset. This project therefore
provides two severity estimators, chosen by `severity.method` in `config.yaml`:

| Estimator | When it is used | Shown in the app as |
|---|---|---|
| **Trained classifier** (MobileNetV3-small on defect crops) | `models/severity_classifier.pt` exists | `Medium (82%)` |
| **Rule-based estimate** | No classifier trained yet | `Medium (estimate)` |

`method: auto` (the default) uses the classifier when it exists and the
rule-based estimate otherwise. You can also set `model`, `heuristic` or `none`.

**Rule-based estimate.** The score combines two measurable cues:
`score = 0.6 × size + 0.4 × density`.
- *size* is the crack length relative to the image for longitudinal and
  transverse cracks, and the square root of the area fraction for potholes and
  alligator cracks.
- *density* is the fraction of pixels inside the box that are clearly darker
  than the surrounding road, a proxy for crack width or pothole depth.

A score below 0.33 is Low, below 0.66 is Medium, and anything higher is High.
These thresholds are documented starting values in `config.yaml`. They are
**not** learned from ground truth, and the app always labels the result as an
estimate.

### Training the severity classifier

1. **Create crops.** This can run before the detector is trained, because it uses
   the annotated boxes:
   ```bash
   python src/crop_defects.py --source data/yolo/images/train --ground-truth --max-per-class 150 --save-crops
   ```
   This writes `data/severity_crops/<image>_gt_001.jpg` and `metadata.csv`.
   Without `--ground-truth`, the script crops the trained detector's detections
   instead, for example on the unlabeled official test images:
   `--source data/raw/RDD2022_India/India/test/images`.
2. **Label them:**
   ```bash
   streamlit run label_severity.py
   ```
   Click **Low / Medium / High** for each crop, or **Unclear** to skip it. The
   full image is shown next to the crop so you can judge size against the lane.
   The rules are in [docs/severity_rubric.md](docs/severity_rubric.md) and are
   also shown in the tool's sidebar. Labels are saved to
   `data/severity_crops/labels.csv` after every click, so you can stop at any time.
   Aim for at least 30 crops per level; more is better.
3. **Train:**
   ```bash
   python src/train_severity.py            # CPU is fine: a few hundred crops train in minutes
   ```
   This fine-tunes an ImageNet-pretrained MobileNetV3-small. The validation
   split is stratified by level and grouped by source image. The best model is
   saved to `models/severity_classifier.pt`. `reports/severity/` gets
   classification reports and confusion matrices for **both** the classifier and
   the rule-based estimate on the same validation crops, so you can report how
   well each one agrees with human judgement.
4. **Use it:** restart `streamlit run app.py`. The app switches to the trained
   classifier automatically.

## 11. Future work

- More severity labels from several annotators, with inter-annotator agreement measured.
- Calibrating the rule-based thresholds against the human labels.
- Group-aware splitting using real sequence/GPS metadata, if it becomes available.
- Adding other RDD2022 countries for more training data, then evaluating on India only.
- Hyper-parameter tuning, larger models (`yolo11s/m`), and higher input resolution for thin cracks.
- Export to ONNX/TensorRT for edge deployment.
- Video/stream inference and GPS-tagged damage maps.
