# TT Method & Advanced Cascade Classifier for Hyperspectral Image Classification

Code repository accompanying the paper submitted to **Journal of Applied Remote Sensing**.

## Overview

This repository contains Python code for multiclass classification of hyperspectral satellite imagery using the **TT method (Tsubomatsu & Tonooka method)** and an **Advanced Cascade Classifier (ACC)**. The methods are evaluated against standard machine learning classifiers on mineral mapping tasks at the Cuprite mining district, Nevada, USA.

---

## Directory Structure

```
GithubACC/
├── README.md                          # This file
├── testClass251206.py                 # Main script: satellite image classification pipeline
├── test10graphGOD5.py                 # Validation script: synthetic data benchmark & graph generation
├── spectral_ele251206.py              # Spectral analysis: separability metrics (JM, TD, Bhattacharyya)
│
├── TTmethod/                          # TT Method classifier package
│   ├── __init__.py
│   └── classTTmethod.py               # TTClassifier: greedy iterative OvR multiclass classifier
│
└── validation_lib/                    # Validation & model library package
    ├── __init__.py
    ├── config.py                      # Configuration class for validation scenarios
    ├── advanced_cascade.py            # AdvancedCascadeClassifier (ACC): cascade + OVR fallback
    ├── NewTT6.py                      # ACC variant with useOVR toggle (used in test10graphGOD5.py)
    ├── feature_expander2.py           # FeatureExpansionClassifier: spectral feature engineering
    ├── high_pref_configs.py           # Pre-defined high-performance ACC configurations
    ├── custom_transformers.py         # SelectFirstKFeaturesAndScale transformer
    └── TTmethod/
        ├── __init__.py
        └── classTTmethod.py           # TTClassifier (copy for validation_lib internal use)
```

---

## Script Descriptions

### `testClass251206.py` — Satellite Image Classification Pipeline
The main classification script. Loads ENVI hyperspectral imagery and TIF label maps, applies multiple classifiers (RF, OVR, XGBoost, MLP, etc. + ACC), and outputs classification maps, accuracy metrics (Accuracy, MCC, Macro F1), and confusion matrices.

**Input data paths** (must be configured before running):
- `feature_data_dir`: Directory containing ENVI format multispectral/hyperspectral feature images (`.hdr` + binary data(should be BIP or BIL formated))
- `label_data_dir`: Directory containing ground-truth label images in TIF format
- `base_output_dir`: Output directory for classification results and images

### `test10graphGOD5.py` — Synthetic Data Validation & Graphs
Generates synthetic multi-class datasets using `sklearn.datasets.make_classification` and evaluates classifier performance across varying conditions (number of classes, class separability, sample size, class imbalance). Produces summary plots comparing all models.

**Usage**: Run from the repository root directory. Output logs, CSVs, and plots are saved to the script directory.

### `spectral_ele251206.py` — Spectral Separability Analysis
Calculates inter-class separability metrics (Bhattacharyya distance, Jeffries-Matusita distance, Transformed Divergence) between mineral classes in hyperspectral images. Generates mean spectra plots, PCA scatter plots, and JM distance histograms.

**Input data paths** (configured in the `__main__` block):
- `base_dir_str`: Base directory for satellite image data
- `target_area`, `feature_subdir`, `label_subdir`: Subdirectory settings

---

## Key Classes

### `TTClassifier` (`TTmethod/classTTmethod.py`)
Greedy iterative One-vs-Rest (OvR) multiclass classifier. Each class is assigned a binary OvR classifier, and the processing order is determined by training-set F1 scores (highest first). Unclassified samples cascade to subsequent classifiers.

### `AdvancedCascadeClassifier` (`validation_lib/advanced_cascade.py`, `validation_lib/NewTT6.py`)
Extended cascade classifier with:
- **Feature transformation**: Optional feature scaling/selection between cascade iterations
- **OVR fallback**: Probability-threshold-based fallback classification for remaining unclassified samples
- **Adaptive threshold**: Automatically determined OVR probability threshold based on validation set performance
- **`useOVR` flag** (NewTT6.py variant): Toggle to enable/disable OVR fallback

### `FeatureExpansionClassifier` (`validation_lib/feature_expander2.py`)
Wrapper classifier that generates additional spectral features (differences, slopes, areas between adjacent bands) before classification.

---

## Requirements

```
Python >= 3.8
numpy
scipy
scikit-learn
matplotlib
pandas
xgboost
lightgbm
spectral        # for ENVI file I/O
rasterio        # for TIF file I/O
GDAL            # for geospatial image processing
Pillow          # for image saving
packaging
```

Install dependencies:
```bash
pip install numpy scipy scikit-learn matplotlib pandas xgboost lightgbm spectral rasterio GDAL Pillow packaging
```

---

## How to Run

### 1. Configure Input Data Paths

Before running, edit the path variables in each script to point to your local data:

- **`testClass251206.py`**: Edit `feature_data_dir`, `label_data_dir`, `base_output_dir` (lines ~149–154)
- **`spectral_ele251206.py`**: Edit `base_dir_str`, `target_area`, `feature_subdir`, `label_subdir` (lines ~698–701)

### 2. Run Scripts

```bash
# Satellite image classification
python testClass251206.py

# Synthetic data validation and graph generation
python test10graphGOD5.py

# Spectral separability analysis
python spectral_ele251206.py
```

---

## Code, Data, and Materials Availability

The source code for this study is available at [https://github.com/tsubomax/ACC](https://github.com/tsubomax/ACC).

AVIRIS data are publicly available from the NASA AVIRIS Data Portal (https://aviris.jpl.nasa.gov/dataportal/). ASTER and EMIT data are available from the NASA EARTHDATA Portal (https://search.earthdata.nasa.gov/). HISUI data can be obtained through the Tellus platform (https://www.tellusxdp.com/).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
