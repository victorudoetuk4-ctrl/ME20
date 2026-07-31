# Dataset — Skin Disease Dataset (Acne vs Eczema)

## Source

**Name:** Skin Disease Dataset  
**Provider:** Kaggle — [pacificrm/skindiseasedataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset)  
**License:** Community Data License Agreement

## Description

The dataset contains images of **22 skin disease classes** organised into pre-made `train/`
and `test/` splits under a `SkinDisease/` root folder:

```
SkinDisease/
├── train/
│   ├── Acne/
│   ├── Eczema/
│   └── ... (20 other classes)
└── test/
    ├── Acne/
    ├── Eczema/
    └── ... (20 other classes)
```

This project uses **only** the `Acne` and `Eczema` class folders.

| Class folder | Label | Description |
|---|---|---|
| `Acne` | Acne | Inflammatory skin condition with pimples, blackheads, and whiteheads |
| `Eczema` | Eczema | Chronic inflammatory condition causing itchy, red, and cracked skin |

## Split Strategy

The dataset provides pre-made `train/` and `test/` folders. The notebook copies images into
a local split directory and carves out a **15% validation set** from the training data per class:

| Split | Source |
|-------|--------|
| `train` | 85% of dataset `train/` |
| `val`   | 15% of dataset `train/` |
| `test`  | Full dataset `test/` (unchanged) |

## Download (in notebook)

```python
import kagglehub
path = kagglehub.dataset_download("pacificrm/skindiseasedataset")
```

> The dataset is **not** stored in this repository. Run the notebook cell above on Google Colab
> to download it automatically.
