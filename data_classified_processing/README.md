# Notebooks Directory

## Data Factory Notebooks

```
data_collection --> image_extraction --> chart_detection --> chart_classification
```

### data_collection.ipynb

- Source: arXiv API (cs.CV, cs.LG, cs.CL, cs.AI, stat.ML)
- Target: 10,000 PDFs, 200 papers per query, rate limit 3s

### image_extraction.ipynb

- Method: PyMuPDF (fitz) embedded image extraction
- Filters: min 100px, max 4096px, min area 10,000px²

### chart_detection.ipynb

- Model: `yolo`, confidence threshold 0.5

### chart_classification.ipynb

- Model: `resnet18`
- Classes: `area`, `bar`, `box`, `heatmap`, `histogram`, `line`, `pie`, `scatter`
- Confidence threshold: 0.7 (below → `uncertain/`)

---

## Prerequisites

```bash
pip install -e .
pip install torch torchvision ultralytics pymupdf arxiv
pip install jupyterlab ipywidgets matplotlib tqdm
```