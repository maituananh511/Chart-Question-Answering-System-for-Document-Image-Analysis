# Data Card: Multi-Task Chart Understanding Dataset

## 1. Overview
This dataset is designed to support two specialized tasks in the field of chart analysis, combining various data sources:
1. **Fine-tuning Multimodal LLM (Vintern V2 1B):** Uses `viet_chart_qa_dataset` and `self_built_vietnamese_dataset` to train capabilities for chart question answering and multimodal reasoning.
2. **Classification (ResNet18):** Uses the categorized chart dataset (`Classified Charts`) to train a classifier for identifying chart types, with a separate independent set for final evaluation.

## 2. Model Pipelines

| Model | Task | Dataset Source |
|---|---|---|
| **Vintern V2 1B** | Chart QA & Reasoning | `viet_chart_qa_dataset` & `self_built_vietnamese_dataset` |
| **ResNet18** | Chart Classification | `Classified Charts` (32,364 samples) + External Evaluation Set (608 samples) |

## 3. Dataset Construction & Methodology (Vintern V2 1B)

To train the **Vintern V2 1B** model, we perform data splitting based on a combination of the existing dataset and the self-built dataset:

* **Training Set:** 30,000 samples randomly sampled from `viet_chart_qa_dataset` combined with the remainder of the `self_built_vietnamese_dataset` (totaling 44,671 training samples).
* **Test Set:** 200 samples from `viet_chart_qa_dataset` and 500 samples from `self_built_vietnamese_dataset`.

### 3.1. Distribution Table for Vintern V2 1B

| Dataset Source | Total Train | Total Test |
| :--- | :--- | :--- |
| `viet_chart_qa_dataset` | 30,000 | 200 |
| `self_built_vietnamese_dataset` | 14,671 | 500 |
| **Total** | **44,671** | **700** |

## 4. Data Sources

| Source | Type | Count | Collection Method |
|---|---|---|---|
| arxiv.org | Research paper PDFs | 15,000+ | Automated download via arxiv API |
| Extracted via PyMuPDF | Chart images | 46,911 | PDF page + bounding box extraction |
| Classified via Gemini | Chart type labels | 32,364 | Multi-model classification with confidence thresholding |

## 5. Dataset Composition (ResNet18 Pipeline)

### 5.1. Data Splitting & Evaluation Strategy
The ResNet18 model is trained and validated using the **Classified Chart dataset** (32,364 images). After the training process is complete, the model is evaluated using a strictly **External Independent Test Set**.

| Dataset Component | Count | Usage |
| :--- | :--- | :--- |
| **Training/Val Pool** | 32,364 | Used for training and validation (e.g., 80/10/10 split) |
| **External Test Set** | 608 | **Independent hold-out test** (76 samples/class × 8 classes) |

*Note: The External Test Set is completely separate from the 32,364 training samples to ensure unbiased performance evaluation.*

### 5.2. Classified Charts Distribution (Training Pool)
This data is utilized to train the **ResNet18** model.

| Chart Type | Count | Share |
|---|---|---|
| line | 12,930 | 40.0% |
| scatter | 6,278 | 19.4% |
| bar | 5,745 | 17.8% |
| heatmap | 4,073 | 12.6% |
| histogram | 1,006 | 3.1% |
| box | 880 | 2.7% |
| pie | 835 | 2.6% |
| area | 617 | 1.9% |
| **Total** | **32,364** | 100% |

## 6. Metadata & Schema
Each record in the dataset includes the following key fields:
* `id`: Unique identifier.
* `image`: Chart image.
* `description`: Brief description (used for Vintern).
* `conversations`: Conversation pairs for fine-tuning (used for Vintern).
* `groundtruth`: Ground truth answer for the question (used for Vintern).
* `label`: Chart type category (used for ResNet18).