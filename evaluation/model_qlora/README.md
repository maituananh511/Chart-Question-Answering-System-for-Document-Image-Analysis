# Evaluation Notebooks

Fine-tuning vision-language models with QLoRA on a Vietnamese Chart QA dataset. Results are evaluated against baseline and compared across architectures.

---

## Dataset

| Split | Samples |
|---|---|
| viet_chart_dataset (train) | ~44,671 |
| vi_chart_dataset (test) | 500 |
| viet_chart_vqa (test) | 200 |
| **Total (train)** | **~44,671** |

---

## Train Config

| Parameter | Value |
|---|---|
| `technique` | QLoRA |
| `batch_size` | 8–16 |
| `dtype` | bfloat16 |
| `device` | CUDA (single GPU) |

---

## Metrics

`BLEU` · `METEOR` · `ROUGE-1` · `ROUGE-2` · `ROUGE-L` · `BERTScore`

Vietnamese tokenization via `underthesea`.

---

## Models

### Vintern-1B-v2 + LoRA
- **Notebook:** `eval_vintern_lora.ipynb`
- **Repo:** `5CD-AI/Vintern-1B-v2`
- **Type:** Vision-language model (baseline)
- **Details:** Fine-tuned on the combined chart dataset. Serves as the reference baseline for all other evaluations.

### Qwen2-VL-2B-Instruct + QLoRA
- **Notebook:** `eval_qwen2_vl_lora.ipynb`
- **Repo:** `Qwen/Qwen2-VL-2B-Instruct`
- **Type:** Vision-language model
- **Details:** QLoRA fine-tuned on the chart dataset. Workflow: Load Base → QLoRA Fine-tune → Merge Weights → Eval.

### InternVL2-1B + QLoRA
- **Notebook:** `merge_lora_internvl_trained_and_eval.ipynb`
- **Repo:** `OpenGVLab/InternVL2-1B`
- **Type:** Vision-language model
- **Details:** Requires manual LoRA weight merging before evaluation. Key mapping: `language_model.base_model.model.*` → `language_model.*`.

---

## Baseline

**Vintern-1B-v2 + LoRA** fine-tuned on `vi_chart_dataset` + `viet_chart_vqa` (~44k samples). Results are loaded from a precomputed CSV in most notebooks rather than re-run.

---

## Requirements

```bash
pip install transformers>=4.49.0
pip install datasets huggingface_hub hf_transfer
pip install torch torchvision
pip install bitsandbytes peft accelerate
pip install rouge_score evaluate
pip install bert-score nltk
pip install underthesea            # Vietnamese tokenizer
pip install numpy==1.26.4
```

**For Qwen models:**
```bash
pip install qwen-vl-utils
```

**For InternVL2:**
```bash
pip install peft==0.11.1 timm einops
git clone https://github.com/OpenGVLab/InternVL.git
pip install -e InternVL/internvl_chat/
```

> Note: some notebooks (Gemma-2, Llama, InternVL2) require `transformers==4.44.2` — check the first cell of each notebook.

---

## Output

Each notebook saves a per-sample CSV and prints a summary of all metrics to the console.
