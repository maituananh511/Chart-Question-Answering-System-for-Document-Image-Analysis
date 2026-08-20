# Evaluation Notebooks

Zero-shot evaluation of base models (no fine-tuning) on a Vietnamese Chart VQA dataset. Results are compared against a fine-tuned Vintern-1B-v2 + LoRA baseline.

---

## Dataset

| Split | Samples |
|---|---|
| vi_chart_dataset (test) | 500 |
| Data Vietnamese (viet_chart_vqa) | 200 |
| **Total** | **700** |

---

## Eval Config

| Parameter | Value |
|---|---|
| `max_new_tokens` | 64 |
| `batch_size` | 8–16 |
| `dtype` | bfloat16 |
| `device` | CUDA (single GPU) |

---

## Metrics

`BLEU` · `METEOR` · `ROUGE-1` · `ROUGE-2` · `ROUGE-L` · `BERTScore`

Vietnamese tokenization via `underthesea`.

---

## Models

### Gemma-2-2B-it
- **Notebook:** `eval_gemma2_vs_vintern_lora.ipynb`
- **Repo:** `google/gemma-2-2b-it`
- **Type:** Text-only causal LM
- **Compared against:** Vintern-1B-v2 LoRA (precomputed CSV)

### Qwen2.5-VL-2B-Instruct
- **Notebook:** `eval_qwen2_5_vl_2b_instruct_vs_vintern_lora_finetuned.ipynb`
- **Repo:** `Qwen/Qwen2-VL-2B-Instruct`
- **Type:** Vision-language model
- **Compared against:** Vintern-1B-v2 LoRA (precomputed CSV)

### Qwen3-VL-2B-ChartQA
- **Notebook:** `eval_qwen3_vl_2b_chartqa.ipynb`
- **Repo:** `Nhaass/Qwen3-VL-2B-ChartQA`
- **Type:** Vision-language model, pre-trained on ChartQA (not this dataset)

### SeaLLMs-v3-1.5B-Chat
- **Notebook:** `eval_seallms_v3_1_5b_vs_vintern_lora_finetuned.ipynb`
- **Repo:** `SeaLLMs/SeaLLMs-v3-1.5B-Chat`
- **Type:** Text-only causal LM with Southeast Asian language support
- **Compared against:** Vintern-1B-v2 LoRA (precomputed CSV)

### StableLM-2-1.6B-Chat
- **Notebook:** `eval_stablelm2_1_6b_vs_vintern_lora_finetuned.ipynb`
- **Repo:** `stabilityai/stablelm-2-1_6b-chat`
- **Type:** Text-only causal LM
- **Compared against:** Vintern-1B-v2 LoRA (precomputed CSV)

### Llama-3.2-1B-Instruct
- **Notebook:** `eval_vintern_lora_vs_llama_3_2_1b_instruct.ipynb`
- **Repo:** `unsloth/Llama-3.2-1B-Instruct`
- **Type:** Text-only causal LM
- **Compared against:** Vintern-1B-v2 LoRA (runs directly in the same notebook)

---

## Baseline

**Vintern-1B-v2 + LoRA** fine-tuned on `vi_chart_dataset` + `viet_chart_vqa` (44k samples). Results are loaded from a precomputed CSV in most notebooks rather than re-run.

---

## Requirements

```bash
pip install transformers>=4.49.0
pip install datasets huggingface_hub hf_transfer
pip install torch torchvision
pip install qwen-vl-utils          # Qwen2.5-VL and Qwen3-VL only
pip install rouge_score evaluate
pip install bert-score nltk
pip install underthesea            # Vietnamese tokenizer
pip install numpy==1.26.4
```

> Note: some notebooks require `transformers==4.44.2` (Gemma-2, Llama) — check the first cell of each notebook.

---

## Output

Each notebook saves a per-sample CSV and prints a summary of all metrics to the console.