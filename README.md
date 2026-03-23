# 🧠 LLM Fine-Tuning with LoRA and QLoRA

Fine-tuning **Microsoft Phi-2** to write in the style of Ernest Hemingway, using Parameter-Efficient Fine-Tuning (PEFT) with LoRA and QLoRA. Built for Google Colab with a T4 GPU.

This project is structured as two progressive notebooks — the first covers data preparation and model loading, and the second covers the actual fine-tuning process using LoRA/QLoRA.

---

## 📂 Project Structure

```
LLM-Fine-tuning/
│
├── Data_Preparation_and_Model_Loading.ipynb   # Notebook 1: Data prep, tokenization & quantization
├── PEFT_with_LoRA_and_QLoRA.ipynb             # Notebook 2: LoRA config, training & evaluation
├── MenWithoutWomenCleaned.txt                 # Training data (Hemingway)
├── TheSunAlsoRisesCleaned.txt                 # Evaluation data (Hemingway)
└── README.md
```

---

## 📖 Notebooks Overview

### Notebook 1 — Data Preparation & Model Loading

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wendyshiro/LLM-Fine-tuning/blob/main/Data_Preparation_and_Model_Loading.ipynb)

Covers the foundational steps needed before any fine-tuning can happen:

- **Data loading** — Downloads two Hemingway books as raw text; splits into training (*Men Without Women*) and evaluation (*The Sun Also Rises*) sets
- **Tokenization** — Loads the Phi-2 tokenizer and explores truncation, padding, and overflow strategies; uses a sliding window approach (`return_overflowing_tokens=True`) to convert full books into 250-token training chunks
- **Causal language modelling setup** — Explains why `labels = input_ids` and how the training framework handles the one-position shift automatically
- **Quantization** — Configures `BitsAndBytesConfig` for optional 4-bit quantization; explores `fp4` vs `nf4` formats and the memory vs. perplexity trade-off
- **Model loading** — Loads Phi-2 from Hugging Face with a pinned revision for reproducibility; maps the full model to a single GPU

---

### Notebook 2 — PEFT with LoRA and QLoRA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Wendyshiro/LLM-Fine-tuning/blob/main/PEFT_with_LoRA_and_QLoRA.ipynb)

Builds on Notebook 1 to configure and run the actual fine-tuning:

- **Base model sampling** — Generates text from the untuned Phi-2 model as a baseline for comparison
- **Model inspection** — Prints the full model architecture to identify which modules to target with LoRA
- **LoRA configuration** — Sets up `LoraConfig` with rank (`r=32`), scaling (`lora_alpha=64`), dropout, and target modules (`fc1`, `fc2`, `q_proj`, `k_proj`, `v_proj`, `dense`)
- **Trainable parameter analysis** — Compares trainable vs. total parameters to demonstrate PEFT efficiency
- **Training** — Runs the fine-tuning loop using `Trainer` with cosine learning rate scheduling, gradient checkpointing, and gradient clipping
- **Post-training sampling** — Compares model output before and after fine-tuning using the same prompts to measure stylistic change

---

## 🔑 Key Concepts

**LoRA (Low-Rank Adaptation)** — Instead of updating all 2.7B parameters of Phi-2 during fine-tuning, LoRA injects small trainable low-rank matrices into specific layers. This reduces the number of trainable parameters dramatically, saving GPU memory and training time while preserving most of the model's original knowledge.

**QLoRA** — Combines quantization with LoRA: the base model is loaded in 4-bit precision (reducing memory by ~4×), while LoRA adapters are trained in 16-bit. This makes it possible to fine-tune large models on consumer GPUs.

**Causal Language Modelling** — The model learns to predict the next token given all previous tokens. For a sequence `[A, B, C, D]`, it learns to predict `B` from `A`, `C` from `A,B`, and so on. No labelled data is required — the text itself is both input and target.

**Perplexity** — A measure of how confidently a model predicts text. Lower is better. Used in this project to evaluate whether quantization meaningfully degrades model quality.

**Gradient Checkpointing** — A memory optimisation that trades a small amount of compute speed for significantly lower GPU memory usage during training — essential for fitting fine-tuning within a T4 GPU's 16 GB VRAM.

---

## ⚙️ Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `lora_r` | 32 | LoRA rank — higher = more flexible but more memory |
| `lora_alpha` | 64 | LoRA scaling factor (rule of thumb: `2 × lora_r`) |
| `lora_dropout` | 0.1 | Prevents over-reliance on any single weight |
| `context_length` | 250 | Tokens per training chunk |
| `num_train_epochs` | 3 | Full passes over the training dataset |
| `learning_rate` | 1e-4 | Starting learning rate for AdamW optimizer |
| `lr_scheduler_type` | cosine | Gradually reduces LR towards end of training |
| `per_device_train_batch_size` | 2 | Limited by T4 VRAM |
| `gradient_accumulation_steps` | 2 | Effective batch size = 2 × 2 = 4 |
| `use_4bit` | False (default) | Set to `True` to enable QLoRA |

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `torch` | 2.5.1 | Deep learning engine (CUDA 12.1) |
| `transformers` | 4.56.2 | Phi-2 model & tokenizer |
| `datasets` | 4.0.0 | Data loading and preprocessing |
| `peft` | 0.17.1 | LoRA / QLoRA fine-tuning |
| `bitsandbytes` | 0.47.0 | 4-bit quantization |
| `accelerate` | 1.10.1 | GPU training utilities |

---

## 🚀 Getting Started

### Requirements
- Google Colab with a **T4 GPU** runtime (`Runtime → Change runtime type → T4 GPU`)
- No local setup needed — all dependencies are installed within the notebooks

### Run Order
Run the notebooks **in order**:
1. `Data_Preparation_and_Model_Loading.ipynb`
2. `PEFT_with_LoRA_and_QLoRA.ipynb`

> Notebook 2 re-runs the setup from Notebook 1 in its first cells, so you can also run it standalone.

### Enabling QLoRA
To switch from LoRA to QLoRA, simply change this line in Notebook 2:
```python
use_4bit = False  # Change to True for QLoRA
```
This loads the model in 4-bit precision and enables `fp16` training automatically.

---

## 📚 References & Credits

- Original notebook by [@maximelabonne](https://github.com/mlabonne/llm-course/blob/main/Fine_tune_Llama_2_in_Google_Colab.ipynb)
- Based on [Younes Belkada's GitHub Gist](https://gist.github.com/younesbelkada/9f7f75c94bdc1981c8ca5cc937d4a4da)
- Phi-2 fine-tuning reference from [Brev.dev notebooks](https://github.com/brevdev/notebooks/blob/main/phi2-finetune-own-data.ipynb)
- [Microsoft Phi-2 on Hugging Face](https://huggingface.co/microsoft/phi-2)
- Course: [Quantic LLM Fine-Tuning](https://github.com/quanticedu/llm-fine-tuning)

---

## 📝 License

This project is for educational purposes as part of the Quantic LLM Fine-Tuning course.