# Lightweight Fine-Tuning with GPT-2

This project demonstrates parameter-efficient fine-tuning of GPT-2 using LoRA (Low-Rank Adaptation) and QLoRA for financial news topic classification.

## Project Overview

This notebook implements and compares multiple approaches to fine-tune GPT-2 on a financial news topic classification task:

- **Base Model**: GPT-2 (openai-community/gpt2)
- **PEFT Technique**: LoRA (Low-Rank Adaptation)
- **Dataset**: [Twitter Financial News Topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic)
- **Task**: Multi-class text classification (20 categories)
- **Evaluation Metrics**: Accuracy, Precision (weighted), Recall (weighted), F1-Score (weighted)

## Dataset

The project uses the Twitter Financial News Topic dataset from Hugging Face with 20 different financial news categories:

0. Analyst Update
1. Fed | Central Banks
2. Company | Product News
3. Treasuries | Corporate Debt
4. Dividend
5. Earnings
6. Energy | Oil
7. Financials
8. Currencies
9. General News | Opinion
10. Gold | Metals | Materials
11. IPO
12. Legal | Regulation
13. M&A | Investments
14. Macro
15. Markets
16. Politics
17. Personnel Change
18. Stock Commentary
19. Stock Movement

- **Training Set**: 16,990 samples
- **Validation Set**: 4,117 samples

## Models Trained

The project trains and evaluates four different model variants:

### 1. **GPT-2 (Baseline)**
- Pre-trained GPT-2 without fine-tuning
- **Performance**: ~0.78% accuracy (poor baseline)

### 2. **GPT-2 + LoRA**
- GPT-2 fine-tuned with LoRA adaptation
- **Trainable Parameters**: ~4.8M out of 129.2M (3.7%)
- **Performance**: ~89.1% accuracy
- **Training Time**: ~945 seconds for 2 epochs

#### LoRA Configuration:
- `r` (rank): 128
- `lora_alpha` (scaling factor): 128
- `lora_dropout`: 0.05
- `target_modules`: Query and Value attention layers
- `modules_to_save`: Score layer

### 3. **GPT-2 Merged LoRA**
- LoRA weights merged with the base model
- **Performance**: ~89.1% accuracy (same as GPT-2 + LoRA)
- **Advantage**: Single model file, no separate PEFT weights needed

### 4. **GPT-2 QLoRA**
- GPT-2 quantized to 4-bit + LoRA fine-tuning
- **Quantization Config**:
  - 4-bit quantization with NF4 data type
  - Double quantization enabled
  - Compute dtype: bfloat16
- **Trainable Parameters**: ~2.4M
- **Performance**: ~1.77% accuracy (underperformed in this case)
- **Note**: Better suited for GPU memory constraints

## Project Structure

```
fine_tuning_gpt2/
├── README.md
└── notebooks/
    └── LightweightFineTuning.ipynb  # Main implementation notebook
```

## Key Implementation Steps

### 1. Data Loading and Preprocessing
- Load dataset from Hugging Face
- Tokenize text with GPT-2 tokenizer
- Pad sequences (GPT-2 has 1024 token context window)
- Truncate sequences exceeding max length

### 2. Model Loading
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "openai-community/gpt2",
    num_labels=20,
    id2label=ID2LABEL_DICT,
    label2id=LABEL2ID_DICT
)
```

### 3. LoRA Adaptation
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="lora_only",
    modules_to_save=["score"],
)

peft_model = get_peft_model(model, lora_config)
```

### 4. Training
- **Optimizer Learning Rate**: 5e-4
- **Batch Size**: 4 (per device)
- **Epochs**: 2
- **Weight Decay**: 0.01
- **Evaluation Strategy**: Per epoch
- **Save Strategy**: Per epoch (load best model at end)

### 5. Evaluation
Models are evaluated using:
- **Accuracy**: Overall correct predictions
- **Precision (Weighted)**: Per-class precision averaged by support
- **Recall (Weighted)**: Per-class recall averaged by support
- **F1-Score (Weighted)**: Harmonic mean of precision and recall

## Results Summary

| Model | Accuracy | Precision (W) | Recall (W) | F1-Score (W) |
|-------|----------|---------------|-----------|--------------|
| GPT-2 Baseline | 0.78% | 0.006% | 0.78% | 0.012% |
| **GPT-2 + LoRA** | **89.07%** | **89.10%** | **89.07%** | **89.05%** |
| **GPT-2 Merged LoRA** | **89.07%** | **89.10%** | **89.07%** | **89.05%** |
| GPT-2 QLoRA | 1.77% | 0.031% | 1.77% | 0.062% |

**Key Findings**:
- LoRA fine-tuning dramatically improves model performance (~114x improvement)
- Only 3.7% of parameters need training with LoRA
- Merged LoRA model achieves same performance but with simpler deployment
- QLoRA (quantized) struggled with this task configuration

## Requirements

- torch
- transformers
- datasets
- peft
- scikit-learn
- pandas
- numpy
- bitsandbytes (for QLoRA)

## Usage

To run the notebook:

```bash
jupyter notebook notebooks/LightweightFineTuning.ipynb
```

## Key Technical Notes

### Why LoRA?
- Reduces trainable parameters from 129M to 4.8M (96.3% reduction)
- Maintains model quality with minimal performance trade-off
- Speeds up training significantly
- Enables fine-tuning on memory-constrained devices

### Tokenization Details
- GPT-2 uses byte-pair encoding with 50,257 tokens
- Context window limited to 1,024 tokens
- Padding applied on the right side
- EOS token used as padding token

### Why QLoRA Underperformed
- 4-bit quantization may have lost important information for this task
- Model was already memory-efficient with LoRA alone
- Different hyperparameters might be needed for better QLoRA performance

## References

- [PEFT Library Documentation](https://huggingface.co/docs/peft/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [Twitter Financial News Topic Dataset](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic)
