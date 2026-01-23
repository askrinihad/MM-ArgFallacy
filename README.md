# MM-ArgFallacy — Multimodal Argumentative Fallacy Detection (Text + Audio)

This repository presents **Argumentative Fallacy Detection (AFD)** using the **MM-USED-fallacy** dataset from the **MAMKit** library:
- MAMKit: https://github.com/nlp-unibo/mamkit  
- This project repo: https://github.com/askrinihad/MM-ArgFallacy

We explore:
- **Text-only** fallacy detection
- **Text + Audio** fallacy detection using multiple multimodal fusion strategies

---

##  Implemented Methods

### 1) Text-only baseline
A text-only classifier based on RoBERTa using the MM-USED-fallacy dataset.

### 2) Text + Audio (multimodal) approaches
We implement and compare three multimodal strategies:

#### A) Context-aware multimodal classifier (Late/simple concatenation)
- Simple concatenation of **text** and **audio** embeddings  
- Incorporates **textual context** from previous turns in the conversation

#### B) Transformer Gated Fusion
- Fusion of text and audio embeddings using:
  - **Cross-attention** mechanism  
  - Followed by **feature-wise gating**

#### C) Jointly Gated Fusion
- **Partial fine-tuning** of text and audio encoders  
- **Feature-wise gated fusion**, trained jointly with the classifier

---

##  Prerequisites

Before installing requirements, ensure you have:

- **Python 3.10** (MAMKit is tested with this version)
- Git

---

## Installation

```bash
git clone https://github.com/askrinihad/MM-ArgFallacy.git
cd MM-ArgFallacy
pip install -r requirements.txt
```

## Using fine-tuned models

If you want to test the fine-tuned models, you can download them from the **GitHub Releases** page. The available models include:

- `roberta-text_only-finetuned` — **Text-only** fine-tuned model  
- `cross_attn_gated_fusion_best.pt` — **Transformer Gated Fusion**  
- `late-fusion_classifier.pt` — **Late fusion**

> After downloading, place the model files in the expected checkpoint directory used by your scripts (see your training/testing scripts for the exact path).

---

##  Dataset / Framework

This work relies on **MM-USED-fallacy** through **MAMKit**:  
https://github.com/nlp-unibo/mamkit



