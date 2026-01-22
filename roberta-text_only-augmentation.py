

import os, json, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Using device:", device)


with open("afd_text_splits_frozen.json") as f:
    data = json.load(f)

train_texts  = [x["text"] for x in data["train"]]
train_labels = [int(x["label"]) for x in data["train"]]
val_texts    = [x["text"] for x in data["val"]]
val_labels   = [int(x["label"]) for x in data["val"]]
test_texts   = [x["text"] for x in data["test"]]
test_labels  = [int(x["label"]) for x in data["test"]]

print(f"Train: {len(train_labels)} | Val: {len(val_labels)} | Test: {len(test_labels)}")
print("Train positive rate:", float(np.mean(train_labels)))
print("Val positive rate:", float(np.mean(val_labels)))
print("Test positive rate:", float(np.mean(test_labels)))



PARA_MODEL_NAME = "Vamsi/T5_Paraphrase_Paws"

para_tok = AutoTokenizer.from_pretrained(PARA_MODEL_NAME)
para_model = AutoModelForSeq2SeqLM.from_pretrained(PARA_MODEL_NAME).to(device)
para_model.eval()

@torch.no_grad()
def paraphrase_t5(text: str, n_return: int = 2, max_len: int = 128) -> list[str]:
    """
    Generate paraphrases using T5 paraphrase model.
    We keep output short-ish to reduce label drift.
    """
    prompt = f"paraphrase: {text} </s>"
    enc = para_tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_len
    ).to(device)

    outs = para_model.generate(
        **enc,
        num_beams=6,
        num_return_sequences=n_return,
        temperature=1.2,
        max_length=max_len,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    paras = []
    for o in outs:
        s = para_tok.decode(o, skip_special_tokens=True).strip()
        if s and s.lower() != text.lower():
            paras.append(s)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for p in paras:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


AUG_PER_POS = 2            
MAX_AUG_TOTAL = 20000       
MINORITY_LABEL = 1

aug_texts, aug_labels = [], []
pos_count = sum(train_labels)
print("Positive examples in train:", pos_count)

# Augment minority class only
for t, y in zip(train_texts, train_labels):
    if y == MINORITY_LABEL:
        paras = paraphrase_t5(t, n_return=AUG_PER_POS, max_len=128)
        for p in paras:
            aug_texts.append(p)
            aug_labels.append(MINORITY_LABEL)
            if len(aug_texts) >= MAX_AUG_TOTAL:
                break
    if len(aug_texts) >= MAX_AUG_TOTAL:
        break

train_texts_aug = train_texts + aug_texts
train_labels_aug = train_labels + aug_labels

print(f"Added {len(aug_texts)} paraphrases.")
print(f"Augmented train size: {len(train_labels_aug)} | New positive rate: {float(np.mean(train_labels_aug)):.4f}")


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
MAX_LEN = 256  

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = TextDataset(train_texts_aug, train_labels_aug)
val_ds   = TextDataset(val_texts, val_labels)
test_ds  = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)


model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

# Practical choice: unfreeze last 6 layers + classifier (helps stability on small data)
for n, p in model.named_parameters():
    p.requires_grad = (
        "encoder.layer.6" in n or
        "encoder.layer.7" in n or
        "encoder.layer.8" in n or
        "encoder.layer.9" in n or
        "encoder.layer.10" in n or
        "encoder.layer.11" in n or
        "classifier" in n
    )

# weighted cross-entropy
classes = np.unique(train_labels_aug)
weights = compute_class_weight("balanced", classes=classes, y=train_labels_aug)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)


optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5, weight_decay=0.01)

num_epochs = 10
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

#training loop with early stopping
save_dir = "./roberta_text_only_aug_best"
os.makedirs(save_dir, exist_ok=True)

best_f1 = 0.0
patience = 3
wait = 0

print("\n Training RoBERTa text-only with T5 paraphrase augmentation")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    # ---- validation ----
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_true, val_preds, average="binary")
    print(f"Epoch {epoch+1}/{num_epochs} | Loss {total_loss/len(train_loader):.4f} | Val F1 {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        wait = 0
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("💾 Saved best model.")
    else:
        wait += 1
        if wait >= patience:
            print("⏹ Early stopping")
            break

#test evaluation
best_model = RobertaForSequenceClassification.from_pretrained(save_dir).to(device)
best_model.eval()

test_preds, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = best_model(input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(logits, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_true.extend(labels.cpu().numpy())

print("\n FINAL TEST BINARY F1:", f1_score(test_true, test_preds, average="binary"))
print(classification_report(test_true, test_preds, digits=4))
print("Best Val F1:", best_f1)
print("Saved at:", save_dir)
