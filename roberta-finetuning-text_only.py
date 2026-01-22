import json, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)


with open("afd_text_splits_frozen.json") as f:
    data = json.load(f)

train_texts  = [x["text"] for x in data["train"]]
train_labels = [x["label"] for x in data["train"]]
val_texts    = [x["text"] for x in data["val"]]
val_labels   = [x["label"] for x in data["val"]]
test_texts   = [x["text"] for x in data["test"]]
test_labels  = [x["label"] for x in data["test"]]

print(f"Train: {len(train_labels)} | Val: {len(val_labels)} | Test: {len(test_labels)}")
print("Test positive rate:", np.mean(test_labels))

#dataset and dataloader
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(texts, truncation=True, padding="max_length", max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = TextDataset(train_texts, train_labels)
val_ds   = TextDataset(val_texts, val_labels)
test_ds  = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False)

#model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

# Freeze all layers except last 3 + classifier
for n, p in model.named_parameters():
    p.requires_grad = ("encoder.layer.9" in n or 
                       "encoder.layer.10" in n or 
                       "encoder.layer.11" in n or 
                       "classifier" in n)

#weighted loss
classes = np.unique(train_labels)
weights = compute_class_weight("balanced", classes=classes, y=train_labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean() if self.reduction=="mean" else loss.sum()

criterion = FocalLoss(alpha=class_weights)


optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
num_training_steps = len(train_loader) * 8  # 8 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)
clip_value = 1.0

#training loop with early stopping
best_f1, patience, wait = 0.0, 4, 0
num_epochs = 5
print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(ids, attention_mask=mask).logits
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    # ---- validation ----
    model.eval()
    val_preds, val_labels_eval = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(ids, attention_mask=mask).logits
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels_eval.extend(labels.cpu().numpy())

    val_f1 = f1_score(val_labels_eval, val_preds, average="binary")
    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Binary F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        wait = 0
        model.save_pretrained("./roberta_text_only_finetuned")
        tokenizer.save_pretrained("./roberta_text_only_finetuned")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping")
            break

#test evaluation
model = RobertaForSequenceClassification.from_pretrained("./roberta_text_only_finetuned").to(device)# gave f1 0.24

model.eval()

test_preds, test_labels_eval = [], []
with torch.no_grad():
    for batch in test_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(ids, attention_mask=mask).logits
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels_eval.extend(labels.cpu().numpy())

print("\n RoBERTa Test Binary F1:", f1_score(test_labels_eval, test_preds))
print(classification_report(test_labels_eval, test_preds, digits=4))

# -------------------------
# 8. Sample predictions
# -------------------------
print("\nSome test examples (text -> true / pred):")
for i in range(min(5, len(test_texts))):
    print(f'"{test_texts[i]}" -> {test_labels[i]} / {test_preds[i]}')
