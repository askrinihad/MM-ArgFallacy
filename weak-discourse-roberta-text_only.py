import os, json, random, numpy as np, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Device:", device)


with open("afd_text_splits_frozen.json") as f:
    data = json.load(f)


#  Weak discourse markers

ATTACK = ["but", "however", "actually", "that's false", "you are wrong"]
JUSTIFY = ["because", "since", "therefore", "thus"]
GENERALIZE = ["everyone", "always", "never", "all"]

def discourse_tag(text):
    t = text.lower()
    tags = []
    if any(w in t for w in ATTACK):
        tags.append("[ATTACK]")
    if any(w in t for w in JUSTIFY):
        tags.append("[JUSTIFY]")
    if any(w in t for w in GENERALIZE):
        tags.append("[GENERALIZE]")
    return " ".join(tags) + " " + text

def build_text_with_tags(split):
    texts, labels = [], []
    for i, ex in enumerate(split):
        cur = discourse_tag(ex["text"])
        combined = f"[ROLE] Utterance {i+1} [UTT] {cur}"
        texts.append(combined)
        labels.append(int(ex["label"]))
    return texts, labels

train_texts, train_labels = build_text_with_tags(data["train"])
val_texts, val_labels     = build_text_with_tags(data["val"])
test_texts, test_labels   = build_text_with_tags(data["test"])

print("Train size:", len(train_labels), "Positive rate:", np.mean(train_labels))


#  Dataset

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
MAX_LEN = 256

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(texts, truncation=True, padding="max_length", max_length=MAX_LEN)
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
val_loader   = DataLoader(val_ds, batch_size=32)
test_loader  = DataLoader(test_ds, batch_size=32)

#model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)

# Unfreeze last 4 layers + classifier
for name, p in model.named_parameters():
    p.requires_grad = any(x in name for x in ["encoder.layer.8","encoder.layer.9","encoder.layer.10","encoder.layer.11","classifier"])

# Class weights for imbalance
classes = np.unique(train_labels)
weights = compute_class_weight("balanced", classes=classes, y=train_labels)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1.5e-5, weight_decay=0.01)

num_epochs = 12
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)

# -----------------------------
#  Training with threshold tuning

best_val_f1 = 0
patience, wait = 3, 0
save_dir = "./roberta_fallacy_best"
os.makedirs(save_dir, exist_ok=True)

print("\n Training pure text + weak discourse signals")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
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

    # Validation
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask=attention_mask).logits
            val_probs.extend(torch.softmax(logits, dim=1)[:,1].cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    # Threshold tuning
    best_t, best_f1_epoch = 0.5, 0
    for t in np.linspace(0.2,0.8,25):
        preds = (np.array(val_probs)>=t).astype(int)
        f1 = f1_score(val_true, preds)
        if f1>best_f1_epoch:
            best_f1_epoch = f1
            best_t = t

    print(f"Epoch {epoch+1} | Loss {total_loss/len(train_loader):.4f} | Val F1 {best_f1_epoch:.4f} @ t={best_t:.2f}")

    if best_f1_epoch > best_val_f1:
        best_val_f1 = best_f1_epoch
        wait = 0
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        np.save(os.path.join(save_dir,"threshold.npy"), best_t)
        print(" Saved best model")
    else:
        wait += 1
        if wait>=patience:
            print("⏹ Early stopping")
            break

# -----------------------------
#  Test evaluation

model = RobertaForSequenceClassification.from_pretrained(save_dir).to(device)
threshold = float(np.load(os.path.join(save_dir,"threshold.npy")))
model.eval()

test_probs, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids, attention_mask=attention_mask).logits
        test_probs.extend(torch.softmax(logits, dim=1)[:,1].cpu().numpy())
        test_true.extend(labels.cpu().numpy())

test_preds = (np.array(test_probs)>=threshold).astype(int)

print("\n FINAL TEST F1:", f1_score(test_true,test_preds))
print(classification_report(test_true,test_preds,digits=4))
print("Best Val F1:", best_val_f1)
print("Threshold:", threshold)
