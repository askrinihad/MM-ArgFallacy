

import json, random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score, classification_report


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


with open("afd_text_splits_frozen.json") as f:
    data = json.load(f)

test_texts  = [x["text"] for x in data["test"]]
test_labels = [x["label"] for x in data["test"]]

print(f"Test samples: {len(test_labels)}")
print(f"Positive rate: {np.mean(test_labels):.4f}")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=128
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


MODEL_DIR = "./roberta_text_only_finetuned"
#MODEL_DIR = "./saved_models_balanced/roberta_finetuned"
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()


test_loader = DataLoader(
    TextDataset(test_texts, test_labels, tokenizer),
    batch_size=32,
    shuffle=False
)


preds, gold = [], []

with torch.no_grad():
    for batch in test_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        logits = model(ids, attention_mask=mask).logits
        batch_preds = torch.argmax(logits, dim=1)

        preds.extend(batch_preds.cpu().numpy())
        gold.extend(labels.numpy())

preds = np.array(preds)
gold  = np.array(gold)

#metrics
print("\ TEST RESULTS")
print("Binary F1:", f1_score(gold, preds, average="binary"))
print("Macro  F1:", f1_score(gold, preds, average="macro"))

print("\nClassification report:")
print(classification_report(gold, preds, digits=4))

print("\nPrediction distribution:")
print("Pred 0:", np.sum(preds == 0))
print("Pred 1:", np.sum(preds == 1))
