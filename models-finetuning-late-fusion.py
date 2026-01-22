import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random
import os


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = (
     torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Using device:", device)

#load frozen data splits
with open("afd_text_audio_splits_frozen.json") as f:
    frozen_splits = json.load(f)

train_data = frozen_splits["train"]
val_data   = frozen_splits["val"]
test_data  = frozen_splits["test"]

train_texts = [x["text"] for x in train_data]
train_audios = [x["audio"] for x in train_data]
train_labels = [x["label"] for x in train_data]

val_texts = [x["text"] for x in val_data]
val_audios = [x["audio"] for x in val_data]
val_labels = [x["label"] for x in val_data]

test_texts = [x["text"] for x in test_data]
test_audios = [x["audio"] for x in test_data]
test_labels = [x["label"] for x in test_data]

#compute class weights
classes = np.unique(train_labels)
class_weights = compute_class_weight("balanced", classes=classes, y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class weights:", class_weights)

#focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1-pt)**self.gamma) * ce
        return loss.mean() if self.reduction=="mean" else loss.sum()


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ntu-spml/distilhubert")
MAX_AUDIO_LEN = 160000  # 10 sec approx

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

class AudioDataset(Dataset):
    def __init__(self, paths, labels, extractor, max_len=MAX_AUDIO_LEN):
        self.paths = paths
        self.labels = labels
        self.extractor = extractor
        self.max_len = max_len
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        import torchaudio
        path = self.paths[idx]
        try:
            wav, sr = torchaudio.load(path)
        except:
            wav = torch.zeros(self.max_len)
            sr = 16000
        if wav.numel() == 0: wav = torch.zeros(self.max_len)
        if sr != 16000: wav = torchaudio.transforms.Resample(sr,16000)(wav)
        wav = wav.squeeze(0)
        max_val = wav.abs().max()
        if max_val>0: wav = wav/max_val
        if wav.size(0)<self.max_len:
            wav = F.pad(wav,(0,self.max_len - wav.size(0)))
        else:
            wav = wav[:self.max_len]
        feats = self.extractor(wav, sampling_rate=16000, return_tensors="pt").input_values.squeeze(0)
        return feats, torch.tensor(self.labels[idx], dtype=torch.long)

train_text_ds = TextDataset(train_texts, train_labels, tokenizer)
val_text_ds   = TextDataset(val_texts, val_labels, tokenizer)
test_text_ds  = TextDataset(test_texts, test_labels, tokenizer)

train_audio_ds = AudioDataset(train_audios, train_labels, audio_extractor)
val_audio_ds   = AudioDataset(val_audios, val_labels, audio_extractor)
test_audio_ds  = AudioDataset(test_audios, test_labels, audio_extractor)

#roBERTa fine-tuning
text_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
optimizer_text = torch.optim.AdamW(text_model.parameters(), lr=2e-5)
criterion_text = FocalLoss(alpha=class_weights)
train_loader_text = DataLoader(train_text_ds, batch_size=16, shuffle=True)
val_loader_text   = DataLoader(val_text_ds, batch_size=16)

# Training loop
print(" Fine-tuning RoBERTa...")
for epoch in range(3):
    text_model.train()
    total_loss = 0
    for batch in train_loader_text:
        optimizer_text.zero_grad()
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = text_model(input_ids, attention_mask=mask).logits
        loss = criterion_text(logits, labels)
        loss.backward()
        optimizer_text.step()
        total_loss += loss.item()
    print(f"[Text] Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader_text):.4f}")

# Evaluate
text_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in val_loader_text:
        input_ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = text_model(input_ids, attention_mask=mask).logits
        all_preds.extend(torch.argmax(logits,1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
f1_text = f1_score(all_labels, all_preds, average="binary")
print(f" RoBERTa F1 on val: {f1_text:.4f}")

torch.save(text_model.state_dict(), "./roberta_finetuned.pt")
text_model.eval()

#distilHuBERT fine-tuning
audio_model = Wav2Vec2Model.from_pretrained("ntu-spml/distilhubert").to(device)
for p in audio_model.parameters(): p.requires_grad = False
audio_classifier = nn.Linear(audio_model.config.hidden_size,2).to(device)
optimizer_audio = torch.optim.AdamW(audio_classifier.parameters(), lr=1e-4)
criterion_audio = FocalLoss(alpha=class_weights)
train_loader_audio = DataLoader(train_audio_ds, batch_size=4, shuffle=True)
val_loader_audio   = DataLoader(val_audio_ds, batch_size=4)

print("Fine-tuning DistilHuBERT classifier...")
for epoch in range(3):
    total_loss = 0
    audio_classifier.train()
    for feats, labels in train_loader_audio:
        feats, labels = feats.to(device), labels.to(device)
        optimizer_audio.zero_grad()
        with torch.no_grad():
            emb = audio_model(feats).last_hidden_state.mean(dim=1)
        logits = audio_classifier(emb)
        loss = criterion_audio(logits, labels)
        loss.backward()
        optimizer_audio.step()
        total_loss += loss.item()
    print(f"[Audio] Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader_audio):.4f}")

# Evaluate
audio_classifier.eval()
audio_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for feats, labels in val_loader_audio:
        feats, labels = feats.to(device), labels.to(device)
        emb = audio_model(feats).last_hidden_state.mean(dim=1)
        logits = audio_classifier(emb)
        all_preds.extend(torch.argmax(logits,1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
f1_audio = f1_score(all_labels, all_preds, average="binary")
print(f" DistilHuBERT F1 on val: {f1_audio:.4f}")

torch.save(audio_classifier.state_dict(), "./distilhubert_classifier.pt")

#compute embeddings for fusion
def compute_text_emb(model, dataset, batch_size=16):
    model.eval()
    embs=[]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            out = model.roberta(input_ids, mask).last_hidden_state[:,0,:]
            embs.append(out.cpu())
    return torch.cat(embs)

def compute_audio_emb(model, classifier, dataset, batch_size=4):
    model.eval(); classifier.eval()
    embs=[]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for feats, _ in loader:
            feats = feats.to(device)
            out = model(feats).last_hidden_state.mean(dim=1)
            embs.append(out.cpu())
    return torch.cat(embs)

train_text_emb = compute_text_emb(text_model, train_text_ds)
val_text_emb   = compute_text_emb(text_model, val_text_ds)
test_text_emb  = compute_text_emb(text_model, test_text_ds)

train_audio_emb = compute_audio_emb(audio_model, audio_classifier, train_audio_ds)
val_audio_emb   = compute_audio_emb(audio_model, audio_classifier, val_audio_ds)
test_audio_emb  = compute_audio_emb(audio_model, audio_classifier, test_audio_ds)

# Save embeddings
os.makedirs("embeddings", exist_ok=True)
torch.save(train_text_emb,"embeddings/train_text.pt")
torch.save(val_text_emb,"embeddings/val_text.pt")
torch.save(test_text_emb,"embeddings/test_text.pt")
torch.save(train_audio_emb,"embeddings/train_audio.pt")
torch.save(val_audio_emb,"embeddings/val_audio.pt")
torch.save(test_audio_emb,"embeddings/test_audio.pt")
torch.save(torch.tensor(train_labels),"embeddings/train_labels.pt")
torch.save(torch.tensor(val_labels),"embeddings/val_labels.pt")
torch.save(torch.tensor(test_labels),"embeddings/test_labels.pt")
print(" Embeddings saved.")

#late fusion training
class FusionClassifier(nn.Module):
    def __init__(self, dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
    def forward(self,x): return self.fc(x)

train_emb = torch.cat([train_text_emb, train_audio_emb], dim=1)
val_emb   = torch.cat([val_text_emb, val_audio_emb], dim=1)
test_emb  = torch.cat([test_text_emb, test_audio_emb], dim=1)

train_labels_t = torch.tensor(train_labels)
test_labels_t  = torch.tensor(test_labels)

fusion_model = FusionClassifier(train_emb.size(1)).to(device)
optimizer_f = torch.optim.AdamW(fusion_model.parameters(), lr=1e-3)
criterion_f = FocalLoss(alpha=class_weights)

train_loader_f = DataLoader(torch.utils.data.TensorDataset(train_emb, train_labels_t), batch_size=16, shuffle=True)

print(" Training late fusion classifier...")
for epoch in range(100):
    total_loss=0
    fusion_model.train()
    for x,y in train_loader_f:
        x,y = x.to(device), y.to(device)
        optimizer_f.zero_grad()
        loss = criterion_f(fusion_model(x),y)
        loss.backward()
        optimizer_f.step()
        total_loss+=loss.item()
    print(f"[Fusion] Epoch {epoch+1} | Avg Loss: {total_loss/len(train_loader_f):.4f}")

torch.save(fusion_model.state_dict(),"fusion_classifier.pt")
print("Fusion classifier saved.")

# Evaluate fusion
fusion_model.eval()
preds=[]
with torch.no_grad():
    for x,y in DataLoader(torch.utils.data.TensorDataset(test_emb, test_labels_t), batch_size=16):
        logits = fusion_model(x.to(device))
        preds.extend(torch.argmax(logits,1).cpu().numpy())
f1_final = f1_score(test_labels_t.numpy(), preds, average="binary")
print(f"\n FINAL Late Fusion F1 (binary): {f1_final:.4f}")
