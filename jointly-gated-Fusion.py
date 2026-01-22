
import json, random, torch, torchaudio, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, RobertaModel, Wav2Vec2Model, Wav2Vec2FeatureExtractor, AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

#load data splits
with open("afd_text_audio_splits_frozen.json") as f:
    data = json.load(f)

def unpack(split):
    return [x["text"] for x in split], [x["audio"] for x in split], [x["label"] for x in split]

train_texts, train_audios, train_labels = unpack(data["train"])
val_texts, val_audios, val_labels = unpack(data["val"])
test_texts, test_audios, test_labels = unpack(data["test"])

train_labels_np = np.array(train_labels)
class_counts = np.bincount(train_labels_np)
print("Train positive rate:", class_counts[1]/len(train_labels))


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
audio_extractor = Wav2Vec2FeatureExtractor.from_pretrained("ntu-spml/distilhubert")
MAX_AUDIO_LEN = 160000  # ~10 sec


class MultimodalDataset(Dataset):
    def __init__(self, texts, audios, labels):
        self.texts = texts
        self.audios = audios
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # --- Text ---
        enc = tokenizer(
            self.texts[idx], padding="max_length", truncation=True, max_length=128, return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # --- Audio ---
        path = self.audios[idx]
        try:
            wav, sr = torchaudio.load(path)
        except:
            wav = torch.zeros(MAX_AUDIO_LEN)
            sr = 16000

        if wav.numel() == 0:
            wav = torch.zeros(MAX_AUDIO_LEN)
            sr = 16000

        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        wav = wav.squeeze(0)
        if wav.size(0) < MAX_AUDIO_LEN:
            wav = F.pad(wav, (0, MAX_AUDIO_LEN - wav.size(0)))
        else:
            wav = wav[:MAX_AUDIO_LEN]

        feats = audio_extractor(wav, sampling_rate=16000, return_tensors="pt").input_values.squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "audio": feats, "labels": label}

train_ds = MultimodalDataset(train_texts, train_audios, train_labels)
val_ds = MultimodalDataset(val_texts, val_audios, val_labels)
test_ds = MultimodalDataset(test_texts, test_audios, test_labels)

# Weighted sampler for imbalance
class_weights_np = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels_np)
weights = 1.0 / class_counts
sample_weights = weights[train_labels_np]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

#models
text_model = RobertaModel.from_pretrained("roberta-base").to(device)
audio_model = Wav2Vec2Model.from_pretrained("ntu-spml/distilhubert").to(device)

# Unfreeze last 2 audio layers
for n, p in audio_model.named_parameters():
    p.requires_grad = ("encoder.layers.10" in n or "encoder.layers.11" in n)

#gated fusion model
class GatedFusion(nn.Module):
    def __init__(self, dim_text, dim_audio, hidden=256, dropout=0.2):
        super().__init__()
        self.text_fc  = nn.Linear(dim_text, hidden)
        self.audio_fc = nn.Linear(dim_audio, hidden)

        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, 2)
        )

    def forward(self, t, a):
        t_h = F.relu(self.text_fc(t))
        a_h = F.relu(self.audio_fc(a))
        g = self.gate(torch.cat([t_h, a_h], dim=1))
        fused = t_h + g * a_h
        logits = self.classifier(fused)
        return logits

fusion_model = GatedFusion(text_model.config.hidden_size, audio_model.config.hidden_size).to(device)

#focal loss with class weights
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float).to(device)
criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)

#optimizer
optimizer = AdamW([
    {"params": fusion_model.parameters(), "lr": 1e-3},
    {"params": text_model.parameters(), "lr": 2e-5},
    {"params": [p for p in audio_model.parameters() if p.requires_grad], "lr": 1e-5},
], weight_decay=1e-2)

#training loop
best_f1 = 0.0
patience = 5
wait = 0
num_epochs = 10

for epoch in range(num_epochs):
    fusion_model.train()
    text_model.train()
    audio_model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        labels = batch["labels"].to(device)

        t_emb = text_model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).last_hidden_state[:,0,:]
        a_emb = audio_model(batch["audio"].to(device)).last_hidden_state.mean(dim=1)

        logits = fusion_model(t_emb, a_emb)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # Validation
    fusion_model.eval()
    text_model.eval()
    audio_model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].to(device)
            t_emb = text_model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).last_hidden_state[:,0,:]
            a_emb = audio_model(batch["audio"].to(device)).last_hidden_state.mean(dim=1)
            logits = fusion_model(t_emb, a_emb)
            val_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    val_f1 = f1_score(val_true, val_preds)
    print(f"Epoch {epoch+1} | Loss {total_loss/len(train_loader):.4f} | Val F1 {val_f1:.4f}")

    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        wait = 0
        torch.save({
            "fusion": fusion_model.state_dict(),
            "text": text_model.state_dict(),
            "audio": audio_model.state_dict(),
            "tokenizer": tokenizer
        }, "./best_gated_fusion.pth")
    else:
        wait += 1
        if wait >= patience:
            print("⏹ Early stopping")
            break

#evaluation on test set
checkpoint = torch.load("./best_gated_fusion.pth")
fusion_model.load_state_dict(checkpoint["fusion"])
text_model.load_state_dict(checkpoint["text"])
audio_model.load_state_dict(checkpoint["audio"])
fusion_model.eval()
text_model.eval()
audio_model.eval()

test_preds, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        labels = batch["labels"].to(device)
        t_emb = text_model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).last_hidden_state[:,0,:]
        a_emb = audio_model(batch["audio"].to(device)).last_hidden_state.mean(dim=1)
        logits = fusion_model(t_emb, a_emb)
        test_preds.extend(torch.argmax(logits, 1).cpu().numpy())
        test_true.extend(labels.cpu().numpy())

print("\n FINAL TEST F1:", f1_score(test_true, test_preds))
