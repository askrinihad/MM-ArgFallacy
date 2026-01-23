import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)


train_text_emb = torch.load("embeddings/train_text.pt")
train_audio_emb = torch.load("embeddings/train_audio.pt")
train_labels = torch.tensor(torch.load("embeddings/train_labels.pt"))

val_text_emb = torch.load("embeddings/val_text.pt")
val_audio_emb = torch.load("embeddings/val_audio.pt")
val_labels = torch.tensor(torch.load("embeddings/val_labels.pt"))

test_text_emb = torch.load("embeddings/test_text.pt")
test_audio_emb = torch.load("embeddings/test_audio.pt")
test_labels = torch.tensor(torch.load("embeddings/test_labels.pt"))

#align embeddings and labels 
min_len_train = min(train_text_emb.shape[0], train_audio_emb.shape[0], train_labels.shape[0])
train_text_emb = train_text_emb[:min_len_train]
train_audio_emb = train_audio_emb[:min_len_train]
train_labels = train_labels[:min_len_train]

min_len_val = min(val_text_emb.shape[0], val_audio_emb.shape[0], val_labels.shape[0])
val_text_emb = val_text_emb[:min_len_val]
val_audio_emb = val_audio_emb[:min_len_val]
val_labels = val_labels[:min_len_val]

min_len_test = min(test_text_emb.shape[0], test_audio_emb.shape[0], test_labels.shape[0])
test_text_emb = test_text_emb[:min_len_test]
test_audio_emb = test_audio_emb[:min_len_test]
test_labels = test_labels[:min_len_test]

#concatenate embeddings for late fusion
train_emb = torch.cat([train_text_emb, train_audio_emb], dim=1)
val_emb = torch.cat([val_text_emb, val_audio_emb], dim=1)
test_emb = torch.cat([test_text_emb, test_audio_emb], dim=1)

#classifier model for late fusion
class FusionClassifier(nn.Module):
    def __init__(self, dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)
    def forward(self, x):
        return self.fc(x)

fusion_model = FusionClassifier(train_emb.size(1), 2).to(device)

classes = torch.unique(train_labels)
class_weights = torch.tensor([len(train_labels)/(2*(train_labels==c).sum()) for c in classes], dtype=torch.float).to(device)


criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-3)


train_loader_f = DataLoader(TensorDataset(train_emb, train_labels), batch_size=16, shuffle=True)
val_loader_f = DataLoader(TensorDataset(val_emb, val_labels), batch_size=16)

#training loop
epochs = 100
for epoch in range(epochs):
    fusion_model.train()
    total_loss = 0
    for x, y in train_loader_f:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = fusion_model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader_f):.4f}")


torch.save({
    "state_dict": fusion_model.state_dict(),
    "input_dim": train_emb.size(1)
}, "./saved_models_balanced/fusion/late-fusion_classifier.pt")
print(" Fusion classifier saved")


fusion_model.eval()
preds = []
with torch.no_grad():
    for i in range(0, len(test_labels), 16):
        x = test_emb[i:i+16].to(device)
        logits = fusion_model(x)
        preds.extend(torch.argmax(logits, 1).cpu().numpy())

f1_final = f1_score(test_labels.numpy(), preds, average="binary")
print(f"\n FINAL FUSION F1 (binary): {f1_final:.4f}")
