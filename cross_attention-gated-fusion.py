import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print("Using device:", device)

train_text = torch.load("embeddings/train_text.pt")
train_audio = torch.load("embeddings/train_audio.pt")
val_text = torch.load("embeddings/val_text.pt")
val_audio = torch.load("embeddings/val_audio.pt")
test_text = torch.load("embeddings/test_text.pt")
test_audio = torch.load("embeddings/test_audio.pt")

train_labels = torch.tensor(torch.load("embeddings/train_labels.pt"), dtype=torch.long)
val_labels = torch.tensor(torch.load("embeddings/val_labels.pt"), dtype=torch.long)
test_labels = torch.tensor(torch.load("embeddings/test_labels.pt"), dtype=torch.long)


train_ds = TensorDataset(train_text, train_audio, train_labels)
val_ds = TensorDataset(val_text, val_audio, val_labels)
test_ds = TensorDataset(test_text, test_audio, test_labels)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)
test_loader = DataLoader(test_ds, batch_size=16)

#focal loss with class weights
classes = torch.unique(train_labels)
weights = compute_class_weight("balanced", classes=classes.numpy(), y=train_labels.numpy())
class_weights = torch.tensor(weights, dtype=torch.float).to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

#cross-attention gated fusion model
class CrossAttnGatedFusion(nn.Module):
    def __init__(self, dim_text, dim_audio, hidden_dim=256, nhead=2, dropout=0.2):
        super().__init__()
        # Project embeddings to same hidden dimension
        self.text_proj = nn.Linear(dim_text, hidden_dim)
        self.audio_proj = nn.Linear(dim_audio, hidden_dim)

        # Cross-Attention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True, dropout=dropout)
        
        # LayerNorm + Dropout
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

        # Gating MLP (per-feature)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, t, a):
        # Project embeddings
        t_h = F.relu(self.text_proj(t))  # (B, H)
        a_h = F.relu(self.audio_proj(a)) # (B, H)

        # Stack sequence for attention: text queries audio
        q = t_h.unsqueeze(1)  # (B, 1, H)
        kv = a_h.unsqueeze(1) # (B, 1, H)

        attn_out, _ = self.attn(q, kv, kv)  # (B, 1, H)
        attn_out = attn_out.squeeze(1)       # (B, H)

        # Residual + LayerNorm
        fused = self.ln1(t_h + self.drop(attn_out))

        # Gating
        gate_input = torch.cat([fused, a_h], dim=1)
        gated = fused * self.gate_mlp(gate_input)

        # Final LayerNorm + classifier
        out = self.classifier(self.ln2(gated))
        return out

# Instantiate model
model = CrossAttnGatedFusion(train_text.size(1), train_audio.size(1)).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

criterion = FocalLoss(alpha=class_weights)

#training loop
best_f1 = 0.0
epochs =30
print("📦 Training Cross-Attention Gated Fusion")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for t_emb, a_emb, labels in train_loader:
        t_emb, a_emb, labels = t_emb.to(device), a_emb.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(t_emb, a_emb)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()

    # Validation
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for t_emb, a_emb, labels in val_loader:
            t_emb, a_emb, labels = t_emb.to(device), a_emb.to(device), labels.to(device)
            logits = model(t_emb, a_emb)
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_true.extend(labels.cpu().numpy())
    val_f1 = f1_score(val_true, val_preds, average="binary")
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

    # Save best model
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), "./saved_models_balanced/fusion/cross_attn_gated_fusion_best.pt")
        print("💾 Best model saved!")

#evaluation
model.load_state_dict(torch.load("./saved_models_balanced/fusion/cross_attn_gated_fusion_best.pt"))
model.eval()
test_preds = []
with torch.no_grad():
    for t_emb, a_emb, labels in test_loader:
        t_emb, a_emb = t_emb.to(device), a_emb.to(device)
        logits = model(t_emb, a_emb)
        test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

test_f1 = f1_score(test_labels.numpy(), test_preds, average="binary")
print(f"\n FINAL CROSS-ATTN GATED FUSION F1: {test_f1:.4f}")
