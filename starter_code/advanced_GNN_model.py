import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, GATConv
from torch_geometric.nn.norm import LayerNorm
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 1. Load data
# -----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "../data")

train_df = pd.read_csv(os.path.join(DATA, "train.csv"))   # cfRNA
test_df  = pd.read_csv(os.path.join(DATA, "test.csv"))    # placenta
test_labels_path = os.path.join(DATA, "test_labels.csv")
test_labels = pd.read_csv(test_labels_path, index_col=0) if os.path.exists(test_labels_path) else None
edges_df = pd.read_csv(os.path.join(DATA, "graph_edges.csv"))
node_df  = pd.read_csv(os.path.join(DATA, "node_types.csv"))

# -----------------------------
# DROP ROWS WITHOUT TARGETS & CLEAN DATA
# -----------------------------
missing_targets = train_df['disease_labels'].isna().sum()
print(f"{missing_targets} missing values in target column")

if missing_targets > 0:
    print("Dropping rows without target labels from training set...")
    train_df = train_df.dropna(subset=['disease_labels'])
    print(f"Training set now has {len(train_df)} samples with valid labels")

# Identify target column
target_col = 'disease_labels' if 'disease_labels' in train_df.columns else 'target'
has_test_labels = test_labels is not None

# ========================================
# Display Target Distribution
# ========================================
print("\n" + "="*70)
print("  📊 TARGET FEATURE DISTRIBUTION")
print("="*70)

print("\n🔹 TRAINING DATA (cfRNA):")
train_counts = train_df['disease_labels'].value_counts().sort_index()
print(f"   Total samples: {len(train_df)}")
for target_val, count in train_counts.items():
    pct = (count / len(train_df)) * 100
    label = "control" if target_val == 0 else "preeclampsia"
    print(f"   Class {target_val} ({label}): {count} samples ({pct:.1f}%)")

print("\n🔹 TESTING DATA (Placenta):")
print(f"   Total samples: {len(test_df)}")
if has_test_labels:
    test_counts = pd.Series(test_labels.iloc[:, 0]).value_counts().sort_index()
    for target_val, count in test_counts.items():
        pct = (count / len(test_df)) * 100
        label = "control" if target_val == 0 else "preeclampsia"
        print(f"   Class {target_val} ({label}): {count} samples ({pct:.1f}%)")
else:
    print("   ⚠️  No labels (inductive task - labels hidden for evaluation)")

print("="*70 + "\n")

# -----------------------------
# 2. Node indexing
# -----------------------------
node_ids = node_df["node_id"].tolist()
node_map = {nid: i for i, nid in enumerate(node_ids)}
NUM_NODES = len(node_ids)

# -----------------------------
# 3. Graph construction - ENHANCED with edge weighting
# -----------------------------
def build_graph_with_weights(allowed_edge_types):
    data = HeteroData()
    data["node"].num_nodes = NUM_NODES

    for etype in allowed_edge_types:
        df = edges_df[edges_df.edge_type == etype]
        if len(df) == 0:
            continue
            
        src = torch.tensor([node_map[i] for i in df.src], dtype=torch.long)
        dst = torch.tensor([node_map[i] for i in df.dst], dtype=torch.long)
        data["node", etype, "node"].edge_index = torch.stack([src, dst])
        
        # Add edge weights based on type (similarity edges get higher weight)
        if etype == "similarity":
            edge_weight = torch.ones(len(src), dtype=torch.float) * 1.0
        elif etype == "ancestry":
            edge_weight = torch.ones(len(src), dtype=torch.float) * 0.8
        else:
            edge_weight = torch.ones(len(src), dtype=torch.float) * 0.5
            
        data["node", etype, "node"].edge_weight = edge_weight

    return data

# Use both edge types for better inductive transfer
USE_ANCESTRY_IN_TEST = True  # Changed from False to True
train_graph = build_graph_with_weights(["similarity", "ancestry"])
test_graph  = build_graph_with_weights(["similarity", "ancestry"])

# -----------------------------
# 4. Node features - ENHANCED with normalization and feature selection
# -----------------------------
# Only use columns that exist in both train and test datasets
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)
shared_cols = train_cols.intersection(test_cols)

# Exclude non-feature columns
exclude_cols = ["node_id", target_col, "sample_id", "Unnamed: 0"] if "Unnamed: 0" in shared_cols else ["node_id", target_col, "sample_id"]
feat_cols = [c for c in shared_cols if c not in exclude_cols]
feat_cols = sorted(feat_cols)  # For consistent ordering

# Remove constant and near-constant features
print(f"Initial feature count: {len(feat_cols)}")
train_feat_values = train_df[feat_cols].values
variances = np.var(train_feat_values, axis=0)
constant_mask = variances < 1e-6
if np.any(constant_mask):
    constant_features = [feat_cols[i] for i in range(len(feat_cols)) if constant_mask[i]]
    print(f"Dropping {len(constant_features)} constant/near-constant features")
    feat_cols = [f for i, f in enumerate(feat_cols) if not constant_mask[i]]

print(f"Final feature count: {len(feat_cols)}")

X = torch.zeros((NUM_NODES, len(feat_cols)))

train_idx = torch.tensor([node_map[i] for i in train_df.node_id], dtype=torch.long)
test_idx  = torch.tensor([node_map[i] for i in test_df.node_id], dtype=torch.long)

# Fill features and apply normalization
train_features = train_df[feat_cols].values.astype(np.float32)
test_features = test_df[feat_cols].values.astype(np.float32)

# Standardize features (z-score normalization)
mean = np.mean(train_features, axis=0)
std = np.std(train_features, axis=0)
std[std == 0] = 1.0  # Avoid division by zero

train_features_norm = (train_features - mean) / std
test_features_norm = (test_features - mean) / std

X[train_idx] = torch.tensor(train_features_norm, dtype=torch.float)
X[test_idx]  = torch.tensor(test_features_norm, dtype=torch.float)

train_graph["node"].x = X
test_graph["node"].x  = X

# -----------------------------
# 5. Labels (train only)
# -----------------------------
y = -1 * np.ones(NUM_NODES, dtype=int)
y[train_idx] = train_df[target_col].values.astype(int)
y = torch.tensor(y, dtype=torch.long)
print(f"✅ Labels assigned. Train nodes: {len(train_idx)}, Total nodes: {NUM_NODES}")
train_graph["node"].y = y
test_graph["node"].y = y

# -----------------------------
# 5b. Train/validation split - ENHANCED with stratification
# -----------------------------
train_labels = train_df[target_col].values.astype(int)
train_idx_np = train_idx.cpu().numpy()

# Use stratified k-fold for better validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx_split, val_idx_split = next(skf.split(train_idx_np, train_labels))

train_idx_split = torch.tensor(train_idx_np[train_idx_split], dtype=torch.long)
val_idx_split = torch.tensor(train_idx_np[val_idx_split], dtype=torch.long)

print(f"Train split size: {len(train_idx_split)}, Validation split size: {len(val_idx_split)}")

# -----------------------------
# 6. Enhanced GraphSAGE model with residual connections and layer norm
# -----------------------------
class EnhancedSAGEBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.3, use_residual=True):
        super().__init__()
        self.conv = SAGEConv(in_c, out_c, aggr='mean', normalize=True)
        self.norm = LayerNorm(out_c)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual and in_c == out_c
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x, edge_index):
        identity = x
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.activation(x)
        if self.use_residual:
            x = x + identity
        x = self.dropout(x)
        return x

class EnhancedGNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_c, hid_c)
        self.input_norm = LayerNorm(hid_c)
        
        # Hidden layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hid_c if i > 0 else hid_c
            out_dim = hid_c
            # Use residual for all but first layer if dimensions match
            use_res = i > 0
            self.layers.append(EnhancedSAGEBlock(in_dim, out_dim, dropout, use_res))
        
        # Output layer
        self.output_proj = nn.Sequential(
            nn.Linear(hid_c, hid_c // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hid_c // 2, out_c)
        )
        
    def forward(self, x, edge_index):
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.leaky_relu(x, 0.1)
        
        # Apply graph layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # Output projection
        return self.output_proj(x)

num_classes = len(train_df[target_col].unique())
base_model = EnhancedGNN(
    in_c=X.size(1), 
    hid_c=128,  # Increased from 64
    out_c=num_classes,
    num_layers=3,  # Increased from 2
    dropout=0.4
)

# Convert to hetero model
model = to_hetero(base_model, train_graph.metadata(), aggr="mean")

# -----------------------------
# 7. Training setup with advanced techniques
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
train_graph = train_graph.to(device)
test_graph = test_graph.to(device)
y = y.to(device)

# Enhanced class weighting with smoothing
num_samples = len(train_df)
class_counts = train_df[target_col].value_counts().sort_index()

# Apply label smoothing for weights to prevent overfitting to minority class
smooth_factor = 0.1
smoothed_counts = class_counts.values * (1 - smooth_factor) + smooth_factor * (num_samples / len(class_counts))

weights = torch.tensor([
    num_samples / (2 * smoothed_counts[0]),
    num_samples / (2 * smoothed_counts[1])
], dtype=torch.float).to(device)

print(f"Using smoothed class weights: {weights.cpu().numpy()}")

# Loss function with label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Optimizer with weight decay and AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=5e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=20, verbose=True
)

# -----------------------------
# 8. Enhanced Training with Neighborhood Sampling
# -----------------------------
# Adaptive neighbor sampling based on graph density
num_neighbors = {etype: [10, 10, 5] for etype in train_graph.edge_types}  # Adjusted for 3 layers

train_loader = NeighborLoader(
    train_graph,
    input_nodes=("node", train_idx_split),
    num_neighbors=num_neighbors,
    batch_size=32,  # Increased from 16
    shuffle=True,
    drop_last=True
)

print("Starting enhanced neighborhood mini-batch training...")
best_val_f1 = 0.0  # Track F1 instead of loss
best_state = None
patience = 100
patience_left = patience

for epoch in range(1, 5001):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x_dict, batch.edge_index_dict)["node"]
        
        # Get root nodes
        root_nodes = torch.arange(batch.batch_size_dict["node"], device=device)
        batch_labels = batch["node"].y[root_nodes]
        
        # Compute loss
        loss = criterion(out[root_nodes], batch_labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(train_graph.x_dict, train_graph.edge_index_dict)["node"]
        val_preds = val_logits[val_idx_split.to(device)].argmax(dim=1).cpu().numpy()
        val_true = y[val_idx_split.to(device)].cpu().numpy()
        
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        val_loss = criterion(val_logits[val_idx_split.to(device)], y[val_idx_split.to(device)]).item()
        
        # Update scheduler
        scheduler.step(val_loss)
    
    # Save best model based on F1 score
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        patience_left = patience
    else:
        patience_left -= 1
    
    if epoch % 200 == 0:
        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:04d} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")
    
    if patience_left == 0:
        print(f"Early stopping at epoch {epoch:04d} (best val F1: {best_val_f1:.4f})")
        break

if best_state is not None:
    model.load_state_dict(best_state)

# -----------------------------
# 9. Inductive testing with ensemble prediction
# -----------------------------
print("\nGenerating inductive predictions for placenta nodes...")
model.eval()

# Use test-time augmentation with different edge types
with torch.no_grad():
    # Primary prediction
    logits = model(test_graph.x_dict, test_graph.edge_index_dict)["node"]
    preds = logits[test_idx].argmax(dim=1).cpu().numpy()
    proba = torch.softmax(logits[test_idx], dim=1).cpu().numpy()
    
    # Confidence calibration using temperature scaling
    temperature = 1.2
    calibrated_proba = torch.softmax(logits[test_idx] / temperature, dim=1).cpu().numpy()

# Evaluate on training set
print("\n" + "="*70)
print("  📊 TRAINING SET EVALUATION METRICS")
print("="*70)
with torch.no_grad():
    train_logits = model(train_graph.x_dict, train_graph.edge_index_dict)["node"]
    train_preds = train_logits[train_idx].argmax(dim=1).cpu().numpy()
    train_true = y[train_idx].cpu().numpy()

train_acc = accuracy_score(train_true, train_preds)
train_prec = precision_score(train_true, train_preds, zero_division=0)
train_rec = recall_score(train_true, train_preds, zero_division=0)
train_f1 = f1_score(train_true, train_preds, zero_division=0)
train_cm = confusion_matrix(train_true, train_preds)

print(f"\n  Accuracy:     {train_acc:.4f}")
print(f"  Precision:    {train_prec:.4f}")
print(f"  Recall:       {train_rec:.4f}")
print(f"  F1-Score:     {train_f1:.4f}")
print(f"\n  Confusion Matrix:")
print(f"     TN={train_cm[0,0]:3d}  FP={train_cm[0,1]:3d}")
print(f"     FN={train_cm[1,0]:3d}  TP={train_cm[1,1]:3d}")

# Test set prediction statistics
print("\n" + "="*70)
print("  🔮 TEST SET PREDICTIONS & EVALUATION (INDUCTIVE - PLACENTA)")
print("="*70)

pred_counts = np.bincount(preds, minlength=2)
print(f"\n📌 Predicted Labels for {len(preds)} Placenta Nodes:")
print(f"   Class 0 (control):       {pred_counts[0]:3d} nodes ({pred_counts[0]/len(preds)*100:.1f}%)")
print(f"   Class 1 (preeclampsia):  {pred_counts[1]:3d} nodes ({pred_counts[1]/len(preds)*100:.1f}%)")

print(f"\n📊 Prediction Confidence Analysis:")
max_conf = calibrated_proba.max(axis=1)
print(f"   Mean max confidence: {max_conf.mean():.4f}")
print(f"   Min confidence:      {max_conf.min():.4f}")
print(f"   Max confidence:      {max_conf.max():.4f}")
print(f"   Std deviation:       {max_conf.std():.4f}")

# Count high confidence predictions
high_conf_mask = max_conf >= 0.9
print(f"\n   High confidence (≥0.90): {high_conf_mask.sum()} predictions ({high_conf_mask.sum()/len(preds)*100:.1f}%)")
med_conf_mask = (max_conf >= 0.7) & (max_conf < 0.9)
print(f"   Medium confidence (0.70-0.89): {med_conf_mask.sum()} predictions ({med_conf_mask.sum()/len(preds)*100:.1f}%)")
low_conf_mask = max_conf < 0.7
print(f"   Low confidence (<0.70): {low_conf_mask.sum()} predictions ({low_conf_mask.sum()/len(preds)*100:.1f}%)")

# Test set evaluation against true labels
if test_labels is not None:
    print("\n" + "="*70)
    print("  📊 TEST SET EVALUATION METRICS (Against True Labels)")
    print("="*70)

    test_true = test_labels.iloc[:, 0].values.astype(int)

    test_acc = accuracy_score(test_true, preds)
    test_prec = precision_score(test_true, preds, zero_division=0)
    test_rec = recall_score(test_true, preds, zero_division=0)
    test_f1 = f1_score(test_true, preds, zero_division=0)
    test_cm = confusion_matrix(test_true, preds)

    print(f"\n  Accuracy:     {test_acc:.4f}")
    print(f"  Precision:    {test_prec:.4f}")
    print(f"  Recall:       {test_rec:.4f}")
    print(f"  F1-Score:     {test_f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"     TN={test_cm[0,0]:3d}  FP={test_cm[0,1]:3d}")
    print(f"     FN={test_cm[1,0]:3d}  TP={test_cm[1,1]:3d}")

    print(f"\n📊 Prediction Breakdown by True Label:")
    print(f"   True Class 0 samples: {(test_true == 0).sum()} nodes")
    print(f"   True Class 1 samples: {(test_true == 1).sum()} nodes")

    correct_mask = preds == test_true
    incorrect_mask = preds != test_true

    print(f"\n   Correct predictions: {correct_mask.sum()} ({correct_mask.sum()/len(preds)*100:.1f}%)")
    print(f"   Incorrect predictions: {incorrect_mask.sum()} ({incorrect_mask.sum()/len(preds)*100:.1f}%)")

    if correct_mask.sum() > 0:
        print(f"   Mean confidence (correct): {max_conf[correct_mask].mean():.4f}")
    if incorrect_mask.sum() > 0:
        print(f"   Mean confidence (incorrect): {max_conf[incorrect_mask].mean():.4f}")

    print("\n" + "="*70)
else:
    print("\n" + "="*70)
    print("  📊 TEST SET EVALUATION METRICS")
    print("="*70)
    print("  ⚠️  test_labels.csv not found. Skipping test-set evaluation.")
    print("\n" + "="*70)

# -----------------------------
# 10. Save predictions with calibrated probabilities
# -----------------------------
os.makedirs("submissions", exist_ok=True)

# Hard predictions
submission_hard = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": preds
})
submission_hard.to_csv("submissions/enhanced_gnn_preds.csv", index=False)

# Soft predictions with calibrated confidence
submission_soft = pd.DataFrame({
    "node_id": test_df.node_id,
    "target": preds,
    "confidence_control": calibrated_proba[:, 0],
    "confidence_preeclampsia": calibrated_proba[:, 1],
    "raw_confidence_control": proba[:, 0],
    "raw_confidence_preeclampsia": proba[:, 1]
})
submission_soft.to_csv("submissions/enhanced_gnn_preds_with_confidence.csv", index=False)

# Save model metadata for reproducibility
model_info = {
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.4,
    "num_features": len(feat_cols),
    "best_val_f1": best_val_f1,
    "use_ancestry": USE_ANCESTRY_IN_TEST
}
pd.Series(model_info).to_csv("submissions/model_info.csv")

print("\n✅ Enhanced predictions saved successfully!")
print(f"   Hard: submissions/enhanced_gnn_preds.csv")
print(f"   Soft: submissions/enhanced_gnn_preds_with_confidence.csv")
print(f"   Model info: submissions/model_info.csv")
print(f"   Total predictions: {len(preds)}")
