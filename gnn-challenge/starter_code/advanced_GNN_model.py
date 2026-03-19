import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.nn.norm import LayerNorm

# -----------------------------
# Helper: SAFE OUTPUT HANDLER
# -----------------------------
def get_node_output(out_dict):
    if isinstance(out_dict, dict):
        if "node" in out_dict:
            return out_dict["node"]
        else:
            key = list(out_dict.keys())[0]
            print(f"⚠️ Using fallback key: {key}")
            return out_dict[key]
    return out_dict


# -----------------------------
# Load Data
# -----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "../data")

train_df = pd.read_csv(os.path.join(DATA, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA, "test.csv"))
edges_df = pd.read_csv(os.path.join(DATA, "graph_edges.csv"))
node_df  = pd.read_csv(os.path.join(DATA, "node_types.csv"))

target_col = "disease_labels"

train_df = train_df.dropna(subset=[target_col])

# -----------------------------
# Node mapping
# -----------------------------
node_map = {nid: i for i, nid in enumerate(node_df.node_id)}
NUM_NODES = len(node_map)

# -----------------------------
# Graph
# -----------------------------
def build_graph():
    data = HeteroData()
    data["node"].num_nodes = NUM_NODES

    for etype in edges_df.edge_type.unique():
        df = edges_df[edges_df.edge_type == etype]
        src = torch.tensor([node_map[i] for i in df.src])
        dst = torch.tensor([node_map[i] for i in df.dst])
        data["node", etype, "node"].edge_index = torch.stack([src, dst])

    return data

train_graph = build_graph()
test_graph  = build_graph()

# -----------------------------
# Features
# -----------------------------
feat_cols = [c for c in train_df.columns if c not in ["node_id", target_col]]

feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train_df[c])]

X = torch.zeros((NUM_NODES, len(feat_cols)))

train_idx = torch.tensor([node_map[i] for i in train_df.node_id])
test_idx  = torch.tensor([node_map[i] for i in test_df.node_id])

train_feat = train_df[feat_cols].values.astype(np.float32)
test_feat  = test_df[feat_cols].values.astype(np.float32)

train_feat = np.nan_to_num(train_feat)
test_feat  = np.nan_to_num(test_feat)

mean = train_feat.mean(0)
std  = train_feat.std(0)
std[std == 0] = 1

train_feat = (train_feat - mean) / std
test_feat  = (test_feat - mean) / std

X[train_idx] = torch.tensor(train_feat)
X[test_idx]  = torch.tensor(test_feat)

train_graph["node"].x = X
test_graph["node"].x  = X

# -----------------------------
# Labels
# -----------------------------
y = -1 * np.ones(NUM_NODES)
y[train_idx] = train_df[target_col].values
y = torch.tensor(y, dtype=torch.long)

train_graph["node"].y = y
test_graph["node"].y  = y

# -----------------------------
# Split
# -----------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx_np = train_idx.numpy()
labels_np = train_df[target_col].values

train_split, val_split = next(skf.split(train_idx_np, labels_np))

train_idx_split = torch.tensor(train_idx_np[train_split])
val_idx_split   = torch.tensor(train_idx_np[val_split])

# -----------------------------
# Model
# -----------------------------
class GNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super().__init__()
        self.conv1 = SAGEConv(in_c, hid_c)
        self.conv2 = SAGEConv(hid_c, hid_c)
        self.lin   = nn.Linear(hid_c, out_c)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

base_model = GNN(X.size(1), 128, 2)
model = to_hetero(base_model, train_graph.metadata())

# -----------------------------
# Training setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
train_graph = train_graph.to(device)
test_graph  = test_graph.to(device)
y = y.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# Neighbor Loader
# -----------------------------
num_neighbors = {etype: [10, 10] for etype in train_graph.edge_types}

train_loader = NeighborLoader(
    train_graph,
    input_nodes=("node", train_idx_split),
    num_neighbors=num_neighbors,
    batch_size=32,
    shuffle=True
)

# -----------------------------
# TRAIN LOOP (FIXED)
# -----------------------------
best_f1 = 0

for epoch in range(1, 501):
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out_dict = model(batch.x_dict, batch.edge_index_dict)
        node_out = get_node_output(out_dict)

        root_size = batch["node"].batch_size
        root_nodes = torch.arange(root_size, device=device)

        out = node_out[root_nodes]
        labels = batch["node"].y[root_nodes]

        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

    # -------- VALIDATION --------
    model.eval()
    with torch.no_grad():
        val_out = model(train_graph.x_dict, train_graph.edge_index_dict)
        val_out = get_node_output(val_out)

        preds = val_out[val_idx_split].argmax(1).cpu().numpy()
        true  = y[val_idx_split].cpu().numpy()

        f1 = f1_score(true, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_state = model.state_dict()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Val F1: {f1:.4f}")

model.load_state_dict(best_state)

# -----------------------------
# TEST PREDICTION
# -----------------------------
model.eval()
with torch.no_grad():
    out = model(test_graph.x_dict, test_graph.edge_index_dict)
    out = get_node_output(out)

    logits = out[test_idx]
    preds = logits.argmax(1).cpu().numpy()

# -----------------------------
# SAVE
# -----------------------------
os.makedirs("submissions", exist_ok=True)

pd.DataFrame({
    "node_id": test_df.node_id,
    "target": preds
}).to_csv("submissions/final_preds.csv", index=False)

print("✅ DONE - Predictions saved!")
