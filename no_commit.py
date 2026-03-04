import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 1. Load Raw Data
A = pd.read_csv("data/public/adjacency_matrix.csv", index_col=0).values
train = pd.read_csv("data/public/train.csv")
test = pd.read_csv("data/public/test.csv")

feat_cols = [c for c in train.columns if c not in ["node_id", "sample_id", "disease_labels"]]
all_nodes = pd.concat([train[["node_id"] + feat_cols], test[["node_id"] + feat_cols]], ignore_index=True)
all_nodes = all_nodes.drop_duplicates(subset=["node_id"], keep="last")

# 2. Convert to Tensors
X = torch.tensor(all_nodes[feat_cols].values, dtype=torch.float32)
src, dst = np.where(A > 0)
edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

# 3. Create PyG Data Object
data = Data(x=X, edge_index=edge_index)

# Map labels to indices
y = torch.full((X.size(0),), -1, dtype=torch.long)
node_to_idx = {node_id: i for i, node_id in enumerate(all_nodes["node_id"])}

for _, row in train.iterrows():
    idx = node_to_idx[row["node_id"]]
    y[idx] = int(row["disease_labels"])

data.y = y

# Create Masks
train_mask = torch.zeros(X.size(0), dtype=torch.bool)
test_mask = torch.zeros(X.size(0), dtype=torch.bool)

for node_id in train["node_id"]:
    train_mask[node_to_idx[node_id]] = True
for node_id in test["node_id"]:
    test_mask[node_to_idx[node_id]] = True

data.train_mask = train_mask
data.test_mask = test_mask

# --- MODEL DEFINITION ---
class VanillaGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Initialize Model
num_classes = int(train['disease_labels'].nunique())
model = VanillaGCN(
    in_channels=data.num_node_features, 
    hidden_channels=16, 
    out_channels=num_classes
)

# --- SIMPLE TRAINING LOOP ---
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item():.4f}')

# --- INFERENCE & SUBMISSION ---
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1) 

# Ensure we use the correct node_ids for the test set
test_node_ids = all_nodes.iloc[data.test_mask.numpy()]["node_id"].values
test_predictions = preds[data.test_mask].cpu().numpy()

submission = pd.DataFrame({
    "node_id": test_node_ids,
    "disease_labels": test_predictions
})

submission.to_csv("predictions.csv", index=False)
print("Submission saved to predictions.csv")





