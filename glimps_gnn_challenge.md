# A.4 C04: GLIMPS-GNN

## A.4.1 Competition Overview

The GLIMPS-GNN (Graph-based Liquid-biopsy Inductive Modeling for PreeclampSia) challenge is an inductive node-classification task over a heterogeneous biological graph. Given $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with feature matrix $\mathbf{X} \in \mathbb{R}^{|\mathcal{V}| \times d}$, the goal is to learn a mapping $f(v_i \mid \mathcal{G}_{\text{train}}) \rightarrow y_i$ for each test node $v_i$, where $y_i \in \{0,1\}$ with $0=$ Control and $1=$ Preeclampsia. The training subgraph comprises cell-free RNA (cfRNA) from maternal plasma, while the inductive test set contains unseen placental RNA-seq samples. Models must therefore generalize across a substantial domain shift between liquid-biopsy and tissue transcriptomics without label leakage.

## A.4.2 Dataset Description

Data are sourced from GEO: maternal plasma cfRNA (GSE192902) and placental RNA-seq (GSE234729). Nodes are individual samples; edges encode similarity and ancestry. The graph contains $|\mathcal{V}| = 320$ nodes and $|\mathcal{E}| \approx 3{,}200$ edges. Node features are harmonized gene-expression profiles. The training set has $n_{\text{train}} = 209$ cfRNA samples; the test set has $n_{\text{test}} = 111$ placenta samples with moderately imbalanced classes.

The benchmark tests four difficulties: (i) cross-domain distribution shift between plasma cfRNA and placental tissue; (ii) high-dimensional sparsity (large $d$, small $n$); (iii) noisy, partially missing metadata; and (iv) strict inductive inference on entirely unseen test nodes.

## A.4.3 Input Files

Organizers provide tabular CSVs and serialized PyG artifacts:

- `train.csv` / `test.csv` — Node features and labels (labels withheld for test);
- `adjacency_matrix.csv` — Dense $\mathbf{A} \in \{0,1\}^{320 \times 320}$;
- `graph_edges.csv` — Sparse COO edge list with `src`, `dst`, `edge_type`;
- `node_types.csv` — Node ID registry aligning $\mathbf{A}$, $\mathbf{X}$, and edges;
- `graph_artifacts.pt` — Pre-built PyTorch Geometric `HeteroData` objects.

Two reference implementations are included: a three-layer MLP baseline with BatchNorm and inverse-frequency class weighting, and an advanced GraphSAGE inductive GNN with neighborhood mini-batch sampling and heterogeneous edge-type support.

## A.4.4 Submission and Evaluation

The competition uses an automated GitHub Actions pipeline:

1. **Generate** a `predictions.csv` with columns `id` and `y_pred` (probability or hard label);
2. **Encrypt** it with the organizer GPG public key to produce `predictions.csv.enc`;
3. **Submit** the encrypted file plus `metadata.json` (team, run, model name, type, submitter) via pull request;
4. **Auto-score:** the workflow decrypts, validates format, scores against hidden labels, posts metrics, updates the leaderboard, and closes the PR.

Performance is measured by macro-averaged F1-score (primary), which grants equal weight to both classes and discourages majority-class collapse. Accuracy, precision, and recall are secondary. Probabilistic outputs are thresholded at 0.5. Each participant may submit exactly once (CI-enforced), and training must remain within a 3-hour CPU budget.
