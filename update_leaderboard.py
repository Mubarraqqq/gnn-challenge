#!/usr/bin/env python3
"""
update_leaderboard.py

Scores new submissions and updates leaderboard/leaderboard.csv (source of truth)
and regenerates leaderboard.md + docs/leaderboard.csv for GitHub Pages.
"""

import os
import csv
import json
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

# Paths
SUBMISSIONS_DIR = Path("submissions/inbox")
DATA_DIR = Path("data")
LEADERBOARD_CSV = Path("leaderboard/leaderboard.csv")
LEADERBOARD_MD = Path("leaderboard.md")
DOCS_LEADERBOARD_CSV = Path("docs/leaderboard.csv")
ORGANIZER_SUBMISSIONS = [
    ("submissions/advanced_gnn_preds.csv", "organizers", "advanced_gnn", "Advanced GNN (GraphSAGE)", "human", "organizer_bot"),
]

# Load ground truth
TEST_LABELS = DATA_DIR / "test_labels.csv"
if not TEST_LABELS.exists():
    raise FileNotFoundError("data/test_labels.csv not found. Hidden labels are required for scoring.")

test_labels = pd.read_csv(TEST_LABELS, index_col=0)
if test_labels.shape[1] == 0:
    raise ValueError("test_labels.csv must have at least one column with labels.")

test_true = test_labels.iloc[:, 0].values.astype(int)

# Sync organizer baseline predictions into inbox format if present.
def _sync_organizer_submissions():
    for src, team, run_id, model_name, model_type, submitter in ORGANIZER_SUBMISSIONS:
        src_path = Path(src)
        if not src_path.exists():
            continue
        out_dir = SUBMISSIONS_DIR / team / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_out = out_dir / "predictions.csv"
        meta_out = out_dir / "metadata.json"

        with src_path.open(newline="") as f_in, pred_out.open("w", newline="") as f_out:
            reader = csv.DictReader(f_in)
            if "node_id" not in reader.fieldnames or "target" not in reader.fieldnames:
                print(f"⚠️ Skipping {src}: expected columns node_id,target")
                continue
            writer = csv.DictWriter(f_out, fieldnames=["id", "y_pred"])
            writer.writeheader()
            for row in reader:
                writer.writerow({"id": row["node_id"], "y_pred": row["target"]})

        meta = {
            "team": team,
            "run_id": run_id,
            "model_name": model_name,
            "model_type": model_type,
            "submitter": submitter,
        }
        with meta_out.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

_sync_organizer_submissions()


def _discover_submission_files():
    encrypted = list(SUBMISSIONS_DIR.glob("*/*/predictions.csv.enc"))
    plaintext = list(SUBMISSIONS_DIR.glob("*/*/predictions.csv"))

    by_key = {}
    for path in encrypted:
        key = (path.parent.parent.name, path.parent.name)
        by_key[key] = path

    for path in plaintext:
        key = (path.parent.parent.name, path.parent.name)
        if key not in by_key:
            by_key[key] = path

    return list(by_key.values())


def _decrypt_submission_file(enc_path: Path, out_dir: Path):
    out_path = out_dir / f"{enc_path.parent.parent.name}__{enc_path.parent.name}__predictions.csv"
    passphrase = os.environ.get("SUBMISSION_PRIVATE_KEY_PASSPHRASE", "")

    cmd = ["gpg", "--batch", "--yes"]
    if passphrase:
        cmd.extend(["--pinentry-mode", "loopback", "--passphrase", passphrase])
    cmd.extend(["--decrypt", "-o", str(out_path), str(enc_path)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print("⚠️ gpg is not installed. Encrypted submissions cannot be scored in this environment.")
        return None
    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        reason = stderr[-1] if stderr else "gpg decryption failed"
        print(f"⚠️ Failed to decrypt {enc_path}: {reason}")
        return None
    return out_path

# Ensure leaderboard CSV exists
LEADERBOARD_CSV.parent.mkdir(parents=True, exist_ok=True)
if not LEADERBOARD_CSV.exists():
    with open(LEADERBOARD_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "team",
            "run_id",
            "model",
            "model_type",
            "f1_score",
            "accuracy",
            "precision",
            "recall",
            "submission_date",
            "submitter",
            "submitter_url",
            "pr_number",
            "submission_path",
        ])

leaderboard_df = pd.read_csv(LEADERBOARD_CSV)
prev_by_key = {}
if not leaderboard_df.empty:
    for _, row in leaderboard_df.iterrows():
        key = (str(row.get("team", "")), str(row.get("run_id", "")))
        prev_by_key[key] = row

# Find submissions
submission_files = _discover_submission_files()
print(f"🔍 Found {len(submission_files)} submission file(s) in submissions/inbox")

new_rows = []

with tempfile.TemporaryDirectory(prefix="leaderboard_decrypt_") as decrypt_dir:
    decrypt_dir_path = Path(decrypt_dir)

    for pred_path in submission_files:
        team = pred_path.parent.parent.name
        run_id = pred_path.parent.name
        key = (team, run_id)

        meta_path = pred_path.parent / "metadata.json"
        if not meta_path.exists():
            print(f"⚠️ Missing metadata.json for {pred_path}. Skipping.")
            continue

        score_path = pred_path
        if pred_path.name.endswith(".enc"):
            score_path = _decrypt_submission_file(pred_path, decrypt_dir_path)
            if score_path is None:
                continue

        try:
            submission = pd.read_csv(score_path)
        except Exception as e:
            print(f"❌ Failed to read {pred_path}: {e}")
            continue

        # Validate format: id, y_pred
        if "id" not in submission.columns or "y_pred" not in submission.columns:
            print(f"⚠️ Invalid format in {pred_path}. Required columns: id, y_pred")
            continue

        # Align predictions with test labels using id
        submission = submission.rename(columns={"id": "node_id"})
        if submission["y_pred"].dtype.kind in {"f", "c"}:
            proba = submission["y_pred"].astype(float).values
            preds = (proba >= 0.5).astype(int)
        else:
            preds = submission["y_pred"].astype(int).values

        if len(preds) != len(test_true):
            print(f"⚠️ Length mismatch in {pred_path}. Expected {len(test_true)}, got {len(preds)}")
            continue

        # Compute metrics
        f1 = f1_score(test_true, preds, average="macro", zero_division=0)
        acc = accuracy_score(test_true, preds)
        prec = precision_score(test_true, preds, zero_division=0)
        rec = recall_score(test_true, preds, zero_division=0)
        _ = confusion_matrix(test_true, preds)

        # Metadata
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

        model_name = meta.get("model_name", f"{team}-{run_id}")
        model_type = meta.get("model_type", "unknown")

        key = (team, run_id)
        submitter = meta.get("submitter", "participant")
        submitter_url = f"https://github.com/{submitter}" if submitter else ""

        prev = prev_by_key.get(key)
        metrics_unchanged = False
        if prev is not None:
            try:
                metrics_unchanged = (
                    abs(float(prev["f1_score"]) - f1) < 1e-9
                    and abs(float(prev["accuracy"]) - acc) < 1e-9
                    and abs(float(prev["precision"]) - prec) < 1e-9
                    and abs(float(prev["recall"]) - rec) < 1e-9
                )
            except Exception:
                metrics_unchanged = False

        submission_date = prev["submission_date"] if (metrics_unchanged and prev is not None) else datetime.now().strftime("%Y-%m-%d %H:%M")

        new_rows.append({
            "rank": 0,
            "team": team,
            "run_id": run_id,
            "model": model_name,
            "model_type": model_type,
            "f1_score": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "submission_date": submission_date,
            "submitter": submitter,
            "submitter_url": submitter_url,
            "pr_number": meta.get("pr_number", ""),
            "submission_path": str(pred_path),
        })

if not new_rows:
    print("⚠️ No valid submissions found in inbox.")
    exit(0)

# Rebuild from inbox and re-rank
updated_df = pd.DataFrame(new_rows)
updated_df = updated_df.sort_values(by=["f1_score", "accuracy"], ascending=False, ignore_index=True)
metric_cols = ["f1_score", "accuracy", "precision", "recall"]
updated_df[metric_cols] = updated_df[metric_cols].round(2)
# Tied F1 scores share the same rank.
updated_df["rank"] = updated_df["f1_score"].rank(method="dense", ascending=False).astype(int)

# Write CSV (source of truth)
updated_df.to_csv(LEADERBOARD_CSV, index=False)

# Copy CSV to docs for GitHub Pages
DOCS_LEADERBOARD_CSV.parent.mkdir(parents=True, exist_ok=True)
updated_df.to_csv(DOCS_LEADERBOARD_CSV, index=False)

# Render markdown leaderboard
md_lines = [
    "# 🏆 GNN Challenge Leaderboard",
    "",
    "## Current Leaderboard",
    "",
    "| Rank | Team | Run | Model | Type | F1-Score | Accuracy | Precision | Recall | Date | Submitter |",
    "|------|------|-----|-------|------|----------|----------|-----------|--------|------|-----------|",
]

for _, row in updated_df.iterrows():
    md_lines.append(
        "| {rank} | {team} | {run_id} | {model} | {model_type} | {f1_score:.2f} | {accuracy:.2f} | {precision:.2f} | {recall:.2f} | {submission_date} | {submitter} |".format(
            rank=int(row["rank"]),
            team=row["team"],
            run_id=row["run_id"],
            model=row["model"],
            model_type=row["model_type"],
            f1_score=row["f1_score"],
            accuracy=row["accuracy"],
            precision=row["precision"],
            recall=row["recall"],
            submission_date=row["submission_date"],
            submitter=row["submitter"],
        )
    )

md_lines.extend([
    "",
    "## Notes",
    "- This leaderboard is auto-generated from `leaderboard/leaderboard.csv`.",
    "- Submissions must follow the `submissions/inbox/<team>/<run_id>/predictions.csv.enc` format.",
])

LEADERBOARD_MD.write_text("\n".join(md_lines), encoding="utf-8")

print(f"✅ Leaderboard updated with {len(new_rows)} new submission(s)")
print(f"   Top model: {updated_df.iloc[0]['model']} (F1={updated_df.iloc[0]['f1_score']:.2f})")
