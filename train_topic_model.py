"""Reproducible supervised training, grouped holdout evaluation and export.

Held-out passages are grouped by source lesson; calculation cases share a
formula-family group. This measures broad topic classification, not answer
correctness, retrieval relevance, or general hardware reasoning.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from rag.preprocess import load_knowledge_base
from rag.retriever import KnowledgeRetriever
from rag.topic_model import vocabulary_hash

ROOT = Path(__file__).resolve().parent


def fit(x, y, indices, classes):
    torch.manual_seed(42)
    model = nn.Linear(x.shape[1], classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(100):
        optimizer.zero_grad()
        loss = loss_fn(model(x[indices]), y[indices])
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def main():
    torch.set_num_threads(1)
    df = load_knowledge_base(str(ROOT / "data/knowledge_base.csv"))
    retriever = KnowledgeRetriever(df, use_topic=False)
    labels = sorted(df.category.unique().tolist())
    # Do not include explicit category labels/tags in classifier training text.
    features = retriever.vectorizer.transform(df.title + ". " + df.content)
    x = torch.from_numpy(features.toarray().astype(np.float32))
    y = torch.tensor([labels.index(label) for label in df.category], dtype=torch.long)
    groups = df.source_group.to_numpy()
    split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train, test = next(split.split(features, df.category, groups))
    assert not set(groups[train]) & set(groups[test])
    model = fit(x, y, train, len(labels))
    with torch.inference_mode():
        predicted = model(x[test]).argmax(dim=1).numpy()
    majority = np.bincount(y[train].numpy()).argmax()
    metrics = {"task": "four-domain topic classification", "seed": 42, "epochs": 100,
               "train_records": len(train), "validation_records": len(test),
               "train_source_groups": len(set(groups[train])), "validation_source_groups": len(set(groups[test])),
               "accuracy": float(accuracy_score(y[test], predicted)),
               "macro_f1": float(f1_score(y[test], predicted, average="macro")),
               "majority_baseline_accuracy": float(np.mean(y[test].numpy() == majority)),
               "split": "StratifiedGroupKFold 5 folds, first fold held out; source-lesson/formula groups disjoint",
               "limitation": "Vocabulary and IDF fit on the full unlabeled corpus. Validation measures in-corpus topic classification, not unseen-domain QA or retrieval quality."}
    # Refit on all records after evaluation for the shipped inference model.
    model = fit(x, y, np.arange(len(df)), len(labels))
    artifact = {"framework": "PyTorch", "architecture": "Linear TF-IDF topic classifier + softmax",
                "labels": labels, "vocabulary_hash": vocabulary_hash(retriever.vectorizer),
                "dataset_sha256": hashlib.sha256((ROOT / "data/knowledge_base.csv").read_bytes()).hexdigest(),
                "weight": [[round(float(v), 7) for v in row] for row in model.weight.detach().tolist()],
                "bias": [round(float(v), 7) for v in model.bias.detach().tolist()], "metrics": metrics}
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    (out / "topic_classifier.json").write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
