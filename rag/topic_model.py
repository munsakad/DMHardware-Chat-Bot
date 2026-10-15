"""A trained PyTorch topic classifier used as a small retrieval ranking prior."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/topic_classifier.json"


def vocabulary_hash(vectorizer):
    terms = sorted(vectorizer.vocabulary_.items())
    payload = json.dumps({"terms": terms, "idf": [round(float(v), 7) for v in vectorizer.idf_]}, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class TopicModel:
    def __init__(self, vectorizer, path=MODEL_PATH):
        self.artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        if self.artifact["vocabulary_hash"] != vocabulary_hash(vectorizer):
            raise ValueError("Topic model does not match this knowledge base. Run python train_topic_model.py")
        self.labels = self.artifact["labels"]
        self.layer = nn.Linear(len(vectorizer.vocabulary_), len(self.labels))
        self.layer.load_state_dict({"weight": torch.tensor(self.artifact["weight"]), "bias": torch.tensor(self.artifact["bias"])})
        self.layer.eval()

    def probabilities(self, query_vector):
        with torch.inference_mode():
            values = torch.from_numpy(query_vector.toarray().astype(np.float32))
            return torch.softmax(self.layer(values), dim=1)[0].tolist()
