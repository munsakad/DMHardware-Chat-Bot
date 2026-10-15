import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from rag.preprocess import load_knowledge_base
from rag.retriever import KnowledgeRetriever
from rag.topic_model import MODEL_PATH

ROOT = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_knowledge_base(str(ROOT / "data/knowledge_base.csv"))
        cls.retriever = KnowledgeRetriever(cls.df)

    def test_real_cleaned_record_count_and_provenance(self):
        manifest = json.loads((ROOT / "data/manifest.json").read_text())
        self.assertGreaterEqual(len(self.df), 1000)
        self.assertEqual(len(self.df), manifest["records"])
        self.assertEqual(len(set(" ".join(c.split()).lower() for c in self.df.content)), len(self.df))
        self.assertTrue(self.df.source_url.str.startswith("https://").all())
        self.assertEqual(len(self.df.category.unique()), 4)
        self.assertEqual(sum(manifest["record_types"].values()), len(self.df))

    def test_trained_model_matches_dataset_and_has_holdout(self):
        artifact = json.loads(MODEL_PATH.read_text())
        self.assertEqual(artifact["framework"], "PyTorch")
        self.assertEqual(artifact["dataset_sha256"], hashlib.sha256((ROOT / "data/knowledge_base.csv").read_bytes()).hexdigest())
        metrics = artifact["metrics"]
        self.assertGreater(metrics["validation_source_groups"], 0)
        self.assertEqual(metrics["train_records"] + metrics["validation_records"], len(self.df))
        self.assertGreater(metrics["accuracy"], metrics["majority_baseline_accuracy"])

    def test_topic_predictions_and_grounded_retrieval(self):
        for question, category in [("PCB via plated drill copper layers", "PCB Design"),
                                   ("robot wheel encoder odometry motor control", "Robotics"),
                                   ("Verilog VHDL flip-flop sequential digital logic", "Digital Logic Design")]:
            probabilities = self.retriever.classify(question)
            self.assertTrue(all(np.isfinite(v) for v in probabilities.values()))
            self.assertAlmostEqual(sum(probabilities.values()), 1, places=5)
            self.assertEqual(max(probabilities, key=probabilities.get), category)
            results = self.retriever.retrieve(question)
            self.assertTrue(results)
            self.assertEqual(results[0].category, category)
            self.assertTrue(results[0].source_url)

    def test_empty_and_unknown_queries(self):
        for query in ("", "   ", "zzqxxqzq"):
            self.assertEqual(self.retriever.retrieve(query), [])
            self.assertIsNone(self.retriever.classify(query))


if __name__ == "__main__":
    unittest.main()
