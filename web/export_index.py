"""
Exports the TF-IDF retrieval index for the browser version of the chatbot.

The index is built by the same code the Streamlit app uses
(rag.preprocess.load_knowledge_base + rag.retriever.KnowledgeRetriever), then
serialized so web/retriever.js can score queries in the browser:
vocabulary, idf weights, stop words, and the L2-normalized document vectors.

It also writes parity.json: top-k results from the Python retriever for a set
of sample questions, which web/check-parity.mjs compares against the JS port.

Usage: python web/export_index.py <output-dir>
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.preprocess import load_knowledge_base  # noqa: E402
from rag.retriever import KnowledgeRetriever  # noqa: E402

PARITY_QUERIES = [
    "How do I pick a resistor for an LED?",
    "What is a PCB via?",
    "Explain how a JK flip-flop works",
    "PID control for a DC motor",
    "Where should decoupling capacitors go?",
    "Difference between I2C and SPI",
    "How does an H-bridge drive a motor?",
    "Karnaugh map simplification",
    "Why use a ground plane?",
    "Servo vs stepper motor",
    "Ohm's law example with a 9V battery",
    "What are setup and hold times?",
    "Trace width for high current",
    "Op amp non-inverting gain",
    "How do I debounce a push button?",
]


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_knowledge_base(str(ROOT / "data" / "knowledge_base.csv"))
    retriever = KnowledgeRetriever(df)
    vec = retriever.vectorizer
    matrix = retriever.matrix.tocsr()

    docs = []
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        flat = []
        for j, v in zip(row.indices.tolist(), row.data.tolist()):
            flat += [int(j), round(float(v), 7)]
        docs.append(flat)

    index = {
        "records": [
            {
                "title": r["title"],
                "category": r["category"],
                "subcategory": r["subcategory"],
                "content": r["content"],
            }
            for _, r in df.iterrows()
        ],
        "vocab": {term: int(i) for term, i in vec.vocabulary_.items()},
        "idf": [round(float(x), 7) for x in vec.idf_],
        "stop_words": sorted(vec.get_stop_words() or []),
        "docs": docs,
    }
    parity = [
        {"q": q, "top": [{"title": r.title, "score": round(r.score, 6)} for r in retriever.retrieve(q, top_k=5)]}
        for q in PARITY_QUERIES
    ]

    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    (out / "parity.json").write_text(json.dumps(parity, indent=1), encoding="utf-8")
    print(f"Indexed {len(df)} records, {len(index['vocab'])} terms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "web")
