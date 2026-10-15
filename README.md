# DMHardware AI Chatbot

A hardware-engineering RAG assistant with **1,049 deduplicated knowledge records**, TF-IDF retrieval, a **trained PyTorch topic classifier**, and Gemini generation. [Live demo](https://munsakad.github.io/DMHardware-Chat-Bot/) ? [Knowledge explorer](https://munsakad.github.io/DMHardware-Chat-Bot/knowledge.html)

## What the dataset count means

The count is retrieval records across **four domains**, not 1,049 independent knowledge areas:

| Domain | Records |
| --- | ---: |
| Electronics (including embedded/Arduino) | 317 |
| Digital Logic Design (including Verilog/VHDL) | 416 |
| PCB Design | 153 |
| Robotics | 163 |
| Total | 1,049 |

The original repository had 129 references and no PyTorch implementation. This expansion adds 792 distinct passages/code examples from the owner's 120 Hardware Education Hub lessons and 128 explicitly labeled generated engineering calculations. These are not externally measured observations or an independently reviewed textbook corpus.

Every record includes a source URL, source group, and record type. The lesson snapshot in data/source_lessons.json comes from munsakad/Hardware-Engineering-Education-Hub at commit c5fa04da7f8be79333d44a78d2cda816b7c94075. expand_knowledge.py extracts complete paragraphs, list items, code blocks, and tables; it does not create paraphrases to increase the count. The worked examples state their assumptions and formulas. generate_dataset.py removes duplicate normalized content and writes data/manifest.json. The UI displays the cleaned indexed count.

## What PyTorch does

train_topic_model.py trains a four-class linear classifier with PyTorch AdamW and cross-entropy over TF-IDF features from title and content. Category names/tags are excluded from classifier training inputs. Source lessons and formula families are kept together in the holdout split, so related passages/cases do not straddle training and validation.

Evaluation is a reproducible 20% grouped holdout (the first fold of five). models/metrics.json reports accuracy, macro F1, majority baseline, split sizes, and limitations. Vocabulary and IDF are fitted to the full unlabeled corpus: this is in-corpus topic classification, not a test of answer correctness, engineering reasoning, or retrieval relevance. After evaluation, the deployment model is refitted on all records.

At query time:

1. TF-IDF produces lexical similarity scores.
2. The trained model predicts Electronics, Robotics, PCB Design, or Digital Logic Design.
3. Each lexical score is multiplied by 0.85 + 0.15 ? the model's score for that record's domain. This is a soft prior; it does not exclude other domains.
4. Retrieved passages and source URLs ground Gemini's answer.

The Streamlit app performs model inference with PyTorch. GitHub Pages runs the exported weights with equivalent JavaScript math; **PyTorch itself is not running inside the browser**. The deployment checks both model outputs and final retrieval scores against Python. Displayed softmax scores are not calibrated confidence. Gemini is the answer generator; the PyTorch classifier is a small topic model, not a language model.

## Run locally

Use Python 3.12+ and create a project virtual environment, then:

```sh
pip install -r requirements.txt
python generate_dataset.py
python train_topic_model.py
python -m unittest -v test_pipeline
streamlit run app.py
```

Copy .env.example to .env and set GEMINI_API_KEY. The default is gemini-3.5-flash-lite; the sidebar can override it. Regenerate and retrain together after changing the dataset. The model checks its vocabulary/IDF fingerprint before use and fails clearly if it is stale.

## Build and verify the browser app

```sh
python generate_dataset.py
python train_topic_model.py
python -m unittest -v test_pipeline
python web/export_index.py _site
node web/check-parity.mjs _site
```

The GitHub Actions workflow also copies web assets, dataset manifest/CSV, and model metrics into the Pages artifact. No API key is deployed. With no visitor key, the demo returns retrieved source passages and still runs topic classification. The knowledge explorer searches the indexed records without an API key.

## Files to inspect

- generate_dataset.py / expand_knowledge.py: reproducible dataset construction and calculations.
- data/manifest.json / data/knowledge_base.csv: actual record count and provenance.
- rag/retriever.py / rag/topic_model.py: TF-IDF plus PyTorch inference.
- train_topic_model.py / models/metrics.json: training and measured validation results.
- web/retriever.js / web/check-parity.mjs: exported model inference and Python parity checks.

Accurate portfolio wording: ?Built a hardware RAG chatbot over 1,049 deduplicated reference, lesson, code, and worked-example records; trained and evaluated a PyTorch topic classifier used for retrieval reranking.?
