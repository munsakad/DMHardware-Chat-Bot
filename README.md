# DMHardware AI Chatbot

A Gemini-powered chatbot grounded in a curated knowledge base covering
**Electronics, Robotics, PCB Design, and Digital Logic Design**.

Built with a retrieval-augmented generation (RAG) pipeline:
- **Data preprocessing:** pandas / NumPy (cleaning, dedup, normalization)
- **Retrieval:** scikit-learn TF-IDF + cosine similarity, anchoring Gemini's
  answers in domain-specific engineering context
- **Generation:** Google Gemini API (`google-genai` SDK)
- **UI:** Streamlit local web app

---

## 1. Project structure

```
DMHardware-AI-Chatbot/
├── app.py                  # Streamlit chat UI (run this)
├── generate_dataset.py     # Builds/regenerates data/knowledge_base.csv
├── requirements.txt
├── .env.example            # Copy to .env and add your Gemini API key
├── data/
│   ├── knowledge_base.csv  # The curated knowledge base (129 starter records)
│   └── raw_sources/        # (optional) drop your own CSVs here to expand the KB
└── rag/
    ├── preprocess.py        # pandas/NumPy cleaning pipeline
    ├── retriever.py          # scikit-learn TF-IDF retrieval logic
    └── gemini_client.py      # Gemini API wrapper + prompt construction
```

## 2. Setup (VS Code, local machine)

1. **Unzip/move this folder** wherever you keep projects, then open it in VS Code
   (`File > Open Folder...`).

2. **Create a virtual environment** (VS Code terminal — View > Terminal):

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Get a Gemini API key:** https://aistudio.google.com/apikey

5. **Set your key:** copy `.env.example` to `.env` and paste your key in:

   ```bash
   cp .env.example .env       # macOS/Linux
   copy .env.example .env     # Windows
   ```

   Then edit `.env` so it reads:
   ```
   GEMINI_API_KEY=your_real_key_here
   ```

6. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   It will open in your browser at `http://localhost:8501`.

## 3. Using the chatbot

- Ask questions about electronics, robotics, PCB design, or digital logic.
- The sidebar shows how many records are in the knowledge base, lets you
  adjust the Gemini model name and how many records are retrieved per
  question, and has a debug toggle to see exactly which context was
  retrieved for a given answer.
- Chat history persists within a session; click "Clear chat" to reset.

## 4. Growing the knowledge base past 1,000+ records

The starter dataset (`data/knowledge_base.csv`) has 129 original,
hand-curated records so the pipeline works out of the box. To scale it up:

**Option A — add more curated entries directly:**
Edit `generate_dataset.py` and add more `(subcategory, title, content)`
tuples to any of the four category lists, then re-run:
```bash
python generate_dataset.py
```

**Option B — bring your own sources:**
Create CSV files in `data/raw_sources/` with columns
`category, subcategory, title, content` (one row per fact/concept — keep
each entry focused on a single idea for best retrieval quality), then
re-run `python generate_dataset.py`. It will automatically merge them in.

> Note: only add content you have the rights to use (your own notes,
> datasheets you're licensed to reference, open-licensed material, etc.) —
> avoid pasting large verbatim excerpts from copyrighted textbooks/articles.

## 5. Swapping the Gemini model

The default model is set in `rag/gemini_client.py` (`DEFAULT_MODEL`) and can
also be overridden per-session from the sidebar text box in the app. Check
https://ai.google.dev/gemini-api/docs/models for the current list of model
names available to your API key.

## 6. How retrieval works (quick summary)

1. `rag/preprocess.py` loads and cleans `knowledge_base.csv` with pandas/NumPy,
   building a combined `search_text` field per record.
2. `rag/retriever.py` fits a scikit-learn `TfidfVectorizer` over all records'
   `search_text` once at startup, then for each user question computes cosine
   similarity to find the top-k most relevant records.
3. `rag/gemini_client.py` stuffs those retrieved records into the prompt as
   `CONTEXT`, along with recent chat history, and sends it to Gemini with a
   system instruction telling it to ground its answer in that context.

This keeps everything local and dependency-light — no vector database
needed at this dataset size.
