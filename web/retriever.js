// JS port of the query side of rag/retriever.py. The vocabulary, idf weights
// and normalized document vectors come from export_index.py, so this only has
// to vectorize the question the same way sklearn's TfidfVectorizer does:
// lowercase -> tokenize + stem -> drop stop words -> 1-2 grams -> tf * idf -> L2.
// No browser APIs here, so check-parity.mjs can run it under Node in CI.

const WORD_RE = /[a-zA-Z][a-zA-Z0-9']+/g;
const SUFFIXES = [["ies", "y"], ["sses", "ss"], ["ing", ""], ["ed", ""], ["es", ""], ["s", ""]];

export function simpleStem(word) {
  const w = word.toLowerCase();
  for (const [suffix, replacement] of SUFFIXES) {
    if (w.endsWith(suffix) && w.length - suffix.length >= 3) return w.slice(0, -suffix.length) + replacement;
  }
  return w;
}

export function analyze(text, stopWords) {
  const tokens = (text.toLowerCase().match(WORD_RE) || []).map(simpleStem).filter((t) => !stopWords.has(t));
  const grams = tokens.slice();
  for (let i = 0; i + 1 < tokens.length; i++) grams.push(tokens[i] + " " + tokens[i + 1]);
  return grams;
}

export function createRetriever(index) {
  const stopWords = new Set(index.stop_words);
  const vocab = new Map(Object.entries(index.vocab));

  function retrieve(query, topK = 5, minScore = 0.05) {
    if (!query || !query.trim()) return [];
    const counts = new Map();
    for (const gram of analyze(query, stopWords)) {
      const j = vocab.get(gram);
      if (j !== undefined) counts.set(j, (counts.get(j) || 0) + 1);
    }
    const q = new Map();
    let norm = 0;
    for (const [j, c] of counts) {
      const v = c * index.idf[j];
      q.set(j, v);
      norm += v * v;
    }
    norm = Math.sqrt(norm);
    if (!norm) return [];

    // document rows are stored flat: [termIndex, weight, termIndex, weight, ...]
    const scored = index.docs.map((row, i) => {
      let s = 0;
      for (let k = 0; k < row.length; k += 2) {
        const v = q.get(row[k]);
        if (v) s += v * row[k + 1];
      }
      return [s / norm, i];
    });
    scored.sort((a, b) => b[0] - a[0]);
    return scored
      .slice(0, topK)
      .filter(([score]) => score >= minScore)
      .map(([score, i]) => ({ ...index.records[i], score }));
  }

  return { size: index.records.length, retrieve };
}

// same format as KnowledgeRetriever.format_context
export function formatContext(records) {
  if (!records.length) return "No directly relevant records were found in the knowledge base.";
  return records.map((r) => "[" + r.category + " / " + r.subcategory + "] " + r.title + ": " + r.content).join("\n");
}
