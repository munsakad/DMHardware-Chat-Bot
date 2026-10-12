// CI check: the JS retriever must return the same top-k records (and scores)
// as the Python retriever for the sample questions in parity.json.
// Usage: node web/check-parity.mjs <dir containing index.json and parity.json>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createRetriever } from "./retriever.js";

const dir = process.argv[2] || "web";
const index = JSON.parse(readFileSync(join(dir, "index.json"), "utf8"));
const parity = JSON.parse(readFileSync(join(dir, "parity.json"), "utf8"));
const retriever = createRetriever(index);

let failures = 0;
for (const { q, top } of parity) {
  const js = retriever.retrieve(q, 5);
  const sameLength = js.length === top.length;
  const ok =
    sameLength &&
    top.every((p, i) => {
      const r = js[i];
      // allow equal-score ties to come back in either order
      const title = r.title === p.title || Math.abs(r.score - p.score) < 1e-6;
      return title && Math.abs(r.score - p.score) < 1e-4;
    });
  console.log((ok ? "ok   " : "FAIL ") + q + " -> " + js.map((r) => r.title + " (" + r.score.toFixed(3) + ")").join("; "));
  if (!ok) {
    failures += 1;
    console.log("     python: " + top.map((r) => r.title + " (" + r.score.toFixed(3) + ")").join("; "));
  }
}

if (failures) {
  console.error(failures + " question(s) diverged from the Python retriever");
  process.exit(1);
}
console.log("JS retriever matches Python on all " + parity.length + " questions");
