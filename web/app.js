// Browser version of app.py: same retrieval -> grounded prompt -> Gemini flow,
// with the Streamlit widgets rebuilt as plain DOM. A visitor's own Gemini key
// is kept in localStorage and sent only to Google; without one, the bot shows
// the knowledge-base records the retriever matched.
import { createRetriever, formatContext } from "./retriever.js";

// copied from rag/gemini_client.py
const SYSTEM_INSTRUCTION =
  "You are DMHardware AI Chatbot, a domain expert assistant for " +
  "Electronics, Robotics, PCB Design, and Digital Logic Design.\n\n" +
  "Ground your answers primarily in the CONTEXT provided below, which comes from a " +
  "curated hardware-engineering knowledge base. If the context doesn't fully cover " +
  "the question, you may use your own general engineering knowledge to fill gaps, " +
  "but say so explicitly when you go beyond the provided context.\n\n" +
  "Be precise, concise, and practical — this tool is used by someone doing real " +
  "electronics/robotics/PCB/digital-logic work. Use component names, units, and " +
  "correct terminology. If a question is ambiguous, ask a brief clarifying question.";

const KEY_STORAGE = "dmhardware.geminiApiKey";
const SUGGESTIONS = [
  "How do I pick a resistor for an LED?",
  "What is a PCB via, and when do I need one?",
  "Explain how a JK flip-flop works",
  "How does an H-bridge drive a DC motor?",
];

const $ = (id) => document.getElementById(id);
const state = { messages: [], retriever: null, busy: false };

const getKey = () => {
  try {
    return localStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
};
const setKey = (value) => {
  try {
    if (value) localStorage.setItem(KEY_STORAGE, value.trim());
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable */
  }
};

// same shape as build_prompt in rag/gemini_client.py
function buildPrompt(context, history, userMessage) {
  let historyText = "";
  for (const turn of history.slice(-6)) {
    historyText += (turn.role === "user" ? "User" : "Assistant") + ": " + turn.content + "\n";
  }
  return (
    "CONTEXT (retrieved from knowledge base):\n" + context + "\n\n" +
    "CONVERSATION SO FAR:\n" + historyText + "\n\n" +
    "User's new question: " + userMessage + "\n\n" +
    "Answer the user's question now, grounded in the CONTEXT above where relevant."
  );
}

async function generateResponse(prompt) {
  const model = $("model").value.trim() || "gemini-2.5-flash";
  const res = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/models/" + encodeURIComponent(model) + ":generateContent",
    {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": getKey() },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: SYSTEM_INSTRUCTION }] },
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.4 },
      }),
    }
  );
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || "HTTP " + res.status);
  const text = (data?.candidates?.[0]?.content?.parts || []).map((p) => p.text || "").join("");
  if (!text) throw new Error("Gemini returned an empty response");
  return text;
}

function offlineAnswer(records) {
  const tail = "\n\n_Add a Gemini API key in Settings to get a full, grounded answer._";
  if (!records.length) return "**Offline mode.** No directly relevant records were found in the knowledge base." + tail;
  return (
    "**Offline mode.** No Gemini key is set, so here are the knowledge-base records the retriever matched:\n\n" +
    records
      .map((r) => "- **" + r.title + "** _(" + r.category + " / " + r.subcategory + ", score " + r.score.toFixed(2) + ")_: " + r.content)
      .join("\n") +
    tail
  );
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function renderMarkdown(text) {
  if (window.marked && window.DOMPurify) return window.DOMPurify.sanitize(window.marked.parse(text));
  return "<p>" + escapeHtml(text).replace(/\n/g, "<br>") + "</p>";
}

function addMessage(role, content, { context, pending } = {}) {
  const el = document.createElement("div");
  el.className = "msg msg-" + role + (pending ? " pending" : "");
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "🧑" : "🔧";
  const body = document.createElement("div");
  body.className = "body";
  if (context) {
    const details = document.createElement("details");
    details.innerHTML = "<summary>Retrieved context</summary>";
    const pre = document.createElement("pre");
    pre.textContent = context;
    details.appendChild(pre);
    body.appendChild(details);
  }
  const text = document.createElement("div");
  text.className = "text";
  if (pending) text.textContent = "Thinking...";
  else if (role === "user") text.textContent = content;
  else text.innerHTML = renderMarkdown(content);
  body.appendChild(text);
  el.append(avatar, body);
  $("messages").appendChild(el);
  el.scrollIntoView({ block: "end", behavior: "smooth" });
  return { el, text };
}

async function send(userMessage) {
  userMessage = userMessage.trim();
  if (!userMessage || state.busy || !state.retriever) return;
  state.busy = true;
  $("suggestions").hidden = true;
  $("chat-input").value = "";

  const history = state.messages.slice();
  state.messages.push({ role: "user", content: userMessage });
  addMessage("user", userMessage);

  const records = state.retriever.retrieve(userMessage, Number($("top-k").value));
  const context = formatContext(records);
  const pending = addMessage("assistant", "", { pending: true, context: $("show-context").checked ? context : null });

  let answer;
  if (!getKey()) {
    answer = offlineAnswer(records);
  } else {
    try {
      answer = await generateResponse(buildPrompt(context, history, userMessage));
    } catch (e) {
      answer = "Something went wrong calling Gemini: " + e.message;
    }
  }
  pending.el.classList.remove("pending");
  pending.text.innerHTML = renderMarkdown(answer);
  state.messages.push({ role: "assistant", content: answer });
  state.busy = false;
  $("chat-input").focus();
}

function renderKeyState() {
  const key = getKey();
  $("key-status").className = "status " + (key ? "ok" : "warn");
  $("key-status").textContent = key ? "Gemini key saved in this browser ✅" : "No Gemini key: running in offline retrieval mode.";
  $("key-form").hidden = Boolean(key);
  $("key-clear").hidden = !key;
}

function wireUi() {
  $("top-k").addEventListener("input", (e) => ($("top-k-value").textContent = e.target.value));
  $("key-form").addEventListener("submit", (e) => {
    e.preventDefault();
    setKey($("key-input").value);
    $("key-input").value = "";
    renderKeyState();
  });
  $("key-clear").addEventListener("click", () => {
    setKey("");
    renderKeyState();
  });
  $("clear-chat").addEventListener("click", () => {
    state.messages = [];
    $("messages").innerHTML = "";
    $("suggestions").hidden = false;
  });
  $("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    send($("chat-input").value);
  });
  $("menu").addEventListener("click", () => {
    const open = document.body.classList.toggle("sidebar-open");
    $("menu").setAttribute("aria-expanded", String(open));
  });
  for (const q of SUGGESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = q;
    chip.addEventListener("click", () => send(q));
    $("suggestions").appendChild(chip);
  }
  renderKeyState();
}

async function main() {
  wireUi();
  try {
    const index = await fetch("index.json").then((r) => {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
    state.retriever = createRetriever(index);
    $("kb-size").textContent = state.retriever.size.toLocaleString();
  } catch (e) {
    $("kb-size").textContent = "?";
    addMessage("assistant", "Could not load the knowledge base index (" + e.message + ").");
  }
}

main();
