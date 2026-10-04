"""
rag.py -- Retrieval-Augmented Generation (RAG) layer for follow-up questions.

WHAT IT DOES (plain language)
-----------------------------
treatment_db.py already holds vetted advice for all 10 classes. This turns that into a
small searchable knowledge base and answers free-text questions by:
    1. RETRIEVE  -- find the advice snippets most relevant to the question.
    2. GENERATE  -- ask an LLM to answer USING ONLY those snippets (so it can't invent
                    a wrong pesticide or dose; if the answer isn't there it says so).

HOSTED OR LOCAL -- chosen by environment variables (see .env.example), no code change:
  * Local & free (Ollama):  LLM_BASE_URL=http://localhost:11434/v1
  * Hosted (OpenAI):        LLM_BASE_URL=https://api.openai.com/v1  + LLM_API_KEY

This file does NOT touch the image model. It is a separate, additive layer.
"""
import os

# --- Auto-load .env so you never need `source .env` before starting the server. ---
# We look next to this file and one directory up (project root), which covers both
# backend/ (local) and backend-deploy/ (Docker) layouts. On hosts like Render there
# is no .env and the variables come from the dashboard instead -- that works too.
try:
    from dotenv import load_dotenv
    _here = os.path.dirname(os.path.abspath(__file__))
    for _candidate in (os.path.join(_here, ".env"), os.path.join(_here, "..", ".env")):
        if os.path.exists(_candidate):
            load_dotenv(_candidate)
            break
except Exception:
    pass  # dotenv is optional; env vars may already be set in the shell / dashboard

import hashlib
import json

import numpy as np
import requests

from treatment_db import TREATMENT_DB

# ---------------------------------------------------------------------------
# Configuration (read AFTER .env is loaded above).
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
REQUEST_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))

_CACHE_PATH = os.path.join(os.path.dirname(__file__), "rag_index_cache.json")

_CHUNKS = []
_EMBEDDINGS = None
_READY = False


class RAGNotReady(Exception):
    pass


# ---------------------------------------------------------------------------
# 1. Build the corpus from treatment_db.
# ---------------------------------------------------------------------------
def _build_corpus():
    chunks = []
    for disease, info in TREATMENT_DB.items():
        for lang in ("en", "km"):
            name = info.get(f"display_name_{lang}", disease)
            category = info.get("category", "")
            severity = info.get("severity", "")
            header = (f"{name} -- category: {category}; severity: {severity}."
                      if lang == "en" else
                      f"{name} -- ប្រភេទ៖ {category}; កម្រិត៖ {severity}។")
            chunks.append({"text": header, "disease": disease, "lang": lang,
                           "category": category, "severity": severity})
            for item in info.get(f"advice_{lang}", []):
                chunks.append({"text": f"{name}: {item}", "disease": disease,
                               "lang": lang, "category": category, "severity": severity})
    return chunks


def _corpus_fingerprint(chunks):
    h = hashlib.sha256()
    h.update(EMBED_MODEL.encode())
    for c in chunks:
        h.update(c["text"].encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 2. OpenAI-compatible embedding + chat calls.
# ---------------------------------------------------------------------------
def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _embed(texts):
    resp = requests.post(
        f"{BASE_URL}/embeddings",
        headers=_headers(),
        json={"model": EMBED_MODEL, "input": texts},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    vectors = [item["embedding"] for item in sorted(data, key=lambda d: d["index"])]
    return np.array(vectors, dtype=np.float32)


def _chat(messages):
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=_headers(),
        json={"model": CHAT_MODEL, "messages": messages, "temperature": 0.2},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# 3. Build / load the index. Call once at startup.
# ---------------------------------------------------------------------------
def build_index(force=False):
    global _CHUNKS, _EMBEDDINGS, _READY
    corpus = _build_corpus()
    fingerprint = _corpus_fingerprint(corpus)

    if not force and os.path.exists(_CACHE_PATH):
        try:
            cached = json.load(open(_CACHE_PATH, encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                _CHUNKS = cached["chunks"]
                _EMBEDDINGS = np.array(cached["embeddings"], dtype=np.float32)
                _READY = True
                print(f"[rag] loaded {len(_CHUNKS)} chunks from cache")
                return
        except Exception as e:
            print(f"[rag] cache unreadable, rebuilding: {e}")

    print(f"[rag] building index for {len(corpus)} chunks via {BASE_URL} ...")
    embeddings = _embed([c["text"] for c in corpus])
    _CHUNKS = corpus
    _EMBEDDINGS = embeddings
    _READY = True
    try:
        json.dump({"fingerprint": fingerprint, "chunks": corpus,
                   "embeddings": embeddings.tolist()},
                  open(_CACHE_PATH, "w", encoding="utf-8"))
        print(f"[rag] index built and cached at {_CACHE_PATH}")
    except Exception as e:
        print(f"[rag] built index but could not write cache: {e}")


def is_ready():
    return _READY


# ---------------------------------------------------------------------------
# 4. Retrieve.
# ---------------------------------------------------------------------------
def _cosine_scores(query_vec, matrix):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return m @ q


def retrieve(question, lang="en", disease=None, k=4):
    if not _READY:
        raise RAGNotReady("RAG index is not built.")
    qvec = _embed([question])[0]
    scores = _cosine_scores(qvec, _EMBEDDINGS)
    results = []
    for i, chunk in enumerate(_CHUNKS):
        if chunk["lang"] != lang:
            continue
        score = float(scores[i])
        if disease and chunk["disease"] == disease:
            score += 0.15
        results.append((score, chunk))
    results.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in results[:k]]


# ---------------------------------------------------------------------------
# 5. Generate a grounded answer.
# ---------------------------------------------------------------------------
_SYSTEM = {
    "en": (
        "You are a helpful assistant for Cambodian rice farmers. Use the reference "
        "snippets as your primary source for treatment and management advice. You MAY "
        "use your own general knowledge to explain or define terms (e.g. what a pesticide "
        "or a disease is) and to add helpful context. However, for specific pesticide "
        "DOSES, concentrations, or application rates, rely only on the snippets — if a "
        "specific dose or rate is not given, do not invent one; instead advise following "
        "local label instructions or asking an extension officer. Keep answers short, "
        "practical, and in English. End by reminding the farmer this is general guidance, "
        "not a replacement for an extension officer."
    ),
    "km": (
        "អ្នកគឺជាជំនួយការសម្រាប់កសិករដាំស្រូវកម្ពុជា។ ប្រើព័ត៌មានយោងជាប្រភពចម្បងសម្រាប់ការណែនាំ "
        "ព្យាបាល និងគ្រប់គ្រង។ អ្នកអាចប្រើចំណេះដឹងទូទៅរបស់អ្នកដើម្បីពន្យល់ ឬនិយមន័យពាក្យ (ឧ. "
        "ថ្នាំសម្លាប់សត្វល្អិត ឬជំងឺជាអ្វី) និងបន្ថែមបរិបទមានប្រយោជន៍។ ប៉ុន្តែសម្រាប់ *កម្រិត* ឬ "
        "អត្រាប្រើថ្នាំជាក់លាក់ សូមផ្អែកលើព័ត៌មានយោងតែប៉ុណ្ណោះ បើគ្មានកម្រិតជាក់លាក់ កុំបង្កើតដោយខ្លួនឯង "
        "គួរណែនាំឱ្យធ្វើតាមស្លាកថ្នាំ ឬសួរមន្ត្រីកសិកម្ម។ ចម្លើយខ្លី ជាក់ស្តែង ជាភាសាខ្មែរ។ ចុងក្រោយ "
        "រំលឹកថានេះជាការណែនាំទូទៅ មិនមែនជំនួសមន្ត្រីកសិកម្មទេ។"
    ),
}


def answer(question, lang="en", disease=None, history=None, k=4):
    if not _READY:
        raise RAGNotReady("RAG index is not built.")
    chunks = retrieve(question, lang=lang, disease=disease, k=k)
    context = "\n".join(f"- {c['text']}" for c in chunks)
    label = "Reference snippets" if lang == "en" else "ព័ត៌មានយោង"
    messages = [{"role": "system", "content": _SYSTEM.get(lang, _SYSTEM["en"])}]
    if history:
        messages += history[-6:]
    messages.append({"role": "user",
                     "content": f"{label}:\n{context}\n\nQuestion: {question}"})
    reply = _chat(messages)
    return {"answer": reply, "sources": [c["text"] for c in chunks], "used_disease": disease}
