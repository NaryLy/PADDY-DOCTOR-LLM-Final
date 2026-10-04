# What changed in this version (RAG chat + Telegram bot)

This update adds an AI follow-up chat and a Telegram bot, without touching your
training code, model, or existing prediction/history features. Design: **one brain
(the FastAPI backend), three faces (website, Telegram, and the deployed API).**

## New / changed files
| File | Change |
|---|---|
| `backend/rag.py` | **NEW** — RAG layer: builds a searchable knowledge base from `treatment_db.py`, retrieves relevant advice, asks an LLM to answer using only that (grounded). Auto-loads `.env`. |
| `backend/app.py` | Added `POST /api/chat`, in-memory chat memory, `chat_ready` in `/api/health`, RAG build at startup. |
| `backend-deploy/rag.py` | **NEW** — same RAG layer for the deployed API. |
| `backend-deploy/app.py` | Same `/api/chat` additions as the local backend. |
| `backend-deploy/Dockerfile` | Now copies `rag.py` into the image. |
| `backend-deploy/requirements.txt` | Added `numpy`, `requests`, `python-dotenv`. |
| `frontend/index.html` | Added a follow-up chat box. It **auto-hides** if the backend has no LLM configured (so your Netlify site won't show a broken box). |
| `telegram_bot.py` | **NEW** — Telegram bot (photo → diagnosis, text → chat). Talks to the backend over HTTP. |
| `.env.example` | **NEW** — copy to `.env`. Defaults to Ollama. |
| `requirements.txt` | Added `python-dotenv`, `python-telegram-bot`. |
| `models/finetuned_best.pt` | Copied here so the **local** backend runs out of the box. |

## One-time setup
    pip install -r requirements.txt
    cp .env.example .env            # defaults to Ollama; edit if you want OpenAI
    # for Ollama chat:
    ollama pull llama3.1
    ollama pull nomic-embed-text

## Run locally (the backend now reads .env by itself — no `source .env` needed)
    # terminal 1 — make sure Ollama is running (open the Ollama app, or `ollama serve`)
    # terminal 2 — backend + website
    cd backend && uvicorn app:app --host 0.0.0.0 --port 8000
    #   open http://localhost:8000/   (chat box appears when chat_ready is true)
    # terminal 3 — telegram bot
    python telegram_bot.py

## Verify
    curl http://localhost:8000/api/health
    # want: "model_loaded": true  AND  "chat_ready": true

## Startup order for your demo (important)
Ollama running → backend → telegram bot. If the backend starts before Ollama is up,
the chat index fails to build and chat stays disabled until you restart the backend.

## Note about the deployed version (Render + Netlify)
Ollama can't run on Render. Two options for the live site's chat:
  * Leave chat off in production — the box auto-hides, diagnosis still works. (Simplest.)
  * Or set OpenAI env vars (`LLM_BASE_URL`, `LLM_API_KEY`, `CHAT_MODEL`, `EMBED_MODEL`)
    in the Render dashboard to enable chat in the cloud too.

## Python version
Tested target: Python 3.10–3.12 (the Docker image uses 3.10). You're on 3.14 locally;
the Telegram event-loop fix is already included. If torch/other libs ever misbehave on
3.14, recreate your venv on 3.12.
