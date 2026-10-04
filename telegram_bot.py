"""
telegram_bot.py -- a Telegram front door for Rice Disease Doctor.

It contains NO AI. It just forwards messages to your FastAPI backend over HTTP --
the same backend the website uses. "One brain, two faces":
    photo    --Telegram--> this bot --HTTP--> POST /api/predict
    question --Telegram--> this bot --HTTP--> POST /api/chat

START THE BACKEND FIRST, then run this.

SETUP (one time)
  1. Message @BotFather in Telegram -> /newbot -> copy the token.
  2. Put the token in .env as TELEGRAM_BOT_TOKEN=...
  3. pip install python-telegram-bot==21.6 requests python-dotenv
  4. python telegram_bot.py
Uses polling: no public URL / HTTPS needed. Great for a laptop demo.
"""
import os
import asyncio

import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (ApplicationBuilder, CommandHandler, ContextTypes,
                          MessageHandler, filters)

load_dotenv()  # reads .env in the folder you run this from (project root)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

STATE = {}  # chat_id -> {"lang": "en"/"km", "disease": str|None}

WELCOME = {
    "en": ("🌾 *Rice Disease Doctor*\n\n"
           "Send me a *photo* of a rice leaf and I'll diagnose it, then ask follow-up "
           "questions in plain language.\n\n"
           "/km Khmer · /en English · /help"),
    "km": ("🌾 *វេជ្ជបណ្ឌិតជំងឺស្រូវ*\n\n"
           "សូមផ្ញើ *រូបថត* ស្លឹកស្រូវ ខ្ញុំនឹងវិនិច្ឆ័យ រួចអ្នកអាចសួរសំណួរបន្ថែម។\n\n"
           "/en អង់គ្លេស · /km ខ្មែរ · /help"),
}


def _state(chat_id):
    return STATE.setdefault(chat_id, {"lang": "en", "disease": None})


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_markdown(WELCOME[_state(update.effective_chat.id)["lang"]])


async def set_en(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    _state(update.effective_chat.id)["lang"] = "en"
    await update.message.reply_text("Language set to English. Send a leaf photo or a question.")


async def set_km(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    _state(update.effective_chat.id)["lang"] = "km"
    await update.message.reply_text("ប្តូរទៅភាសាខ្មែរ។ សូមផ្ញើរូបថតស្លឹក ឬសំណួរ។")


async def handle_photo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = _state(chat_id)
    lang = st["lang"]
    await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")

    photo = update.message.photo[-1]          # largest size
    tg_file = await ctx.bot.get_file(photo.file_id)
    img_bytes = bytes(await tg_file.download_as_bytearray())

    try:
        resp = requests.post(f"{API_BASE}/api/predict",
                             files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
                             timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        await update.message.reply_text(f"Sorry, diagnosis failed: {e}")
        return

    st["disease"] = data["predicted_label"]
    treatment = data.get("treatment") or {}
    name = treatment.get(f"display_name_{lang}") or data["predicted_label"]
    advice = treatment.get(f"advice_{lang}", [])
    conf = round(data["confidence"] * 100)

    header = (f"*Diagnosis:* {name}\n*Confidence:* {conf}%\n\n" if lang == "en"
              else f"*លទ្ធផល:* {name}\n*ទំនុកចិត្ត:* {conf}%\n\n")
    advice_title = "*Recommended actions:*\n" if lang == "en" else "*វិធានការណែនាំ:*\n"
    body = "\n".join(f"• {a}" for a in advice[:5])
    tail = ("\n\n_Ask me a follow-up question any time._" if lang == "en"
            else "\n\n_អ្នកអាចសួរសំណួរបន្ថែមបានគ្រប់ពេល។_")
    await update.message.reply_markdown(header + advice_title + body + tail)


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = _state(chat_id)
    await ctx.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        resp = requests.post(f"{API_BASE}/api/chat",
                             json={"message": update.message.text, "lang": st["lang"],
                                   "session_id": f"tg-{chat_id}", "disease": st["disease"]},
                             timeout=90)
        if resp.status_code == 503:
            await update.message.reply_text(
                "Chat isn't configured on the server yet. Send a leaf photo for diagnosis instead.")
            return
        resp.raise_for_status()
        await update.message.reply_text(resp.json()["answer"])
    except Exception as e:
        await update.message.reply_text(f"Sorry, I couldn't answer that: {e}")


def main():
    if not TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in your .env (get it from @BotFather).")
    # Python 3.12+ no longer auto-creates an event loop in the main thread, and
    # python-telegram-bot's run_polling() expects one to exist. Create it ourselves.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # Generous timeouts help on slow / flaky networks (Telegram can be slow to reach).
    app = (ApplicationBuilder().token(TOKEN)
           .connect_timeout(30).read_timeout(30).get_updates_read_timeout(30)
           .build())
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("en", set_en))
    app.add_handler(CommandHandler("km", set_km))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    print("Telegram bot running (polling). Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
