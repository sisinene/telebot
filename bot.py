import logging
import os
from collections.abc import Sequence

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise AI assistant in a Telegram chat.",
).strip()
MAX_HISTORY_MESSAGES = 20
TELEGRAM_MESSAGE_LIMIT = 4000

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def require_config() -> None:
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def split_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split long replies at sensible boundaries for Telegram."""
    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return chunks or ["I couldn't produce a response."]


async def ask_groq(messages: Sequence[dict[str, str]]) -> str:
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "temperature": 0.7,
        "max_completion_tokens": 1500,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
    return data["choices"][0]["message"]["content"].strip()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Hi! Send me a message and I'll answer with AI.\n\n"
            "Commands:\n/reset — clear our conversation\n/help — show help"
        )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["history"] = []
    if update.message:
        await update.message.reply_text("Conversation cleared.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Just send a text message. I remember the latest part of this chat. "
            "Use /reset whenever you want a fresh conversation."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    user_text = update.message.text.strip()
    history: list[dict[str, str]] = context.user_data.setdefault("history", [])
    history.append({"role": "user", "content": user_text})
    history[:] = history[-MAX_HISTORY_MESSAGES:]

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        answer = await ask_groq(history)
    except httpx.HTTPStatusError as exc:
        logger.error("Groq returned HTTP %s", exc.response.status_code)
        history.pop()
        await update.message.reply_text(
            "The AI service rejected the request. Check the API key and model, then try again."
        )
        return
    except (httpx.HTTPError, KeyError, IndexError, TypeError):
        logger.exception("Failed to get a valid response from Groq")
        history.pop()
        await update.message.reply_text("I couldn't reach the AI service. Please try again shortly.")
        return

    history.append({"role": "assistant", "content": answer})
    history[:] = history[-MAX_HISTORY_MESSAGES:]
    for chunk in split_message(answer):
        await update.message.reply_text(chunk)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled Telegram bot error", exc_info=context.error)


def main() -> None:
    require_config()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    logger.info("Bot started with model %s", GROQ_MODEL)
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
