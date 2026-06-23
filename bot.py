import logging
import os
import re
import sqlite3
import asyncio
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

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
TELEGRAM_MESSAGE_LIMIT = 4000
MEMORY_DB_PATH = Path(os.getenv("MEMORY_DB_PATH", "bot_memory.sqlite3")).expanduser()
RECENT_MEMORY_MESSAGES = int(os.getenv("RECENT_MEMORY_MESSAGES", "30"))
RELEVANT_MEMORY_MESSAGES = int(os.getenv("RELEVANT_MEMORY_MESSAGES", "12"))
MEMORY_CONTEXT_CHAR_LIMIT = int(os.getenv("MEMORY_CONTEXT_CHAR_LIMIT", "12000"))
REASONING_CHAINS = int(os.getenv("REASONING_CHAINS", "3"))
MAX_REASONING_CHAINS = int(os.getenv("MAX_REASONING_CHAINS", "5"))
REASONING_DRAFT_TOKENS = int(os.getenv("REASONING_DRAFT_TOKENS", "900"))
REASONING_FINAL_TOKENS = int(os.getenv("REASONING_FINAL_TOKENS", "1500"))

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _memory_db_path(db_path: Path | str = MEMORY_DB_PATH) -> Path:
    return Path(db_path).expanduser()


def init_memory_db(db_path: Path | str = MEMORY_DB_PATH) -> None:
    path = _memory_db_path(db_path)
    if path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_chat_user_id
            ON messages (chat_id, user_id, id)
            """
        )


def save_memory(
    chat_id: int,
    user_id: int,
    role: str,
    content: str,
    db_path: Path | str = MEMORY_DB_PATH,
) -> None:
    with closing(sqlite3.connect(_memory_db_path(db_path))) as conn, conn:
        conn.execute(
            """
            INSERT INTO messages (chat_id, user_id, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (chat_id, user_id, role, content),
        )


def clear_memory(chat_id: int, user_id: int, db_path: Path | str = MEMORY_DB_PATH) -> int:
    with closing(sqlite3.connect(_memory_db_path(db_path))) as conn, conn:
        cursor = conn.execute(
            "DELETE FROM messages WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id),
        )
        return cursor.rowcount


def count_memory(chat_id: int, user_id: int, db_path: Path | str = MEMORY_DB_PATH) -> int:
    with closing(sqlite3.connect(_memory_db_path(db_path))) as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND user_id = ?",
            (chat_id, user_id),
        )
        return int(cursor.fetchone()[0])


def get_recent_memory(
    chat_id: int,
    user_id: int,
    limit: int = RECENT_MEMORY_MESSAGES,
    db_path: Path | str = MEMORY_DB_PATH,
) -> list[dict[str, str | int]]:
    with closing(sqlite3.connect(_memory_db_path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, role, content
            FROM messages
            WHERE chat_id = ? AND user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (chat_id, user_id, limit),
        ).fetchall()
    return [
        {"id": row["id"], "role": row["role"], "content": row["content"]}
        for row in reversed(rows)
    ]


def _keywords(text: str) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for word in re.findall(r"[A-Za-z0-9_]{3,}", text.lower()):
        if word not in seen:
            seen.add(word)
            words.append(word)
    return words[:10]


def get_relevant_memory(
    chat_id: int,
    user_id: int,
    query: str,
    excluded_ids: set[int] | None = None,
    limit: int = RELEVANT_MEMORY_MESSAGES,
    db_path: Path | str = MEMORY_DB_PATH,
) -> list[dict[str, str | int]]:
    words = _keywords(query)
    if not words:
        return []

    clauses = " OR ".join("LOWER(content) LIKE ?" for _ in words)
    params: list[object] = [chat_id, user_id, *(f"%{word}%" for word in words)]
    with closing(sqlite3.connect(_memory_db_path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, role, content
            FROM messages
            WHERE chat_id = ? AND user_id = ? AND ({clauses})
            ORDER BY id DESC
            LIMIT ?
            """,
            [*params, limit * 5],
        ).fetchall()

    excluded_ids = excluded_ids or set()
    scored: list[tuple[int, int, sqlite3.Row]] = []
    for row in rows:
        if row["id"] in excluded_ids:
            continue
        content = row["content"].lower()
        score = sum(1 for word in words if word in content)
        scored.append((score, row["id"], row))

    best_rows = [row for _, _, row in sorted(scored, reverse=True)[:limit]]
    best_rows.sort(key=lambda row: row["id"])
    return [
        {"id": row["id"], "role": row["role"], "content": row["content"]}
        for row in best_rows
    ]


def trim_memory_context(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    kept: list[dict[str, str]] = []
    used_chars = 0
    for message in reversed(messages):
        message_chars = len(message["content"])
        if kept and used_chars + message_chars > MEMORY_CONTEXT_CHAR_LIMIT:
            break
        kept.append({"role": message["role"], "content": message["content"]})
        used_chars += message_chars
    return list(reversed(kept))


def build_memory_context(
    chat_id: int,
    user_id: int,
    query: str,
    db_path: Path | str = MEMORY_DB_PATH,
) -> list[dict[str, str]]:
    recent = get_recent_memory(chat_id, user_id, db_path=db_path)
    recent_ids = {int(message["id"]) for message in recent}
    relevant = get_relevant_memory(
        chat_id,
        user_id,
        query,
        excluded_ids=recent_ids,
        db_path=db_path,
    )

    messages: list[dict[str, str]] = []
    if relevant:
        memories = "\n".join(
            f"- {message['role']}: {message['content']}" for message in relevant
        )
        messages.append(
            {
                "role": "system",
                "content": (
                    "Long-term memory from earlier conversations with this user. "
                    "Use it when relevant, but do not mention this memory block unless asked.\n"
                    f"{memories}"
                ),
            }
        )
    messages.extend(
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in recent
    )
    return trim_memory_context(messages)


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


def active_reasoning_chains() -> int:
    """Clamp chain count so config mistakes do not create runaway API calls."""
    return max(1, min(REASONING_CHAINS, MAX_REASONING_CHAINS))


def build_chain_messages(messages: Sequence[dict[str, str]], chain_index: int) -> list[dict[str, str]]:
    perspectives = [
        "Focus on the most direct and practical answer.",
        "Check assumptions, edge cases, and hidden constraints before answering.",
        "Look for a simpler explanation or alternative approach.",
        "Prioritize safety, privacy, and operational reliability.",
        "Use the user's saved context and preferences where relevant.",
    ]
    perspective = perspectives[(chain_index - 1) % len(perspectives)]
    return [
        *messages,
        {
            "role": "system",
            "content": (
                f"Candidate reasoning chain {chain_index}: {perspective} "
                "Think carefully, but do not reveal private reasoning or scratchpad. "
                "Return only the best answer draft."
            ),
        },
    ]


def build_synthesis_messages(
    messages: Sequence[dict[str, str]],
    candidate_answers: Sequence[str],
) -> list[dict[str, str]]:
    candidate_block = "\n\n".join(
        f"Candidate {index}:\n{answer}" for index, answer in enumerate(candidate_answers, start=1)
    )
    return [
        *messages,
        {
            "role": "system",
            "content": (
                "Synthesize the candidate answers into one final Telegram reply. "
                "Prefer correctness, clarity, and usefulness. Resolve conflicts silently. "
                "Do not mention candidates, chains, hidden reasoning, or scratchpad."
            ),
        },
        {
            "role": "user",
            "content": f"Candidate answers to synthesize:\n\n{candidate_block}",
        },
    ]


async def call_groq(
    messages: Sequence[dict[str, str]],
    *,
    temperature: float,
    max_completion_tokens: int,
) -> str:
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
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


async def ask_groq(messages: Sequence[dict[str, str]]) -> str:
    chains = active_reasoning_chains()
    if chains == 1:
        return await call_groq(
            messages,
            temperature=0.7,
            max_completion_tokens=REASONING_FINAL_TOKENS,
        )

    draft_tasks = [
        call_groq(
            build_chain_messages(messages, chain_index),
            temperature=0.55 + (chain_index * 0.08),
            max_completion_tokens=REASONING_DRAFT_TOKENS,
        )
        for chain_index in range(1, chains + 1)
    ]
    candidate_answers = await asyncio.gather(*draft_tasks)
    return await call_groq(
        build_synthesis_messages(messages, candidate_answers),
        temperature=0.35,
        max_completion_tokens=REASONING_FINAL_TOKENS,
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Hi! Send me a message and I'll answer with AI. I keep long-term memory "
            "for this chat, so I can remember earlier conversations after restarts.\n\n"
            "Commands:\n/reset - clear saved memory\n/memory - show saved memory count\n"
            "/reasoning - show reasoning mode\n/help - show help"
        )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_user and update.message:
        deleted = clear_memory(update.effective_chat.id, update.effective_user.id)
        await update.message.reply_text(f"Memory cleared. Removed {deleted} saved messages.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Just send a text message. I save our conversation in long-term memory and "
            "use recent plus relevant older messages when answering. For harder replies, "
            "I create multiple private answer drafts and synthesize one final response. "
            "Use /memory to see how many messages are saved, or /reset to delete your saved memory."
        )


async def memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat and update.effective_user and update.message:
        saved_messages = count_memory(update.effective_chat.id, update.effective_user.id)
        await update.message.reply_text(f"I have {saved_messages} saved messages for this chat.")


async def reasoning_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        chains = active_reasoning_chains()
        if chains == 1:
            await update.message.reply_text("Reasoning mode: single-pass replies.")
        else:
            await update.message.reply_text(
                f"Reasoning mode: {chains} private answer drafts, then one synthesized final reply."
            )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    if not update.effective_chat or not update.effective_user:
        return

    user_text = update.message.text.strip()
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    save_memory(chat_id, user_id, "user", user_text)
    messages = build_memory_context(chat_id, user_id, user_text)

    await update.message.chat.send_action(ChatAction.TYPING)
    try:
        answer = await ask_groq(messages)
    except httpx.HTTPStatusError as exc:
        logger.error("Groq returned HTTP %s", exc.response.status_code)
        await update.message.reply_text(
            "The AI service rejected the request. Check the API key and model, then try again."
        )
        return
    except (httpx.HTTPError, KeyError, IndexError, TypeError):
        logger.exception("Failed to get a valid response from Groq")
        await update.message.reply_text("I couldn't reach the AI service. Please try again shortly.")
        return

    save_memory(chat_id, user_id, "assistant", answer)
    for chunk in split_message(answer):
        await update.message.reply_text(chunk)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled Telegram bot error", exc_info=context.error)


def main() -> None:
    require_config()
    init_memory_db()
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("memory", memory_command))
    app.add_handler(CommandHandler("reasoning", reasoning_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    logger.info(
        "Bot started with model %s, memory database %s, and %s reasoning chain(s)",
        GROQ_MODEL,
        MEMORY_DB_PATH,
        active_reasoning_chains(),
    )
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
