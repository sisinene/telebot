# Telegram AI Bot

A lightweight Telegram chatbot backed by Groq. It stores long-term conversation memory in SQLite for each Telegram chat/user and supports `/start`, `/help`, `/memory`, and `/reset`.

## Security first

The credentials previously pasted into chat should be considered exposed. Revoke them and create new ones before running this project:

1. In Telegram, open **@BotFather**, use `/revoke` for the bot, and copy the replacement token.
2. In the Groq console, delete the exposed API key and create a new key.

Never commit `.env`; it is already ignored by Git.

The local memory database is also ignored by Git, because it can contain private user conversations.

## Run on Windows (PowerShell)

```powershell
cd "path\to\telegram-ai-bot"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python bot.py
```

Put the newly rotated credentials in `.env`, save it, then start the bot. Keep the terminal open while the bot is running.

## Configuration

- `TELEGRAM_BOT_TOKEN`: Telegram token from @BotFather (required)
- `GROQ_API_KEY`: Groq API key (required)
- `GROQ_MODEL`: model ID; defaults to `llama-3.3-70b-versatile`
- `SYSTEM_PROMPT`: controls the assistant's behavior
- `MEMORY_DB_PATH`: SQLite file path; defaults to `bot_memory.sqlite3`
- `RECENT_MEMORY_MESSAGES`: recent messages always included in context; defaults to `30`
- `RELEVANT_MEMORY_MESSAGES`: older matching messages retrieved from long-term memory; defaults to `12`
- `MEMORY_CONTEXT_CHAR_LIMIT`: rough character budget for memory passed to the model; defaults to `12000`

If Groq retires the default model, replace `GROQ_MODEL` with a model currently enabled for your Groq account.

## Memory behavior

Every text message from the user and every bot answer is saved to SQLite. On each reply, the bot sends Groq:

1. the latest conversation window, and
2. relevant older messages found in long-term memory.

This lets the bot remember earlier conversations across restarts without sending the entire database on every request. Use `/memory` to see how many messages are saved for the current chat/user. Use `/reset` to delete saved memory for the current chat/user.

## Deploy

This bot uses long polling, so it runs on any always-on Python host. Configure the same four environment variables in the host's secret manager, install `requirements.txt`, and use `python bot.py` as the start command.
