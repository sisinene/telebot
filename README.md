# Telegram AI Bot

A lightweight Telegram chatbot backed by Groq. It keeps a short, separate conversation history for each Telegram user and supports `/start`, `/help`, and `/reset`.

## Security first

The credentials previously pasted into chat should be considered exposed. Revoke them and create new ones before running this project:

1. In Telegram, open **@BotFather**, use `/revoke` for the bot, and copy the replacement token.
2. In the Groq console, delete the exposed API key and create a new key.

Never commit `.env`; it is already ignored by Git.

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

If Groq retires the default model, replace `GROQ_MODEL` with a model currently enabled for your Groq account.

## Deploy

This bot uses long polling, so it runs on any always-on Python host. Configure the same four environment variables in the host's secret manager, install `requirements.txt`, and use `python bot.py` as the start command.
