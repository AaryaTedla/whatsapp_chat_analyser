# WhatsApp Chat Analyser

A Flask web app that analyzes WhatsApp exported `.txt` chats and shows:
- message statistics and charts
- AI vibe summary (with safe fallback)
- saved report history with downloadable TXT

Live demo: `https://whatsapp-chat-analyser-d6zh.onrender.com`

## Features

- Upload WhatsApp export (`.txt`)
- Parse sender/time/message lines
- Visualize:
  - messages per sender
  - messages per hour
  - messages over time
  - top words
- AI summary:
  - overall vibe
  - top recurring topics
  - funny observation
- Saved reports:
  - list all reports
  - open report view
  - download original TXT

## Tech Stack

- Python 3
- Flask
- Pandas
- Chart.js
- OpenRouter (via OpenAI SDK)
- SQLite

## Local Setup

```bash
cd parent_folder
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Open `http://127.0.0.1:5000`

## Environment Variables

Create `.env` with:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemma-3-4b-it:free
# Optional override:
# DB_PATH=/tmp/lumira_reports.db
```

If `OPENROUTER_MODEL` is not set, the app defaults to `google/gemma-3-4b-it:free`.

### AI Summary Behavior

- The app first attempts a structured AI JSON summary.
- If the model output is invalid/placeholder-like, it retries once.
- If still unusable, it falls back to a local summary heuristic.
- Terminal logs show which path was used:
  - `REAL AI SUMMARY RETURNED`
  - `USING LOCAL FALLBACK ...`

## Render Deployment

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Env vars:
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_MODEL`
  - `DB_PATH=/tmp/lumira_reports.db` (default/free-tier-friendly)

### Free Tier Note

Render free instances do not provide persistent disk. Reports stored in SQLite may reset after restart/redeploy.
This is expected when using `/tmp/lumira_reports.db`.

## Routes

- `/` - main app
- `/reports` - saved reports page
- `/reports?format=json` - reports API list
- `/reports/<id>` - report JSON
- `/r/<id>` - report HTML view
- `/reports/<id>/view` - legacy redirect to `/r/<id>`
- `/reports/<id>/txt` - download original TXT

## Demo Test File

Use `temp_whatsapp_export.txt` for a quick test.
