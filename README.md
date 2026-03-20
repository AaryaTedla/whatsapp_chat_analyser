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
cd lumira_prep
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
OPENROUTER_MODEL=stepfun/step-3.5-flash:free
# Optional override:
# DB_PATH=/tmp/lumira_reports.db
```

## Render Deployment

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Env vars:
  - `OPENROUTER_API_KEY`
  - `OPENROUTER_MODEL`
  - `DB_PATH=/tmp/lumira_reports.db` (free tier)

### Free Tier Note

Render free instances do not provide persistent disk. Reports stored in SQLite may reset after restart/redeploy.

## Routes

- `/` - main app
- `/reports` - saved reports page
- `/reports?format=json` - reports API list
- `/reports/<id>` - report JSON
- `/r/<id>` - report HTML view
- `/reports/<id>/txt` - download original TXT

## Demo Test File

Use `temp_whatsapp_export_50.txt` for a quick test with 50 messages.
