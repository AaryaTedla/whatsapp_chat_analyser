# Lumira - Professional WhatsApp Chat Analyzer

A modern, AI-powered web application for analyzing WhatsApp chat exports. Get intelligent insights about conversation patterns, key topics, and group dynamics.

**Live Demo:** [Coming Soon]

## ✨ Features

### Core Analysis
- 📊 **Advanced Statistics**: Sender distribution, activity patterns over time, hourly breakdowns
- 🤖 **AI-Powered Insights**: Automatic vibe analysis, topic detection, funny observations (powered by OpenRouter)
- 📈 **Interactive Charts**: Real-time visualizations using Chart.js
- 💾 **Report Storage**: Save and revisit all your analysis reports

### New in v2.0
- ⚡ **FastAPI Backend**: Modern, type-safe Python backend
- 📊 **Polars Integration**: Ultra-fast data processing with lazy evaluation
- 📱 **Progressive Web App**: Full PWA support with offline capability
- 🎨 **Professional UI**: Modern, accessible design inspired by industry leaders
- 🔒 **Better Architecture**: SQLAlchemy + best practices throughout
- 📤 **Export Features**: Download reports, share results
- 🎯 **Lazy Evaluation**: Polars lazy evaluation for optimal performance

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
# Clone and enter directory
git clone <repo>
cd lumira_prep

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### Running Locally

```bash
# Development mode (with auto-reload)
python app.py

# Or with uvicorn directly
uvicorn app:app --reload --port 8000

# Production mode
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

Open your browser to `http://localhost:8000`

## 📧 Environment Variables

Create a `.env` file (or copy from `.env.example`):

```env
# OpenRouter API (for AI summaries)
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=google/gemma-3-4b-it:free

# Database
DATABASE_URL=sqlite:///./lumira_reports.db

# Optional
UPLOAD_DIR=./instance/uploads
```

**Getting an API Key:**
1. Sign up at [OpenRouter](https://openrouter.ai)
2. Copy your API key from dashboard
3. Add to `.env`

## 🏗️ Project Structure

```
lumira_prep/
├── app.py                 # FastAPI main application
├── analyser.py            # Polars-based analysis engine
├── parser.py              # WhatsApp export parser
├── requirements.txt       # Python dependencies
├── core/                  # Core modules
│   ├── config.py         # Configuration management
│   ├── database.py       # SQLAlchemy setup
│   └── models.py         # Database models
├── templates/            # HTML templates
│   ├── index.html        # Main analyzer page
│   ├── report_view.html  # Report detail page
│   └── reports.html      # Reports list
├── static/               # PWA assets
│   ├── manifest.json     # PWA manifest
│   └── service-worker.js # Service worker
└── sample_exports/       # Example chat exports
```

## 🔧 Technology Stack

### Backend
- **FastAPI** - Modern async Python web framework
- **Polars** - Lightning-fast DataFrame library with lazy evaluation
- **SQLAlchemy** - ORM for database operations
- **Pydantic** - Data validation using Python type hints
- **OpenRouter API** - Access to multiple LLMs

### Frontend
- **Vanilla JavaScript** - No framework dependencies
- **Chart.js** - Interactive charts and visualizations
- **Progressive Web App** - Offline support, installable
- **Modern CSS** - Clean, responsive design

### Database
- **SQLite** - Default (easy to swap for PostgreSQL)
- **Proper Migrations** - Using SQLAlchemy

## 📊 API Endpoints

### Analysis
- **POST** `/api/analyze` - Analyze a WhatsApp export
  - Body: `file` (multipart/form-data)
  - Returns: Analysis results with stats and AI insights

### Reports
- **GET** `/api/reports` - List all reports (paginated)
  - Query: `limit`, `offset`
- **GET** `/api/reports/{id}` - Get report details
- **GET** `/api/reports/{id}/download` - Download original export
- **GET** `/r/{id}` - View report in HTML

### System
- **GET** `/api/health` - Health check

## 🤖 AI Analysis Details

The AI summary includes:
- **Vibe Summary** - Overall tone and conversation style (2-4 sentences)
- **Top 3 Topics** - Key recurring themes in the chat
- **Funny Observation** - A light, grounded observation about the group dynamic

Powered by OpenRouter, supporting 100+ models (Gemma, Llama, Mistral, Claude, etc.)

## 📱 PWA Features

Lumira is a full Progressive Web App:
- **Install** on home screen (iOS/Android)
- **Offline Support** via Service Worker
- **Background Sync** for queued reports
- **Native-like Experience** with standalone display mode
- **App Shortcuts** for quick access to common actions

## 🎨 UI/UX Highlights

Inspired by modern products like Linear, Vercel, and GitHub:
- Dark theme with subtle gradients
- Smooth animations and transitions
- Touch-optimized interactions
- Mobile-first responsive design
- Accessible color contrasts
- Professional typography

## 🔐 Security & Best Practices

- ✅ Type hints throughout (Python 3.9+)
- ✅ Proper error handling and logging
- ✅ Data validation with Pydantic
- ✅ CORS configured
- ✅ SQL injection protected (SQLAlchemy ORM)
- ✅ File size limits enforced
- ✅ Graceful fallbacks for AI API failures

## 📈 Performance

- **Lazy Evaluation** - Polars lazy frames for large datasets
- **Caching** - Service Worker caches static assets
- **Pagination** - Report listing with limits
- **Optimized Charts** - Downsampled data for 400+ day ranges
- **Fast Parsing** - Efficient regex-based WhatsApp parser

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'polars'"
```bash
pip install --upgrade -r requirements.txt
```

### Service Worker Not Working
- Ensure you're on HTTPS (or localhost for testing)
- Check browser's Application tab → Service Workers
- Clear cache if updating

### AI Summary Not Generating
- Check `OPENROUTER_API_KEY` is set correctly
- Verify API quota at [OpenRouter Dashboard](https://openrouter.ai/account/usage)
- Check terminal logs for API errors

### Database Errors
```bash
# Delete and recreate database
rm lumira_reports.db
python app.py
```

## 📝 Example Usage

1. **Prepare your chat**: Export WhatsApp chat as .txt from the app (Settings → Chats & Calls)
2. **Upload**: Go to Lumira homepage, upload the .txt file
3. **Analyze**: Wait for processing (usually <10s)
4. **Review**: Check stats, charts, and AI insights
5. **Share**: Copy report link or download for sharing

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional visualization types
- More AI providers (Anthropic, OpenAI directly)
- Advanced filtering/search
- Comparative analysis (multiple chats)
- Export formats (PDF, JSON, CSV)
- Self-hosted LLM integration

## 📄 License

[Your License Here]

## 🙋 Support

- 📧 Email: [Your Email]
- 🐙 Issues: [GitHub Issues]
- 💬 Discussions: [GitHub Discussions]

---

**Made with ❤️ | Built with FastAPI + Polars + PWA**


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
