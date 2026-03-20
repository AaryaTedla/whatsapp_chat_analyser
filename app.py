import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from analyser import compute_stats, generate_ai_summary
from parser import parse_whatsapp_export

load_dotenv()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

DB_PATH = os.getenv(
    "DB_PATH",
    "/tmp/lumira_reports.db",
)


def _ensure_instance_dir() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _get_db() -> sqlite3.Connection:
    _ensure_instance_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_db()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                message_count INTEGER NOT NULL,
                txt_content TEXT NOT NULL,
                stats_json TEXT NOT NULL,
                ai_json TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _save_report(
    *,
    filename: str,
    txt_content: str,
    message_count: int,
    stats: Dict[str, Any],
    ai: Dict[str, Any],
) -> int:
    conn = _get_db()
    try:
        cur = conn.execute(
            """
            INSERT INTO reports (filename, uploaded_at, message_count, txt_content, stats_json, ai_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                datetime.now(timezone.utc).isoformat(),
                int(message_count),
                txt_content,
                json.dumps(stats, ensure_ascii=True),
                json.dumps(ai, ensure_ascii=True),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _fetch_reports(limit: int = 100) -> list[Dict[str, Any]]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """
            SELECT id, filename, uploaded_at, message_count
            FROM reports
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _fetch_report(report_id: int) -> Dict[str, Any] | None:
    conn = _get_db()
    try:
        row = conn.execute(
            """
            SELECT id, filename, uploaded_at, message_count, txt_content, stats_json, ai_json
            FROM reports
            WHERE id = ?
            """,
            (int(report_id),),
        ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["stats"] = json.loads(data.pop("stats_json"))
        data["ai"] = json.loads(data.pop("ai_json"))
        return data
    finally:
        conn.close()


def _decode_uploaded_bytes(raw: bytes) -> str:
    """
    WhatsApp exports are usually UTF-8, but we try a safe fallback.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/reports", methods=["GET"])
def reports_page():
    if request.accept_mimetypes.best == "application/json" or request.args.get("format") == "json":
        return jsonify({"reports": _fetch_reports(limit=200)})
    return render_template("reports.html")


@app.route("/reports/<int:report_id>", methods=["GET"])
def get_report(report_id: int):
    report = _fetch_report(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    return jsonify(report)


@app.route("/reports/<int:report_id>/view", methods=["GET"])
def view_report(report_id: int):
    report = _fetch_report(report_id)
    if not report:
        return "Report not found", 404
    return render_template("report_view.html", report=report)


@app.route("/reports/<int:report_id>/txt", methods=["GET"])
def get_report_txt(report_id: int):
    report = _fetch_report(report_id)
    if not report:
        return jsonify({"error": "Report not found"}), 404
    txt_filename = os.path.splitext(report["filename"])[0] + f"_report_{report_id}.txt"
    return Response(
        report["txt_content"],
        content_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{txt_filename}"'},
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "Missing file upload"}), 400

    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".txt"):
        return jsonify({"error": "Please upload a .txt WhatsApp export"}), 400

    try:
        raw = file.read()
        text = _decode_uploaded_bytes(raw)
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    try:
        messages = parse_whatsapp_export(text)
    except Exception as e:
        return jsonify({"error": f"Parse error: {e}"}), 400

    if not messages:
        return jsonify({"error": "Could not parse any WhatsApp messages from that file"}), 400

    try:
        stats: Dict[str, Any] = compute_stats(messages)
        ai: Dict[str, Any] = generate_ai_summary(messages, max_messages=50)
    except Exception as e:
        return jsonify({"error": f"Analysis error: {e}"}), 500

    try:
        report_id = _save_report(
            filename=filename,
            txt_content=text,
            message_count=len(messages),
            stats=stats,
            ai=ai,
        )
    except Exception as e:
        return jsonify({"error": f"Database save error: {e}"}), 500

    return jsonify({"stats": stats, "ai": ai, "report_id": report_id})


if __name__ == "__main__":
    # For local development only.
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug)


_init_db()

