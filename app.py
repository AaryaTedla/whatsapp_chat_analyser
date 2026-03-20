import os
from typing import Any, Dict

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from analyser import compute_stats, generate_ai_summary
from parser import parse_whatsapp_export

load_dotenv()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB


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

    return jsonify({"stats": stats, "ai": ai})


if __name__ == "__main__":
    # For local development only.
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug)

