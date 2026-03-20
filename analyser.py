import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Step 3.5 Flash works reliably; Llama is often rate-limited. Override in .env if needed.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "stepfun/step-3.5-flash:free")


STOP_WORDS = {
    # Common English stop words
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
    "you",
    "your",
    # Common chat tokens / contractions
    "im",
    "ive",
    "ill",
    "dont",
    "cant",
    "wont",
    "isnt",
    "doesnt",
    "didnt",
    "theres",
    "thats",
    "what",
    "whats",
    "how",
    "who",
    "why",
    "when",
    "where",
    "from",
    "are",
    "also",
    "just",
    "like",
    "lol",
    "haha",
    "ok",
    "okay",
    "yeah",
    "sure",
    "ur",
    "u",
    "got",
    "gonna",
    "going",
    "want",
    "wish",
    "know",
    "also",
    "amp",
}

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")


def _pick_sample(messages: List[Dict[str, Any]], n: int = 50) -> List[Dict[str, Any]]:
    if len(messages) <= n:
        return messages
    # Evenly space the sample across the whole conversation.
    # This reduces "recency bias" vs taking the first 50.
    idxs = [int(i * len(messages) / n) for i in range(n)]
    return [messages[i] for i in idxs if i < len(messages)]


def compute_stats(messages: List[Dict[str, Any]], top_senders: int = 10, top_words: int = 20) -> Dict[str, Any]:
    """
    Compute stats from parsed messages.

    Expects each item to have: date, time, datetime (ISO string or None), sender, message.
    """
    df = pd.DataFrame(messages)
    if df.empty:
        return {
            "total_messages": 0,
            "senders": {"labels": [], "counts": []},
            "hours": {"labels": [], "counts": []},
            "days": {"labels": [], "counts": []},
            "top_words": [],
        }

    # Only compute time-series metrics for valid datetimes.
    df_time = df[df["datetime"].notna()].copy()
    df_time["datetime"] = pd.to_datetime(df_time["datetime"], errors="coerce")
    df_time = df_time[df_time["datetime"].notna()]

    df_time["hour"] = df_time["datetime"].dt.hour
    df_time["date_only"] = df_time["datetime"].dt.strftime("%Y-%m-%d")

    total_messages = int(len(df))

    senders_series = df["sender"].fillna("Unknown").value_counts().head(top_senders)
    senders_labels = senders_series.index.tolist()
    senders_counts = senders_series.values.tolist()

    hours_counts = (
        df_time.groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
    )
    hours_labels = [f"{h:02d}" for h in range(24)]
    hours_data = hours_counts.values.tolist()

    days_series = df_time.groupby("date_only").size().sort_index()
    days_labels_raw = days_series.index.tolist()
    days_data_raw = days_series.values.tolist()
    # Cap to 400 points to avoid huge responses and Chart.js overload
    max_days = 400
    if len(days_labels_raw) > max_days:
        df_ts = df_time.copy()
        df_ts["date_only"] = pd.to_datetime(df_ts["date_only"])
        weekly = df_ts.groupby(df_ts["date_only"].dt.to_period("W").astype(str)).size()
        days_labels = weekly.index.tolist()[:max_days]
        days_data = weekly.values.tolist()[:max_days]
    else:
        days_labels = days_labels_raw
        days_data = days_data_raw

    # Tokenize for "most used words".
    tokens: List[str] = []
    for msg in df["message"].fillna("").astype(str).tolist():
        msg = msg.lower()
        tokens.extend(WORD_RE.findall(msg))

    # Filter stop words + short tokens.
    filtered = [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]
    counter = Counter(filtered)
    top = counter.most_common(top_words)

    return {
        "total_messages": total_messages,
        "unique_senders": int(df["sender"].nunique()),
        "senders": {"labels": senders_labels, "counts": senders_counts},
        "hours": {"labels": hours_labels, "counts": hours_data},
        "days": {"labels": days_labels, "counts": days_data},
        "top_words": [{"word": w, "count": c} for w, c in top],
    }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract a JSON object from the model output.
    Handles the common case where the model wraps JSON in a code block.
    """
    # Remove code fences if present.
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    cleaned = cleaned.strip()

    # Find the first {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _deep_collect_text(value: Any) -> List[str]:
    """
    Recursively collect likely text fields from unknown response shapes.
    OpenRouter/provider wrappers can nest the assistant output in different keys.
    """
    out: List[str] = []

    if value is None:
        return out

    if isinstance(value, str):
        s = value.strip()
        if s:
            out.append(s)
        return out

    if isinstance(value, list):
        for item in value:
            out.extend(_deep_collect_text(item))
        return out

    if hasattr(value, "model_dump"):
        try:
            return _deep_collect_text(value.model_dump())
        except Exception:
            pass

    if isinstance(value, dict):
        # Prefer common payload keys first, then scan everything.
        preferred_keys = (
            "text",
            "content",
            "output_text",
            "value",
            "message",
        )
        for key in preferred_keys:
            if key in value:
                out.extend(_deep_collect_text(value.get(key)))

        for key, item in value.items():
            if key not in preferred_keys:
                # Avoid collecting chain-of-thought/metadata-like payloads.
                if key in {"reasoning", "analysis", "thought", "refusal", "annotations"}:
                    continue
                if isinstance(item, (dict, list)) or hasattr(item, "model_dump"):
                    out.extend(_deep_collect_text(item))

    return out


def _is_placeholder_text(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    if not t:
        return False
    placeholders = {
        "2-4 sentences on overall tone",
        "one light observation",
        "topic1",
        "topic2",
        "topic3",
    }
    if t in placeholders:
        return True
    if re.fullmatch(r"topic\s*\d+", t):
        return True
    return False


def _local_fallback_summary(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a reasonable summary when the model returns placeholders/invalid JSON.
    """
    if not messages:
        return {
            "vibe_summary": "The chat appears light and conversational, but there was not enough usable content to summarize confidently.",
            "top_3_topics": ["general chat", "quick updates", "planning"],
            "funny_observation": "Even without full context, the conversation still has classic quick-chat energy.",
        }

    senders = [str(m.get("sender") or "Unknown").strip() for m in messages]
    unique_senders = len(set(s for s in senders if s and s != "Unknown"))

    text_lines = [str(m.get("message") or "").strip() for m in messages]
    text_lines = [t for t in text_lines if t]
    joined = " ".join(text_lines).lower()

    question_count = sum(1 for t in text_lines if "?" in t)
    short_count = sum(1 for t in text_lines if len(t.split()) <= 3)

    tokens: List[str] = []
    for line in text_lines:
        tokens.extend(WORD_RE.findall(line.lower()))
    filtered = [t for t in tokens if t not in STOP_WORDS and len(t) >= 3]
    topics = [w for w, _ in Counter(filtered).most_common(3)]
    while len(topics) < 3:
        topics.append(["updates", "planning", "check-ins"][len(topics)])

    coordination_hits = sum(1 for w in ["tomorrow", "check", "on", "it", "brb", "done", "fixed"] if w in joined)
    tone_parts = []
    if short_count >= max(3, len(text_lines) // 2):
        tone_parts.append("The conversation is fast-paced with lots of short check-ins")
    else:
        tone_parts.append("The conversation is balanced and conversational")

    if question_count > 0:
        tone_parts.append("with a steady back-and-forth of quick questions and responses")

    if coordination_hits > 0:
        tone_parts.append("and a practical, task-focused tone around coordination and updates")

    vibe = ". ".join([p.strip() for p in tone_parts if p]).strip()
    if not vibe.endswith("."):
        vibe += "."
    vibe += f" There are {unique_senders or 'multiple'} active participants, which keeps the chat collaborative."

    funny = "Half the messages read like mission-control status pings: 'On it', 'Brb', and 'Tomorrow?' in rapid succession."

    return {
        "vibe_summary": vibe,
        "top_3_topics": topics[:3],
        "funny_observation": funny,
    }


def generate_ai_summary(messages: List[Dict[str, Any]], max_messages: int = 50) -> Dict[str, Any]:
    """
    Ask OpenRouter (via the OpenAI SDK) for:
    - overall vibe summary
    - top 3 recurring topics
    - one funny observation
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return {
            "error": "Missing OPENROUTER_API_KEY in .env",
            "vibe_summary": None,
            "top_3_topics": [],
            "funny_observation": None,
        }

    # Prefer actual chat lines (skip system-ish senders when possible).
    candidates = [
        m
        for m in messages
        if m.get("message")
        and str(m.get("message")).strip()
        and m.get("sender") not in {"System", "Unknown", None, ""}
    ]
    if len(candidates) >= 3:
        sample_pool = candidates
    else:
        sample_pool = messages

    sample = _pick_sample(sample_pool, n=max_messages)

    def fmt_line(m: Dict[str, Any]) -> str:
        sender = (m.get("sender") or "Unknown").strip()
        date = m.get("date") or ""
        time = m.get("time") or ""
        ts = f"{date} {time}".strip()
        body = str(m.get("message") or "").strip().replace("\n", " ")
        if len(body) > 300:
            body = body[:300] + "..."
        return f"{ts} - {sender}: {body}"

    sample_text = "\n".join(fmt_line(m) for m in sample)

    prompt = f"""You are analyzing a WhatsApp chat. Reply in this exact JSON format (no other text):

{{"vibe_summary": "2-4 sentences on overall tone", "top_3_topics": ["topic1", "topic2", "topic3"], "funny_observation": "one light observation"}}

Do not include explanations, markdown, or thinking text.

Chat sample:
{sample_text}"""

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    def extract_content(msg) -> str:
        """Extract text from message; handles various response shapes."""
        raw = getattr(msg, "content", None) or getattr(msg, "text", None)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        if isinstance(raw, list):
            texts = []
            for p in raw:
                if isinstance(p, dict):
                    texts.append(p.get("text") or p.get("content") or "")
                elif hasattr(p, "text"):
                    texts.append(str(getattr(p, "text", "") or ""))
                elif hasattr(p, "model_dump"):
                    d = p.model_dump()
                    texts.append(str(d.get("text") or d.get("content") or ""))
                else:
                    texts.append(str(p))
            s = " ".join(t for t in texts if t).strip()
            if s:
                return s
        if hasattr(msg, "model_dump"):
            d = msg.model_dump()
            c = d.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):
                out = []
                for x in c:
                    if isinstance(x, dict):
                        out.append(x.get("text") or x.get("content") or "")
                    elif hasattr(x, "model_dump"):
                        out.append(str(x.model_dump().get("text") or x.model_dump().get("content") or ""))
                    else:
                        out.append(str(x))
                s = " ".join(t for t in out if t).strip()
                if s:
                    return s

        # Last-resort deep scan across all keys in model_dump shape.
        if hasattr(msg, "model_dump"):
            try:
                values = _deep_collect_text(msg.model_dump())
                if values:
                    return " ".join(values).strip()
            except Exception:
                pass

        return ""

    try:
        try:
            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
        except Exception:
            # Some providers/models don't support response_format=json_object.
            resp = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
        if not resp.choices:
            return {
                "error": "API returned no choices",
                "vibe_summary": None,
                "top_3_topics": [],
                "funny_observation": None,
            }
        content = extract_content(resp.choices[0].message)
        if not content and hasattr(resp, "model_dump"):
            try:
                all_texts = _deep_collect_text(resp.model_dump())
                if all_texts:
                    content = " ".join(all_texts).strip()
            except Exception:
                pass

        if not content:
            try:
                import httpx
                with httpx.Client(timeout=60) as h:
                    r = h.post(
                        f"{OPENROUTER_BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": OPENROUTER_MODEL,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.3,
                            "max_tokens": 512,
                        },
                    )
                    r.raise_for_status()
                    data = r.json()
                    texts = _deep_collect_text(data)
                    if texts:
                        content = " ".join(texts).strip()
            except Exception:
                pass
    except Exception as e:
        return {
            "error": str(e),
            "vibe_summary": None,
            "top_3_topics": [],
            "funny_observation": None,
        }

    if not content:
        debug_finish = None
        debug_payload = None
        try:
            if resp and getattr(resp, "choices", None):
                debug_finish = getattr(resp.choices[0], "finish_reason", None)
            if resp and hasattr(resp, "model_dump"):
                debug_payload = json.dumps(resp.model_dump(), ensure_ascii=True)[:1200]
        except Exception:
            pass
        if debug_payload:
            print("[generate_ai_summary] Empty extracted content from provider response:")
            print(debug_payload)
        return {
            "error": (
                "Model returned empty."
                + (f" finish_reason={debug_finish}" if debug_finish else "")
                + " Check Flask terminal for response structure."
            ),
            "vibe_summary": None,
            "top_3_topics": [],
            "funny_observation": None,
        }

    try:
        parsed = _extract_json_object(content) or {}
    except Exception:
        parsed = {}

    vibe = (
        parsed.get("vibe_summary") or parsed.get("summary") or parsed.get("overall_vibe")
    )
    vibe = str(vibe).strip() if vibe else None

    topics = parsed.get("top_3_topics") or parsed.get("topics") or parsed.get("main_topics")
    if isinstance(topics, list):
        topics = [str(t).strip() for t in topics[:3] if t]
    elif isinstance(topics, str):
        topics = [t.strip() for t in topics.replace(",", " ").split()[:3] if t.strip()]
    else:
        topics = []

    funny = parsed.get("funny_observation") or parsed.get("funny_note") or parsed.get("observation")
    funny = str(funny).strip() if funny else None

    if not parsed and content:
        fallback = _local_fallback_summary(messages)
        fallback["raw"] = content[:400] if content else None
        fallback["note"] = "Fallback summary used because model response was not valid JSON."
        return fallback

    topics_are_placeholders = bool(topics) and all(_is_placeholder_text(t) for t in topics)
    if _is_placeholder_text(vibe) or _is_placeholder_text(funny) or topics_are_placeholders:
        fallback = _local_fallback_summary(messages)
        fallback["raw"] = content[:400] if content else None
        fallback["note"] = "Fallback summary used because model returned template placeholders."
        return fallback

    return {
        "vibe_summary": vibe or None,
        "top_3_topics": topics,
        "funny_observation": funny or None,
        "raw": content or None,
    }

