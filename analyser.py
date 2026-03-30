import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import polars as pl
from openai import OpenAI

from core.config import OPENROUTER_API_KEY, OPENROUTER_MODEL




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

# Additional filler words to suppress noisy fallback topics.
FALLBACK_TOPIC_STOP_WORDS = {
    "yes",
    "now",
    "one",
    "today",
    "tomorrow",
    "really",
    "thing",
    "things",
    "good",
    "great",
    "nice",
    "work",
    "team",
    "morning",
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
    Compute stats from parsed messages using Polars with lazy evaluation.
    
    Expects each item to have: date, time, datetime, sender, message.
    """
    if not messages:
        return {
            "total_messages": 0,
            "unique_senders": 0,
            "senders": {"labels": [], "counts": []},
            "hours": {"labels": [], "counts": []},
            "days": {"labels": [], "counts": []},
            "top_words": [],
        }

    # Create lazy dataframe
    df_lazy = pl.LazyFrame(messages)
    
    # Collect for basic stats
    df = df_lazy.collect()
    
    total_messages = len(df)
    unique_senders = df.select(pl.col("sender")).n_unique()

    # Process timestamps lazily
    df_time_lazy = (
        df_lazy
        .filter(pl.col("datetime").is_not_null())
        .with_columns(
            pl.col("datetime").str.to_datetime(),
        )
        .with_columns(
            hour=pl.col("datetime").dt.hour(),
            date_only=pl.col("datetime").dt.strftime("%Y-%m-%d"),
        )
    )
    
    df_time = df_time_lazy.collect()
    
    if len(df_time) == 0:
        df_time = df
        has_time = False
    else:
        has_time = True

    # Senders
    senders_stats = (
        df.lazy()
        .select(pl.col("sender").fill_null("Unknown"))
        .group_by("sender")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .limit(top_senders)
        .collect()
    )
    
    senders_labels = senders_stats["sender"].to_list()
    senders_counts = senders_stats["count"].to_list()

    # Hours of day
    if has_time:
        hours_data = [0] * 24
        for h in df_time["hour"].to_list():
            if 0 <= h < 24:
                hours_data[h] += 1
        hours_labels = [f"{h:02d}" for h in range(24)]
    else:
        hours_labels = [f"{h:02d}" for h in range(24)]
        hours_data = [0] * 24

    # Days over time
    if has_time:
        days_stats = (
            df_time_lazy
            .group_by("date_only")
            .agg(pl.len().alias("count"))
            .sort("date_only")
            .collect()
        )
        days_labels = days_stats["date_only"].to_list()
        days_data = days_stats["count"].to_list()
        
        # Cap to 400 points
        max_days = 400
        if len(days_labels) > max_days:
            # Downsample to weekly
            step = len(days_labels) // max_days + 1
            days_labels = days_labels[::step]
            days_data = days_data[::step]
    else:
        days_labels = []
        days_data = []

    # Top words using lazy evaluation
    messages_text = df.select(pl.col("message")).to_series().to_list()
    tokens: List[str] = []
    for msg in messages_text:
        if msg:
            msg_str = str(msg).lower()
            tokens.extend(WORD_RE.findall(msg_str))

    filtered = [t for t in tokens if t not in STOP_WORDS and len(t) >= 2]
    counter = Counter(filtered)
    top = counter.most_common(top_words)

    return {
        "total_messages": int(total_messages),
        "unique_senders": int(unique_senders),
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
        "...",
        "..",
        ".",
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
    filtered = [
        t
        for t in tokens
        if t not in STOP_WORDS
        and t not in FALLBACK_TOPIC_STOP_WORDS
        and len(t) >= 3
    ]
    topics = [w for w, _ in Counter(filtered).most_common(3)]
    while len(topics) < 3:
        topics.append(["updates", "planning", "check-ins"][len(topics)])

    coordination_hits = sum(1 for w in ["tomorrow", "check", "brb", "done", "fixed", "deploy", "release"] if w in joined)
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

    # Build a light observation from actual short lines to avoid fake phrases.
    short_examples: List[str] = []
    seen = set()
    for line in text_lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if len(cleaned.split()) <= 4 and len(cleaned) <= 40:
            key = cleaned.lower()
            if key not in seen:
                seen.add(key)
                short_examples.append(cleaned)
        if len(short_examples) >= 3:
            break

    if short_examples:
        quoted = ", ".join(f"'{x}'" for x in short_examples[:3])
        funny = f"The chat has quick status-update energy, with rapid one-liners like {quoted}."
    else:
        funny = "The conversation keeps a brisk rhythm with concise updates and quick handoffs."

    return {
        "vibe_summary": vibe,
        "top_3_topics": topics[:3],
        "funny_observation": funny,
    }


def generate_ai_summary(
    messages: List[Dict[str, Any]],
    max_messages: int = 50,
) -> Dict[str, Any]:
    """
    Async AI summary generation with fallback.
    """
    if not OPENROUTER_API_KEY:
        result = _local_fallback_summary(messages)
        result["error"] = "Missing OPENROUTER_API_KEY"
        return result

    candidates = [
        m for m in messages
        if m.get("message")
        and str(m.get("message")).strip()
        and m.get("sender") not in {"System", "Unknown", None, ""}
    ]
    sample_pool = candidates if len(candidates) >= 3 else messages
    sample = _pick_sample(sample_pool, n=max_messages)

    def fmt_line(m: Dict[str, Any]) -> str:
        sender = (m.get("sender") or "Unknown").strip()
        date = m.get("date", "")
        time = m.get("time", "")
        ts = f"{date} {time}".strip()
        body = str(m.get("message", "")).strip().replace("\n", " ")
        if len(body) > 300:
            body = body[:300] + "..."
        return f"{ts} - {sender}: {body}"

    sample_text = "\n".join(fmt_line(m) for m in sample)

    prompt = f"""Analyze this WhatsApp chat and return ONLY valid JSON with this shape:
{{"vibe_summary":"...", "top_3_topics":["...","...","..."], "funny_observation":"..."}}

Rules:
1. vibe_summary: 2-4 short sentences on overall tone
2. top_3_topics: exactly 3 concise topic phrases (not placeholders)
3. funny_observation: one light observation grounded in the chat
4. No markdown, no code fences, no extra keys

Chat:
{sample_text}"""

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end+1])
            except json.JSONDecodeError:
                pass
        
        return _local_fallback_summary(messages)
    
    except Exception as e:
        result = _local_fallback_summary(messages)
        result["error"] = str(e)
        return result


def analyze_sentiment(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze sentiment of messages using keyword-based heuristics and optionally AI.
    Returns sentiment scores and trends.
    """
    if not messages:
        return {
            "overall_sentiment": "neutral",
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 1.0,
            "by_sender": {},
            "timeline": []
        }
    
    positive_words = {
        "good", "great", "awesome", "amazing", "excellent", "nice", "love", 
        "happy", "wonderful", "perfect", "fantastic", "brilliant", "cool",
        "awesome", "yay", "lol", "haha", "thanks", "thank", "best"
    }
    
    negative_words = {
        "bad", "terrible", "awful", "hate", "angry", "sad", "disappointed",
        "stupid", "dumb", "sucks", "annoyed", "frustrated", "worst", "horrible",
        "disgusting", "ugh", "no", "nope", "nahi"
    }
    
    sentiments = []
    by_sender = {}
    
    for msg in messages:
        text = str(msg.get("message", "")).lower()
        sender = msg.get("sender", "Unknown")
        
        if not text.strip():
            sentiment = "neutral"
        else:
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            if pos_count > neg_count:
                sentiment = "positive"
            elif neg_count > pos_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        sentiments.append(sentiment)
        
        if sender not in by_sender:
            by_sender[sender] = {"positive": 0, "negative": 0, "neutral": 0}
        by_sender[sender][sentiment] += 1
    
    total = len(sentiments)
    positive_count = sentiments.count("positive")
    negative_count = sentiments.count("negative")
    
    return {
        "overall_sentiment": "positive" if positive_count > negative_count else ("negative" if negative_count > positive_count else "neutral"),
        "positive_ratio": round(positive_count / total * 100, 2) if total > 0 else 0,
        "negative_ratio": round(negative_count / total * 100, 2) if total > 0 else 0,
        "neutral_ratio": round((total - positive_count - negative_count) / total * 100, 2) if total > 0 else 100,
        "by_sender": {sender: {k: v for k, v in counts.items()} for sender, counts in by_sender.items()}
    }


def analyze_emojis(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze emoji usage across the chat.
    Returns most used emojis and emoji distribution by sender.
    """
    import unicodedata
    
    emoji_pattern = re.compile(
        r"["
        r"\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"  # symbols & pictographs
        r"\U0001F680-\U0001F6FF"  # transport & map symbols
        r"\U0001F700-\U0001F77F"  # alchemical symbols
        r"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        r"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        r"\U0001FA00-\U0001FA6F"  # Chess Symbols
        r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        r"\U000200D"  # zero width joiner
        r"\U0000FE0F"  # dingbats
        r"]+",
        flags=re.UNICODE
    )
    
    all_emojis = []
    by_sender = {}
    
    for msg in messages:
        text = msg.get("message", "")
        sender = msg.get("sender", "Unknown")
        
        if text:
            emojis = emoji_pattern.findall(str(text))
            all_emojis.extend(emojis)
            
            if sender not in by_sender:
                by_sender[sender] = {}
            
            for emoji in emojis:
                by_sender[sender][emoji] = by_sender[sender].get(emoji, 0) + 1
    
    emoji_counts = Counter(all_emojis)
    top_emojis = emoji_counts.most_common(15)
    
    return {
        "total_emoji_count": len(all_emojis),
        "unique_emojis": len(emoji_counts),
        "top_emojis": [{"emoji": e, "count": c} for e, c in top_emojis],
        "by_sender": {sender: sorted([(e, c) for e, c in emojis.items()], key=lambda x: x[1], reverse=True)[:5] 
                     for sender, emojis in by_sender.items() if emojis}
    }


def analyze_mentions(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze mentions (@mentions) in the chat.
    """
    mention_pattern = re.compile(r"@[\w\s]+")
    
    all_mentions = []
    mention_by_sender = {}
    
    for msg in messages:
        text = msg.get("message", "")
        sender = msg.get("sender", "Unknown")
        
        if text:
            mentions = mention_pattern.findall(str(text).lower())
            all_mentions.extend(mentions)
            
            if sender not in mention_by_sender:
                mention_by_sender[sender] = []
            mention_by_sender[sender].extend(mentions)
    
    mention_counts = Counter(all_mentions)
    top_mentions = mention_counts.most_common(10)
    
    return {
        "total_mentions": len(all_mentions),
        "unique_mentions": len(mention_counts),
        "top_mentions": [{"mention": m, "count": c} for m, c in top_mentions],
        "mention_by_sender": {sender: Counter(mentions).most_common(5) for sender, mentions in mention_by_sender.items() if mentions}
    }


def analyze_links_and_media(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze links, images, videos, and other media mentions in the chat.
    """
    url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    media_indicators = {
        "image": [".jpg", ".jpeg", ".png", ".gif", ".webp", "<image omitted>"],
        "video": [".mp4", ".avi", ".mov", ".mkv", "<video omitted>"],
        "audio": [".mp3", ".wav", ".m4a", ".aac", "<audio omitted>"],
        "document": [".pdf", ".doc", ".docx", ".xlsx", ".pptx", "<document omitted>"],
    }
    
    links = []
    media_counts = {"image": 0, "video": 0, "audio": 0, "document": 0}
    by_sender = {}
    
    for msg in messages:
        text = str(msg.get("message", ""))
        sender = msg.get("sender", "Unknown")
        
        if sender not in by_sender:
            by_sender[sender] = {"links": 0, "media": 0}
        
        # Extract URLs
        urls = url_pattern.findall(text)
        links.extend(urls)
        by_sender[sender]["links"] += len(urls)
        
        # Check for media indicators
        text_lower = text.lower()
        for media_type, indicators in media_indicators.items():
            if any(ind in text_lower for ind in indicators):
                media_counts[media_type] += 1
                by_sender[sender]["media"] += 1
    
    return {
        "total_links": len(links),
        "unique_links": len(set(links)),
        "sample_links": list(set(links))[:10],
        "media_distribution": media_counts,
        "by_sender": by_sender
    }


def detect_languages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect languages in messages using simple heuristics.
    Returns detected languages and their distribution.
    """
    def detect_simple_language(text: str) -> str:
        """Simple language detection based on character ranges and common words."""
        if not text:
            return "unknown"
        
        text_lower = text.lower()
        
        # Check for common words/patterns
        hindi_indicators = ["hai", "hain", "kya", "nahi", "acha", "theek", "arre", "wo", "yeh"]
        spanish_indicators = ["que", "como", "pero", "porque", "gracias", "hola", "es"]
        french_indicators = ["le", "la", "est", "de", "et", "pour", "bonjour"]
        
        hindi_count = sum(1 for word in hindi_indicators if word in text_lower)
        spanish_count = sum(1 for word in spanish_indicators if word in text_lower)
        french_count = sum(1 for word in french_indicators if word in text_lower)
        
        if hindi_count > 0:
            return "hindi"
        if spanish_count > french_count:
            return "spanish"
        if french_count > 0:
            return "french"
        
        return "english"
    
    languages = []
    for msg in messages:
        text = msg.get("message", "")
        if text:
            lang = detect_simple_language(str(text))
            languages.append(lang)
    
    lang_counts = Counter(languages)
    
    return {
        "detected_languages": dict(lang_counts),
        "primary_language": lang_counts.most_common(1)[0][0] if lang_counts else "unknown",
        "language_distribution": {lang: round(count / len(languages) * 100, 2) 
                                 for lang, count in lang_counts.items()} if languages else {}
    }


def analyze_response_patterns(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze response times and patterns between messages.
    Returns conversation flow metrics.
    """
    if len(messages) < 2:
        return {"average_response_time": None, "conversation_continuity": 0.0}
    
    time_diffs = []
    same_sender_count = 0
    total_pairs = 0
    
    for i in range(1, len(messages)):
        prev_msg = messages[i-1]
        curr_msg = messages[i]
        
        # Simple continuity: count messages from same sender
        if prev_msg.get("sender") == curr_msg.get("sender"):
            same_sender_count += 1
        
        total_pairs += 1
    
    conversation_continuity = (1 - (same_sender_count / total_pairs)) if total_pairs > 0 else 0
    
    return {
        "total_messages": len(messages),
        "conversation_continuity": round(conversation_continuity * 100, 2),
        "average_message_length": round(
            sum(len(str(m.get("message", ""))) for m in messages) / len(messages), 2
        ) if messages else 0,
        "short_message_ratio": round(
            sum(1 for m in messages if len(str(m.get("message", ""))) < 10) / len(messages) * 100, 2
        ) if messages else 0
    }


def analyze_conversation_heatmap(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze activity by hour of day and day of week.
    Returns heatmap data for visualization.
    """
    from datetime import datetime as dt_cls
    
    hourly_activity = Counter()  # 0-23 hours
    daily_activity = Counter()   # 0-6 (Monday-Sunday)
    
    for msg in messages:
        dt = msg.get("datetime")
        if dt:
            # Handle string ISO format or datetime object
            if isinstance(dt, str):
                try:
                    dt = dt_cls.fromisoformat(dt)
                except:
                    continue
            
            if isinstance(dt, dt_cls):
                hourly_activity[dt.hour] += 1
                daily_activity[dt.weekday()] += 1
    
    # Ensure all hours/days have entries
    hourly_dict = {i: hourly_activity.get(i, 0) for i in range(24)}
    daily_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_dict = {daily_names[i]: daily_activity.get(i, 0) for i in range(7)}
    
    return {
        "hourly_activity": hourly_dict,
        "daily_activity": daily_dict,
        "peak_hour": max(hourly_dict.items(), key=lambda x: x[1])[0] if hourly_dict else 0,
        "peak_day": max(daily_dict.items(), key=lambda x: x[1])[0] if daily_dict else "Monday"
    }


def analyze_response_times(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze response times between consecutive messages.
    Identify slow and fast responders.
    """
    from datetime import datetime as dt_cls
    
    if len(messages) < 2:
        return {"average_response_time_minutes": 0, "slow_responders": [], "fast_responders": []}
    
    response_times_by_sender = {}  # sender -> list of response times (seconds)
    
    for i in range(1, len(messages)):
        prev_msg = messages[i-1]
        curr_msg = messages[i]
        
        # Skip if same sender (not a response, just continuation)
        if prev_msg.get("sender") == curr_msg.get("sender"):
            continue
        
        prev_dt = prev_msg.get("datetime")
        curr_dt = curr_msg.get("datetime")
        
        # Parse datetime strings if needed
        if prev_dt and isinstance(prev_dt, str):
            try:
                prev_dt = dt_cls.fromisoformat(prev_dt)
            except:
                prev_dt = None
        if curr_dt and isinstance(curr_dt, str):
            try:
                curr_dt = dt_cls.fromisoformat(curr_dt)
            except:
                curr_dt = None
        
        if prev_dt and curr_dt and isinstance(prev_dt, dt_cls) and isinstance(curr_dt, dt_cls) and curr_dt > prev_dt:
            time_diff = (curr_dt - prev_dt).total_seconds()
            responder = curr_msg.get("sender", "Unknown")
            
            if responder not in response_times_by_sender:
                response_times_by_sender[responder] = []
            response_times_by_sender[responder].append(time_diff)
    
    # Calculate averages per sender
    avg_response_times = {}
    for sender, times in response_times_by_sender.items():
        avg_times = sum(times) / len(times)
        avg_response_times[sender] = round(avg_times / 60, 2)  # Convert to minutes
    
    overall_avg = round(sum(avg_response_times.values()) / len(avg_response_times), 2) if avg_response_times else 0
    
    # Determine slow/fast
    if avg_response_times:
        threshold_slow = overall_avg * 1.5
        threshold_fast = overall_avg * 0.5
        slow_responders = {k: v for k, v in avg_response_times.items() if v > threshold_slow}
        fast_responders = {k: v for k, v in avg_response_times.items() if v < threshold_fast}
    else:
        slow_responders = {}
        fast_responders = {}
    
    return {
        "average_response_time_minutes": overall_avg,
        "all_responders": avg_response_times,
        "slow_responders": slow_responders,
        "fast_responders": fast_responders
    }


def analyze_network_graph(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze who talks to whom most.
    Returns interaction matrix showing conversation patterns.
    """
    interactions = Counter()  # (sender, next_speaker) tuples
    sender_counts = Counter()
    
    for i in range(len(messages) - 1):
        curr_sender = messages[i].get("sender", "Unknown")
        next_sender = messages[i+1].get("sender", "Unknown")
        
        if curr_sender != next_sender:  # Only count when sender changes
            interactions[(curr_sender, next_sender)] += 1
            sender_counts[curr_sender] += 1
    
    # Get top interactions as list of dicts
    top_interactions_list = [
        {"from": sender, "to": recipient, "count": count}
        for (sender, recipient), count in interactions.most_common(20)
    ]
    
    # Build adjacency structure
    network = {}
    for item in top_interactions_list:
        sender = item["from"]
        recipient = item["to"]
        count = item["count"]
        if sender not in network:
            network[sender] = {}
        network[sender][recipient] = count
    
    return {
        "interactions": top_interactions_list,
        "network": network,
        "total_unique_speakers": len(sender_counts)
    }


def get_word_cloud_data(messages: List[Dict[str, Any]], top_n: int = 50) -> Dict[str, Any]:
    """
    Extract most frequent words for word cloud visualization.
    """
    all_words = []
    
    for msg in messages:
        text = str(msg.get("message", "")).lower()
        # Remove URLs and media markers
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        # Split and filter
        words = re.findall(r'\b\w+\b', text)
        all_words.extend([w for w in words if w not in STOP_WORDS and len(w) > 2])
    
    word_freq = Counter(all_words)
    top_words = dict(word_freq.most_common(top_n))
    
    return {
        "words": top_words,
        "unique_words": len(word_freq),
        "total_word_count": len(all_words)
    }


def analyze_topics_over_time(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Track how topics/keywords change over months.
    """
    from datetime import datetime as dt_cls
    
    messages_by_month = {}  # month_str -> list of messages
    
    for msg in messages:
        dt = msg.get("datetime")
        if dt:
            # Handle string ISO format or datetime object
            if isinstance(dt, str):
                try:
                    dt = dt_cls.fromisoformat(dt)
                except:
                    continue
            
            if isinstance(dt, dt_cls):
                month_key = dt.strftime("%Y-%m")
                if month_key not in messages_by_month:
                    messages_by_month[month_key] = []
                messages_by_month[month_key].append(msg)
    
    # Get top words for each month
    topics_by_month = {}
    for month, month_messages in sorted(messages_by_month.items()):
        all_words = []
        for msg in month_messages:
            text = str(msg.get("message", "")).lower()
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'<.*?>', '', text)
            words = re.findall(r'\b\w+\b', text)
            all_words.extend([w for w in words if w not in STOP_WORDS and len(w) > 2])
        
        word_freq = Counter(all_words)
        topics_by_month[month] = dict(word_freq.most_common(10))
    
    return {
        "topics_by_month": topics_by_month,
        "total_months": len(topics_by_month)
    }


def analyze_message_length_distribution(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze message length distribution by sender.
    Identify who writes short vs long messages.
    """
    sender_messages = {}  # sender -> list of message lengths
    
    for msg in messages:
        sender = msg.get("sender", "Unknown")
        msg_text = str(msg.get("message", ""))
        msg_len = len(msg_text)
        
        if sender not in sender_messages:
            sender_messages[sender] = []
        sender_messages[sender].append(msg_len)
    
    # Calculate stats per sender
    sender_stats = {}
    for sender, lengths in sender_messages.items():
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        min_len = min(lengths) if lengths else 0
        long_msg_ratio = sum(1 for l in lengths if l > 100) / len(lengths) * 100 if lengths else 0
        short_msg_ratio = sum(1 for l in lengths if l < 20) / len(lengths) * 100 if lengths else 0
        
        sender_stats[sender] = {
            "average_length": round(avg_len, 2),
            "max_length": max_len,
            "min_length": min_len,
            "total_messages": len(lengths),
            "long_message_ratio": round(long_msg_ratio, 2),
            "short_message_ratio": round(short_msg_ratio, 2)
        }
    
    return {
        "per_sender": sender_stats,
        "longest_average": max((v["average_length"], k) for k, v in sender_stats.items())[1] if sender_stats else None,
        "shortest_average": min((v["average_length"], k) for k, v in sender_stats.items())[1] if sender_stats else None
    }


def detect_repeated_phrases(messages: List[Dict[str, Any]], min_occurrences: int = 3) -> Dict[str, Any]:
    """
    Detect recurring messages, memes, and inside jokes.
    """
    # Group by exact message text
    message_freq = Counter()
    message_senders = {}  # message -> list of senders who said it
    
    for msg in messages:
        text = str(msg.get("message", "")).strip()
        sender = msg.get("sender", "Unknown")
        
        # Only consider reasonably short messages (likely to be memes/jokes)
        if 2 < len(text) < 200:
            message_freq[text] += 1
            if text not in message_senders:
                message_senders[text] = []
            message_senders[text].append(sender)
    
    # Get messages that appear at least min_occurrences times
    repeated = {msg: count for msg, count in message_freq.most_common(50) 
                if count >= min_occurrences}
    
    result = {}
    for msg, count in sorted(repeated.items(), key=lambda x: x[1], reverse=True):
        senders = message_senders.get(msg, [])
        unique_senders = list(set(senders))
        result[msg] = {
            "count": count,
            "senders": unique_senders,
            "unique_senders": len(unique_senders)
        }
    
    return {
        "repeated_phrases": result,
        "total_unique_repeated": len(result)
    }
