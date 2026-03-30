"""Advanced analysis utilities - new features for v2.0."""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import polars as pl


def filter_messages_by_date_range(
    messages: List[Dict[str, Any]],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Filter messages by date range (ISO format: YYYY-MM-DD).
    
    Returns messages within the specified range.
    """
    if not start_date and not end_date:
        return messages
    
    df = pl.DataFrame(messages)
    df = df.filter(pl.col("datetime").is_not_null())
    df = df.with_columns(
        pl.col("datetime").str.to_datetime().dt.date().alias("date_only")
    )
    
    if start_date:
        start = datetime.fromisoformat(start_date).date()
        df = df.filter(pl.col("date_only") >= start)
    
    if end_date:
        end = datetime.fromisoformat(end_date).date()
        df = df.filter(pl.col("date_only") <= end)
    
    return df.to_dicts()


def filter_messages_by_sender(
    messages: List[Dict[str, Any]],
    senders: List[str],
) -> List[Dict[str, Any]]:
    """Filter messages from specific senders."""
    if not senders:
        return messages
    
    return [m for m in messages if m.get("sender") in senders]


def analyze_sender_activity(
    messages: List[Dict[str, Any]],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Advanced sender activity analysis.
    """
    if not messages:
        return {}
    
    df = pl.DataFrame(messages)
    
    sender_stats = (
        df.group_by("sender")
        .agg([
            pl.len().alias("total_messages"),
            pl.col("message").str.lengths().mean().alias("avg_message_length"),
            pl.col("message").str.lengths().max().alias("max_message_length"),
        ])
        .sort("total_messages", descending=True)
        .limit(top_n)
        .to_dicts()
    )
    
    return {
        "total_senders": df.select(pl.col("sender")).n_unique(),
        "active_senders": sender_stats,
    }


def detect_message_patterns(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Detect conversation patterns.
    """
    if not messages:
        return {}
    
    text_blocks = []
    current_block = []
    current_sender = None
    
    for msg in messages:
        sender = msg.get("sender")
        if sender == current_sender and current_block:
            current_block.append(msg.get("message", ""))
        else:
            if current_block:
                text_blocks.append({
                    "sender": current_sender,
                    "messages": current_block,
                    "count": len(current_block),
                })
            current_block = [msg.get("message", "")]
            current_sender = sender
    
    if current_block:
        text_blocks.append({
            "sender": current_sender,
            "messages": current_block,
            "count": len(current_block),
        })
    
    # Find longest monologues (consecutive messages from one sender)
    longest = sorted(text_blocks, key=lambda x: x["count"], reverse=True)[:3]
    
    return {
        "total_blocks": len(text_blocks),
        "longest_monologues": longest,
        "avg_block_size": sum(b["count"] for b in text_blocks) / len(text_blocks) if text_blocks else 0,
    }


def export_to_json(
    report_data: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Export report to JSON format."""
    export_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "2.0.0",
        },
        "report": report_data,
    }
    
    if messages:
        export_data["messages"] = messages
    
    return json.dumps(export_data, indent=2, ensure_ascii=True)


def export_to_csv(
    messages: List[Dict[str, Any]],
    include_fields: List[str] = ["datetime", "sender", "message"],
) -> str:
    """Export messages to CSV format."""
    import csv
    from io import StringIO
    
    if not messages:
        return ""
    
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=include_fields)
    writer.writeheader()
    
    for msg in messages:
        row = {field: msg.get(field, "") for field in include_fields}
        writer.writerow(row)
    
    return output.getvalue()


def get_time_statistics(
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Comprehensive time-based statistics.
    """
    if not messages:
        return {}
    
    df = pl.DataFrame(messages)
    df = df.filter(pl.col("datetime").is_not_null())
    
    if len(df) == 0:
        return {}
    
    df = df.with_columns(
        pl.col("datetime").str.to_datetime(),
    )
    
    # Day of week analysis
    dow_stats = (
        df.with_columns(
            day_of_week=pl.col("datetime").dt.strftime("%A"),
        )
        .group_by("day_of_week")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .to_dicts()
    )
    
    # Day vs night (6am-6pm vs rest)
    day_night = (
        df.with_columns(
            is_day=pl.col("datetime").dt.hour().is_in(range(6, 18)),
        )
        .group_by("is_day")
        .agg(pl.len().alias("count"))
        .to_dicts()
    )
    
    return {
        "day_of_week_distribution": dow_stats,
        "day_night_split": day_night,
    }


def get_growth_metrics(
    messages: List[Dict[str, Any]],
    period_days: int = 30,
) -> Dict[str, Any]:
    """
    Calculate growth/engagement metrics over time.
    """
    if not messages:
        return {}
    
    df = pl.DataFrame(messages)
    df = df.filter(pl.col("datetime").is_not_null())
    
    if len(df) == 0:
        return {}
    
    df = df.with_columns(
        pl.col("datetime").str.to_datetime(),
    )
    
    # Get date range
    min_date = df.select(pl.col("datetime").min()).item()
    max_date = df.select(pl.col("datetime").max()).item()
    
    # Split into periods
    total_days = (max_date - min_date).days
    if total_days < 1:
        return {}
    
    periods = []
    current = min_date
    
    while current < max_date:
        period_end = current + timedelta(days=period_days)
        count = len(df.filter(
            (pl.col("datetime") >= current) &
            (pl.col("datetime") < period_end)
        ))
        
        if count > 0:
            periods.append({
                "period_start": current.isoformat(),
                "messages": count,
            })
        
        current = period_end
    
    return {
        "total_days": total_days,
        "period_length_days": period_days,
        "periods": periods,
    }


def search_messages(
    messages: List[Dict[str, Any]],
    query: str,
    case_sensitive: bool = False,
) -> List[Dict[str, Any]]:
    """
    Full-text search across messages.
    Returns messages matching the search query.
    """
    if not query or not messages:
        return []
    
    search_term = query if case_sensitive else query.lower()
    results = []
    
    for msg in messages:
        text = str(msg.get("message", ""))
        if not case_sensitive:
            text = text.lower()
        
        if search_term in text:
            results.append(msg)
    
    return results


def filter_by_message_type(
    messages: List[Dict[str, Any]],
    message_type: str = "text",  # text, media, link, emoji
) -> List[Dict[str, Any]]:
    """
    Filter messages by type (text, media, links, emoji-heavy).
    """
    import re
    from collections import Counter
    
    if message_type == "text":
        return [m for m in messages if m.get("message") and str(m.get("message")).strip()]
    
    elif message_type == "media":
        media_indicators = ["<image omitted>", "<video omitted>", "<audio omitted>", "<document omitted>"]
        return [m for m in messages if any(ind in str(m.get("message", "")) for ind in media_indicators)]
    
    elif message_type == "link":
        url_pattern = re.compile(r"http[s]?://")
        return [m for m in messages if url_pattern.search(str(m.get("message", "")))]
    
    elif message_type == "emoji":
        emoji_pattern = re.compile(
            r"["
            r"\U0001F600-\U0001F64F"
            r"\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF"
            r"\U0001F700-\U0001F77F"
            r"\U0001F800-\U0001F8FF"
            r"\U0001F900-\U0001F9FF"
            r"\U0001FA00-\U0001FA6F"
            r"\U0001FA70-\U0001FAFF"
            r"]+",
            flags=re.UNICODE
        )
        return [m for m in messages if emoji_pattern.search(str(m.get("message", "")))]
    
    return messages


def generate_pdf_report(
    filename: str,
    stats: Dict[str, Any],
    ai_summary: Dict[str, Any],
    message_count: int,
) -> bytes:
    """
    Generate a simple text-based report (as bytes).
    Returns report data that can be formatted as PDF by frontend or served as text.
    """
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("WhatsApp Chat Analysis Report")
    report_lines.append("=" * 60)
    report_lines.append(f"\nFile: {filename}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Messages: {message_count}")
    report_lines.append(f"Unique Senders: {stats.get('unique_senders', 0)}")
    
    # Top Senders
    if stats.get("senders"):
        report_lines.append("\n" + "=" * 60)
        report_lines.append("Top Senders")
        report_lines.append("=" * 60)
        senders = stats.get("senders", {})
        for sender, count in zip(senders.get("labels", []), senders.get("counts", [])):
            report_lines.append(f"{sender}: {count} messages")
    
    # AI Summary
    report_lines.append("\n" + "=" * 60)
    report_lines.append("AI Analysis")
    report_lines.append("=" * 60)
    if ai_summary.get("vibe_summary"):
        report_lines.append(f"\nVibe: {ai_summary.get('vibe_summary', 'N/A')}")
    if ai_summary.get("top_3_topics"):
        topics = ", ".join(ai_summary.get("top_3_topics", []))
        report_lines.append(f"Topics: {topics}")
    if ai_summary.get("funny_observation"):
        report_lines.append(f"Observation: {ai_summary.get('funny_observation', 'N/A')}")
    
    # Top Words
    if stats.get("top_words"):
        report_lines.append("\n" + "=" * 60)
        report_lines.append("Top Words")
        report_lines.append("=" * 60)
        words = stats.get("top_words", [])[:15]
        for word_obj in words:
            if isinstance(word_obj, dict):
                report_lines.append(f"{word_obj.get('word', '')}: {word_obj.get('count', '')}")
    
    report_lines.append("\n" + "=" * 60)
    report_text = "\n".join(report_lines)
    
    return report_text.encode('utf-8')


def generate_comparison_report(
    reports: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a comparison report across multiple reports.
    """
    if not reports:
        return {}
    
    comparison = {
        "total_reports": len(reports),
        "total_messages": sum(r.get("message_count", 0) for r in reports),
        "avg_messages": 0,
        "reports": []
    }
    
    if reports:
        comparison["avg_messages"] = comparison["total_messages"] / len(reports)
    
    for report in reports:
        comparison["reports"].append({
            "filename": report.get("filename"),
            "message_count": report.get("message_count"),
            "uploaded_at": str(report.get("uploaded_at"))
        })
    
    return comparison
