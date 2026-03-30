"""
Advanced API v2 routes for WhatsApp Chat Analyzer
Provides sophisticated analytics and filtering capabilities
"""

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from core.database import get_db
from core.models import Report
import json
from typing import Optional, List
from parser import parse_whatsapp_export
from analyser import (
    analyze_sentiment, analyze_emojis, analyze_mentions,
    analyze_links_and_media, detect_languages, analyze_response_patterns,
    analyze_conversation_heatmap, analyze_response_times, analyze_network_graph,
    get_word_cloud_data, analyze_topics_over_time, analyze_message_length_distribution,
    detect_repeated_phrases
)
from utils import (
    search_messages, filter_by_message_type, generate_pdf_report,
    filter_messages_by_date_range, filter_messages_by_sender,
    generate_comparison_report
)

router = APIRouter(prefix="/api/v2", tags=["advanced"])


@router.get("/reports/stats")
async def get_reports_stats(db: Session = Depends(get_db)):
    """
    Get aggregated statistics across all reports
    - Total reports, average message count, activity trends
    """
    reports = db.query(Report).all()
    if not reports:
        return {
            "total_reports": 0,
            "total_messages": 0,
            "average_messages": 0,
            "date_range": None
        }
    
    total_messages = sum(r.message_count for r in reports)
    dates = [r.uploaded_at for r in reports if r.uploaded_at]
    
    return {
        "total_reports": len(reports),
        "total_messages": total_messages,
        "average_messages": total_messages // len(reports) if reports else 0,
        "date_range": {
            "earliest": min(dates).isoformat() if dates else None,
            "latest": max(dates).isoformat() if dates else None
        },
        "reports": [
            {
                "id": r.id,
                "filename": r.filename,
                "message_count": r.message_count,
                "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None
            }
            for r in reports
        ]
    }


@router.get("/reports/{report_id}/advanced-stats")
async def get_advanced_stats(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Get advanced statistics for a specific report
    - Sentiment analysis readiness, conversation density, participant engagement
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    try:
        stats = json.loads(report.stats_json) if report.stats_json else {}
    except:
        stats = {}
    
    # Calculate derived metrics
    avg_message_length = 0
    conversation_density = "N/A"
    
    if report.message_count > 0:
        # Estimate average message length from text
        if report.txt_content:
            avg_message_length = len(report.txt_content) // report.message_count
        
        # Conversation density: messages per day
        if report.uploaded_at:
            days_active = max((datetime.utcnow() - report.uploaded_at).days, 1)
            conversation_density = f"{report.message_count // days_active} msgs/day"
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "advanced_metrics": {
            "total_messages": report.message_count,
            "estimated_avg_message_length": avg_message_length,
            "conversation_density": conversation_density,
            "participation_data": stats.get("senders", [])[:10] if isinstance(stats.get("senders"), list) else []
        },
        "base_stats": stats
    }


@router.get("/reports/{report_id}/sentiment")
async def get_sentiment_analysis(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Get sentiment analysis for a report.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    messages = parse_whatsapp_export(report.txt_content)
    sentiment = analyze_sentiment(messages)
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "sentiment_analysis": sentiment
    }


@router.get("/reports/{report_id}/emojis")
async def get_emoji_analysis(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Get emoji analysis for a report.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    messages = parse_whatsapp_export(report.txt_content)
    emojis = analyze_emojis(messages)
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "emoji_analysis": emojis
    }


@router.get("/reports/{report_id}/mentions")
async def get_mentions_analysis(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Get mention analysis for a report.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    messages = parse_whatsapp_export(report.txt_content)
    mentions = analyze_mentions(messages)
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "mention_analysis": mentions
    }


@router.get("/reports/{report_id}/media")
async def get_media_analysis(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Get media and link analysis for a report.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    messages = parse_whatsapp_export(report.txt_content)
    media = analyze_links_and_media(messages)
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "media_analysis": media
    }


@router.get("/reports/{report_id}/languages")
async def get_language_detection(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Get language detection for a report.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    messages = parse_whatsapp_export(report.txt_content)
    languages = detect_languages(messages)
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "language_detection": languages
    }


@router.get("/reports/{report_id}/response-patterns")
async def get_response_patterns(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Get response time and conversation patterns for a report.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    messages = parse_whatsapp_export(report.txt_content)
    patterns = analyze_response_patterns(messages)
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "response_patterns": patterns
    }


@router.get("/reports/{report_id}/search")
async def search_report(
    report_id: int,
    query: str = Query(..., description="Search query"),
    db: Session = Depends(get_db)
):
    """
    Full-text search within a report.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    messages = parse_whatsapp_export(report.txt_content)
    results = search_messages(messages, query)
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "query": query,
        "results_count": len(results),
        "results": results[:100]  # Return top 100 results
    }


@router.get("/reports/{report_id}/filter")
async def filter_report_messages(
    report_id: int,
    message_type: str = Query("text", description="text, media, link, emoji"),
    sender: Optional[str] = Query(None, description="Optional sender filter"),
    db: Session = Depends(get_db)
):
    """
    Filter report messages by type and optionally by sender.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    messages = parse_whatsapp_export(report.txt_content)
    
    # Filter by message type
    filtered = filter_by_message_type(messages, message_type)
    
    # Filter by sender if specified
    if sender:
        filtered = [m for m in filtered if m.get("sender") == sender]
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "filter_type": message_type,
        "sender_filter": sender,
        "results_count": len(filtered),
        "results": filtered[:100]
    }


@router.get("/reports/{report_id}/pdf")
async def download_pdf_report(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Download report as a text/PDF file.
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    try:
        stats = json.loads(report.stats_json) if report.stats_json else {}
        ai_data = json.loads(report.ai_json) if report.ai_json else {}
    except:
        stats = {}
        ai_data = {}
    
    report_bytes = generate_pdf_report(
        report.filename,
        stats,
        ai_data,
        report.message_count
    )
    
    from fastapi.responses import StreamingResponse
    from io import BytesIO
    
    return StreamingResponse(
        content=BytesIO(report_bytes),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={report.filename.replace('.txt', '')}_report.txt"}
    )


@router.get("/reports/{report_id}/timeline")
async def get_activity_timeline(
    report_id: int,
    granularity: str = Query("day", description="hour, day, week, month"),
    db: Session = Depends(get_db)
):
    """
    Get activity timeline for a report at different granularities
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    try:
        stats = json.loads(report.stats_json) if report.stats_json else {}
    except:
        stats = {}
    
    return {
        "report_id": report_id,
        "granularity": granularity,
        "hourly_data": stats.get("hours", []) if granularity in ["hour", "day"] else None,
        "daily_data": stats.get("days", []) if granularity in ["day", "week", "month"] else None,
        "message_count": report.message_count
    }


@router.get("/reports/{report_id}/insights")
async def get_ai_insights(
    report_id: int,
    db: Session = Depends(get_db)
):
    """
    Get AI-generated insights and summary for a report
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}, 404
    
    try:
        ai_data = json.loads(report.ai_json) if report.ai_json else {}
    except:
        ai_data = {}
    
    return {
        "report_id": report_id,
        "filename": report.filename,
        "insights": {
            "vibe": ai_data.get("vibe_summary", "N/A"),
            "topics": ai_data.get("top_3_topics", []),
            "funny_observation": ai_data.get("funny_observation", "N/A")
        },
        "generated_at": report.uploaded_at.isoformat() if report.uploaded_at else None
    }


@router.post("/reports/compare")
async def compare_reports(
    report_ids: List[int] = Query(..., description="List of report IDs to compare"),
    db: Session = Depends(get_db)
):
    """
    Compare statistics across multiple reports
    """
    reports = db.query(Report).filter(Report.id.in_(report_ids)).all()
    if not reports:
        return {"error": "No reports found"}, 404
    
    comparison = {
        "compared_reports": len(reports),
        "reports": []
    }
    
    total_messages = 0
    for report in reports:
        comparison["reports"].append({
            "id": report.id,
            "filename": report.filename,
            "message_count": report.message_count,
            "uploaded_at": report.uploaded_at.isoformat() if report.uploaded_at else None
        })
        total_messages += report.message_count
    
    comparison["total_messages"] = total_messages
    comparison["average_messages"] = total_messages // len(reports) if reports else 0
    
    return comparison


@router.get("/reports/{report_id}/heatmap")
async def get_conversation_heatmap(report_id: int, db: Session = Depends(get_db)):
    """
    Get hourly and daily activity heatmap
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}
    
    try:
        messages = parse_whatsapp_export(report.txt_content)
        heatmap = analyze_conversation_heatmap(messages)
        return heatmap
    except Exception as e:
        return {"error": str(e)}


@router.get("/reports/{report_id}/response-times")
async def get_response_times(report_id: int, db: Session = Depends(get_db)):
    """
    Get response time analytics and identify slow/fast responders
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}
    
    try:
        messages = parse_whatsapp_export(report.txt_content)
        response_times = analyze_response_times(messages)
        return response_times
    except Exception as e:
        return {"error": str(e)}


@router.get("/reports/{report_id}/network")
async def get_network_graph(report_id: int, db: Session = Depends(get_db)):
    """
    Get participant network graph showing who talks to whom
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}
    
    try:
        messages = parse_whatsapp_export(report.txt_content)
        network = analyze_network_graph(messages)
        return network
    except Exception as e:
        return {"error": str(e)}


@router.get("/reports/{report_id}/wordcloud")
async def get_word_cloud(report_id: int, top_n: int = Query(50), db: Session = Depends(get_db)):
    """
    Get word cloud data with most frequent words
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}
    
    try:
        messages = parse_whatsapp_export(report.txt_content)
        wordcloud = get_word_cloud_data(messages, top_n=top_n)
        return wordcloud
    except Exception as e:
        return {"error": str(e)}


@router.get("/reports/{report_id}/topics-timeline")
async def get_topics_timeline(report_id: int, db: Session = Depends(get_db)):
    """
    Get topics and keywords over time by month
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}
    
    try:
        messages = parse_whatsapp_export(report.txt_content)
        topics = analyze_topics_over_time(messages)
        return topics
    except Exception as e:
        return {"error": str(e)}


@router.get("/reports/{report_id}/message-lengths")
async def get_message_lengths(report_id: int, db: Session = Depends(get_db)):
    """
    Get message length distribution by sender
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}
    
    try:
        messages = parse_whatsapp_export(report.txt_content)
        lengths = analyze_message_length_distribution(messages)
        return lengths
    except Exception as e:
        return {"error": str(e)}


@router.get("/reports/{report_id}/repeated-phrases")
async def get_repeated_phrases(report_id: int, min_occurrences: int = Query(3), db: Session = Depends(get_db)):
    """
    Get repeated phrases, memes, and inside jokes
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return {"error": "Report not found"}
    
    try:
        messages = parse_whatsapp_export(report.txt_content)
        phrases = detect_repeated_phrases(messages, min_occurrences=min_occurrences)
        return phrases
    except Exception as e:
        return {"error": str(e)}


@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """
    Detailed health check with database and cache status
    """
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    report_count = db.query(Report).count()
    
    return {
        "status": "ok" if db_status == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "database": {
            "status": db_status,
            "total_reports": report_count
        },
        "version": "2.0"
    }
