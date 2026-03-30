"""FastAPI application with best practices and PWA support."""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from analyser import compute_stats, generate_ai_summary
from core.config import UPLOAD_DIR, MAX_UPLOAD_SIZE, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from core.database import get_db, init_db
from core.models import Report
from parser import parse_whatsapp_export
from routes_advanced import router as advanced_router

# Initialize database
init_db()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Lumira - WhatsApp Chat Analyzer",
    description="Professional AI-powered WhatsApp chat analysis",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include advanced routes
app.include_router(advanced_router)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    logger.warning("Static directory not found, skipping mount")


# Pydantic models
class ReportSummary(BaseModel):
    """Summary of a report without full content."""
    id: int
    filename: str
    uploaded_at: datetime
    message_count: int
    
    class Config:
        from_attributes = True


class ReportDetail(ReportSummary):
    """Full report details including analysis."""
    txt_content: str
    stats: dict
    ai: dict
    
    class Config:
        from_attributes = True


class AnalysisResponse(BaseModel):
    """Response from analysis endpoint."""
    id: int
    filename: str
    message_count: int
    stats: dict
    ai: dict
    uploaded_at: datetime


def _decode_uploaded_bytes(raw: bytes) -> str:
    """Try UTF-8 first, then fallback to latin-1."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="ignore")


def _save_report(
    *,
    filename: str,
    txt_content: str,
    message_count: int,
    stats: dict,
    ai: dict,
    db: Session,
) -> Report:
    """Save a report to the database."""
    report = Report(
        filename=filename,
        message_count=message_count,
        txt_content=txt_content,
        stats_json=json.dumps(stats, ensure_ascii=True),
        ai_json=json.dumps(ai, ensure_ascii=True),
        uploaded_at=datetime.now(timezone.utc),
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


# Routes
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve main page."""
    try:
        with open("templates/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Welcome to Lumira</h1><p>Upload a WhatsApp export to get started.</p>"


@app.get("/reports", response_class=HTMLResponse)
async def reports_page():
    """Serve saved reports page."""
    try:
        with open("templates/reports.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Saved Reports</h1><p>Reports page template not found.</p>"


@app.post("/analyze", response_model=AnalysisResponse, include_in_schema=False)
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Analyze a WhatsApp export file.
    
    - **file**: WhatsApp .txt export file
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    if not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Please upload a .txt WhatsApp export")
    
    # Read and decode file
    try:
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 20MB)")
        
        text = _decode_uploaded_bytes(raw)
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")
    
    # Parse messages
    try:
        messages = parse_whatsapp_export(text)
        if not messages:
            raise HTTPException(status_code=400, detail="No valid messages found in file")
    except Exception as e:
        logger.error(f"Error parsing WhatsApp export: {e}")
        raise HTTPException(status_code=400, detail=f"Could not parse WhatsApp export: {e}")
    
    # Compute statistics
    try:
        stats = compute_stats(messages)
    except Exception as e:
        logger.error(f"Error computing stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error computing statistics: {e}")
    
    # Generate AI summary
    try:
        ai_summary = generate_ai_summary(messages)
    except Exception as e:
        logger.error(f"Error generating AI summary: {e}")
        ai_summary = {
            "vibe_summary": "Unable to generate AI summary",
            "top_3_topics": [],
            "funny_observation": "",
            "error": str(e),
        }
    
    # Save to database
    try:
        report = _save_report(
            filename=file.filename,
            txt_content=text,
            message_count=len(messages),
            stats=stats,
            ai=ai_summary,
            db=db,
        )
        logger.info(f"Saved report {report.id} from {file.filename}")
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving report: {e}")
    
    return AnalysisResponse(
        id=report.id,
        filename=report.filename,
        message_count=report.message_count,
        stats=stats,
        ai=ai_summary,
        uploaded_at=report.uploaded_at,
    )


@app.get("/api/reports", response_model=list[ReportSummary])
async def list_reports(
    limit: int = Query(DEFAULT_PAGE_SIZE, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List saved reports with pagination.
    
    - **limit**: Number of reports (max 200)
    - **offset**: Skip this many reports
    """
    reports = db.query(Report).order_by(Report.id.desc()).offset(offset).limit(limit).all()
    return [ReportSummary.from_orm(r) for r in reports]


@app.get("/api/reports/{report_id}", response_model=ReportDetail)
async def get_report(report_id: int, db: Session = Depends(get_db)):
    """Get full report details."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    detail_dict = {
        "id": report.id,
        "filename": report.filename,
        "uploaded_at": report.uploaded_at.isoformat(),
        "message_count": report.message_count,
        "txt_content": report.txt_content,
        "stats": json.loads(report.stats_json),
        "ai": json.loads(report.ai_json),
    }
    return ReportDetail(**detail_dict)


@app.get("/reports/{report_id}", response_model=ReportDetail, include_in_schema=False)
async def get_report_legacy(report_id: int, db: Session = Depends(get_db)):
    """Legacy alias for report JSON detail endpoint."""
    return await get_report(report_id=report_id, db=db)


@app.get("/r/{report_id}", response_class=HTMLResponse)
async def view_report_page(report_id: int, db: Session = Depends(get_db)):
    """Serve report view page."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return "<h1>Report Not Found</h1>"
    
    try:
        with open("templates/report_view.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"<h1>Report {report_id}</h1><p>View not available</p>"


@app.get("/analytics/{report_id}", response_class=HTMLResponse)
async def view_analytics_page(report_id: int, db: Session = Depends(get_db)):
    """Serve comprehensive analytics dashboard page."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        return "<h1>Report Not Found</h1>"
    
    try:
        with open("templates/analytics_dashboard.html", "r") as f:
            content = f.read()
            # Replace the report ID in the JavaScript
            return content
    except FileNotFoundError:
        return f"<h1>Analytics for Report {report_id}</h1><p>Analytics not available</p>"


@app.get("/api/reports/{report_id}/download")
async def download_report_txt(report_id: int, db: Session = Depends(get_db)):
    """Download original WhatsApp export."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Return JSON with content
    filename = f"{report.filename.replace('.txt', '')}_lumira_{report_id}.txt"
    return JSONResponse(
        content={"content": report.txt_content, "filename": filename}
    )


@app.get("/reports/{report_id}/txt", include_in_schema=False)
async def download_report_txt_legacy(report_id: int, db: Session = Depends(get_db)):
    """Legacy alias for TXT download with attachment response."""
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    filename = f"{report.filename.replace('.txt', '')}_lumira_{report_id}.txt"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return Response(content=report.txt_content, media_type="text/plain; charset=utf-8", headers=headers)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
