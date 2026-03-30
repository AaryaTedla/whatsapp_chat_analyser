"""Database models."""
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from core.database import Base


class Report(Base):
    """WhatsApp chat analysis report."""
    
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    message_count = Column(Integer, nullable=False)
    txt_content = Column(Text, nullable=False)
    stats_json = Column(Text, nullable=False)  # JSON string
    ai_json = Column(Text, nullable=False)  # JSON string
    
    def to_dict(self, include_content=False):
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "filename": self.filename,
            "uploaded_at": self.uploaded_at.isoformat(),
            "message_count": self.message_count,
        }
        if include_content:
            data["txt_content"] = self.txt_content
            data["stats"] = self.stats_json
            data["ai"] = self.ai_json
        return data
