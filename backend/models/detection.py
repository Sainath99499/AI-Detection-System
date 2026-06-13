from sqlalchemy import Column, Integer, String, Float
from app.database.database import Base

class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    content_type = Column(String)
    ai_probability = Column(Float)
    human_probability = Column(Float)
    confidence = Column(String)