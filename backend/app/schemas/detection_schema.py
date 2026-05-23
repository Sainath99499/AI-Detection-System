from pydantic import BaseModel

class DetectionResponse(BaseModel):
    content_type: str
    ai_probability: float
    human_probability: float
    confidence: str