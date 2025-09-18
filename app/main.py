import uvicorn
from fastapi import FastAPI, HTTPException
from app.services.classifier import TextClassifier
from app.telemetry.telemetry import TelemetryService
from eval.evalution import ModelEvaluator
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ClassifyRequest(BaseModel):
    text: str

class ClassifyResponse(BaseModel):
    class_: str = Field(..., alias="class") 
    prompt_used: str
    latency_ms: int
    class Config:
        populate_by_name = True

class FeedbackRequest(BaseModel):
    text: str
    predicted: str
    correct: str

class MetricsResponse(BaseModel):
    total_requests: int
    class_distribution: Dict[str, int]
    feedback_counts: Dict[str, int]
    latency: Dict[str, float]

classifier = TextClassifier()
telemetry = TelemetryService()

app = FastAPI(title="LLM Text Classification API", version="1.0.0")

@app.post("/classify", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest):
    try:
        classification, prompt_used, latency_ms = await classifier.classify(request.text)
        
        # Record metrics
        telemetry.record_classification(classification, latency_ms)
        
        return ClassifyResponse(
            **{"class": classification},
            prompt_used=prompt_used,
            latency_ms=latency_ms
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    telemetry.record_feedback(request.text, request.predicted, request.correct)
    return {"status": "feedback recorded"}

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    return telemetry.get_metrics()

@app.get("/evaluation")
async def get_evaluation():
    model_evaluator = ModelEvaluator()  # Create instance (no await needed)
    result = await model_evaluator.run_full_evaluation()  # Await the async method
    return result

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)