import json
from typing import Dict, List
from datetime import datetime

class TelemetryService:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "class_distribution": {"toxic": 0, "spam": 0, "safe": 0},
            "feedback_counts": {"positive": 0, "negative": 0},
            "latencies": []
        }
        self.feedback_data = []
    
    def record_classification(self, classification: str, latency_ms: int):
        self.metrics["total_requests"] += 1
        self.metrics["class_distribution"][classification] += 1
        self.metrics["latencies"].append(latency_ms)
    
    def record_feedback(self, text: str, predicted: str, correct: str):
        feedback = {
            "text": text,
            "predicted": predicted,
            "correct": correct,
            "timestamp": datetime.now().isoformat()
        }
        self.feedback_data.append(feedback)
        
        # Update feedback counts
        if predicted == correct:
            self.metrics["feedback_counts"]["positive"] += 1
        else:
            self.metrics["feedback_counts"]["negative"] += 1
    
    def get_metrics(self) -> Dict:
        latencies = self.metrics["latencies"]
        return {
            "total_requests": self.metrics["total_requests"],
            "class_distribution": self.metrics["class_distribution"],
            "feedback_counts": self.metrics["feedback_counts"],
            "latency": {
                "avg_ms": sum(latencies) / len(latencies) if latencies else 0
            }
        }