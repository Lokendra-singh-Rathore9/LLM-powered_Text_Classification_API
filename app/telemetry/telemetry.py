import json
import os
from typing import Dict
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
        feedback_file = r"app\\data\\feedback_data.json"


        # If file exists, load old data first
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Append new feedback
        existing_data.append(feedback)
        print("executed")
        # Write back to file
        with open(feedback_file, "w") as f:
            json.dump(existing_data, f, indent=4)
            
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