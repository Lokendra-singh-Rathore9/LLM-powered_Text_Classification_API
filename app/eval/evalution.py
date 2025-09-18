import asyncio
from typing import Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import time
from app.services.classifier import TextClassifier

class ModelEvaluator:
    """Handles evaluation of text classification models"""
    
    def __init__(self):
        self.classifier = TextClassifier()
        self.test_dataset = [
            {"text": "Great product, highly recommend!", "label": "safe"},
            {"text": "You're an idiot and should die", "label": "toxic"},
            {"text": "Buy cheap pills now! Click here!", "label": "spam"},
            {"text": "Thanks for the helpful tutorial", "label": "safe"},
            {"text": "F*ck you, piece of sh*t!", "label": "toxic"},
            {"text": "Make $5000/week from home! Limited offer!", "label": "spam"},
            {"text": "I disagree with your opinion but respect it", "label": "safe"},
            # {"text": "Kill yourself, nobody likes you", "label": "toxic"},
            # {"text": "URGENT! Free iPhone! Click now!", "label": "spam"},
            # {"text": "This movie was really entertaining", "label": "safe"},
            # {"text": "Nice work on the project!", "label": "safe"},
            # {"text": "Go kill yourself you worthless trash", "label": "toxic"},
            # {"text": "CLICK HERE FOR FREE MONEY!!!", "label": "spam"},
            # {"text": "I enjoyed reading your article", "label": "safe"},
            # {"text": "You should die in a fire", "label": "toxic"},
            # {"text": "Win a FREE car! Act now!", "label": "spam"},
            # {"text": "The weather is nice today", "label": "safe"},
            # {"text": "I hope you get cancer and die", "label": "toxic"},
            # {"text": "Earn $$ fast! No experience needed!", "label": "spam"},
            # {"text": "Thank you for your help", "label": "safe"}
        ]
    
    async def evaluate_single_prompt(self, prompt_type: str) -> Dict:
        """Evaluate a single prompt type"""
        print(f"Evaluating {prompt_type} prompt...")
        
        predictions = []
        true_labels = []
        latencies = []
        errors = []
        
        for i, item in enumerate(self.test_dataset):
            try:
                start_time = time.time()
                classification,used_prompt, latency_ms = await self.classifier.classify(
                    item["text"], 
                    prompt_type=prompt_type
                )
                actual_latency = int((time.time() - start_time) * 1000)
                
                predictions.append(classification)
                true_labels.append(item["label"])
                latencies.append(actual_latency)
                
                print(f"  [{i+1:2d}/{len(self.test_dataset)}] {classification:5} | {item['label']:5} | {actual_latency:3d}ms | {item['text'][:50]}")
                
            except Exception as e:
                error_msg = f"Item {i}: {str(e)}"
                errors.append(error_msg)
                predictions.append("safe")  # Default fallback
                true_labels.append(item["label"])
                latencies.append(1000)  # Penalty latency for errors
                print(f"  [{i+1:2d}/{len(self.test_dataset)}] ERROR: {error_msg}")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # Detailed classification report
        class_report = classification_report(
            true_labels, predictions, output_dict=True, zero_division=0
        )
        
        return {
            "prompt_type": prompt_type,
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "total_examples": len(self.test_dataset),
            "errors_count": len(errors),
            "class_report": class_report,
            "predictions": predictions,
            "true_labels": true_labels,
            "latencies": latencies,
            "errors": errors if errors else None
        }
    
    async def run_full_evaluation(self) -> Dict:
        """Run evaluation on both baseline and advanced prompts"""
        print("="*60)
        print("Starting Model Evaluation")
        print("="*60)
        print(f"Dataset: {len(self.test_dataset)} examples")
        print(f"Classes: {set(item['label'] for item in self.test_dataset)}")
        print("-"*60)
        
        results = {}
        
        # Evaluate both prompt types
        for prompt_type in ["baseline", "advanced"]:
            results[prompt_type] = await self.evaluate_single_prompt(prompt_type)
            print(f"\n{prompt_type.upper()} RESULTS:")
            print(f"  Accuracy:  {results[prompt_type]['accuracy']:.3f}")
            print(f"  Precision: {results[prompt_type]['precision']:.3f}")
            print(f"  Recall:    {results[prompt_type]['recall']:.3f}")
            print(f"  F1 Score:  {results[prompt_type]['f1_score']:.3f}")
            print(f"  Avg Latency: {results[prompt_type]['avg_latency_ms']:.1f}ms")
            if results[prompt_type]['errors_count'] > 0:
                print(f"  Errors: {results[prompt_type]['errors_count']}")
            print("-"*60)
        
        # Calculate comparison metrics
        comparison = {
            "accuracy_improvement": round(
                results["advanced"]["accuracy"] - results["baseline"]["accuracy"], 3
            ),
            "f1_improvement": round(
                results["advanced"]["f1_score"] - results["baseline"]["f1_score"], 3
            ),
            "latency_difference_ms": round(
                results["advanced"]["avg_latency_ms"] - results["baseline"]["avg_latency_ms"], 2
            )
        }
        
        # Determine winner
        better_prompt = "advanced" if results["advanced"]["f1_score"] > results["baseline"]["f1_score"] else "baseline"
        if results["advanced"]["f1_score"] == results["baseline"]["f1_score"]:
            better_prompt = "tie"
        
        # Print summary
        print("\nSUMMARY:")
        print(f"  Better Prompt: {better_prompt}")
        print(f"  F1 Improvement: {comparison['f1_improvement']:+.3f}")
        print(f"  Accuracy Improvement: {comparison['accuracy_improvement']:+.3f}")
        print(f"  Latency Impact: {comparison['latency_difference_ms']:+.1f}ms")
        print("="*60)
        
        return {
            "status": "completed",
            "dataset_info": {
                "total_examples": len(self.test_dataset),
                "class_breakdown": {
                    "safe": len([x for x in self.test_dataset if x["label"] == "safe"]),
                    "toxic": len([x for x in self.test_dataset if x["label"] == "toxic"]), 
                    "spam": len([x for x in self.test_dataset if x["label"] == "spam"])
                }
            },
            "baseline_results": results["baseline"],
            "advanced_results": results["advanced"],
            "comparison": comparison,
            "summary": {
                "better_prompt": better_prompt,
                "performance_gain": f"{comparison['f1_improvement']:+.3f} F1 score",
                "speed_impact": f"{comparison['latency_difference_ms']:+.1f}ms latency"
            }
        }
    

if __name__ =="__main__":
    model_evaluator = ModelEvaluator()
    result=asyncio.run(model_evaluator.run_full_evaluation())
