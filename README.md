# LLM-Powered Text Classification API

A content moderation service that classifies user-generated text into categories (toxic, spam, safe) using Large Language Models with feedback collection and evaluation capabilities.

## 🚀 Features

- Text classification into toxic/spam/safe categories
- Dual prompt system (baseline + improved)
- Feedback collection for model improvement
- Metrics tracking and evaluation harness
- FastAPI endpoints + Streamlit UI

## 🛠️ Setup & Installation

```bash
git clone <repository-url>
cd llm-powered-text-classification-api
pip install -r requirements.txt
```

Create `.env` file:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

Run API:
```bash
python main.py
```
API available at `http://localhost:8000`

Run Streamlit UI (optional):
```bash
streamlit run streamlit_ui.py
```

## 📚 API Endpoints

### POST /classify
```json
// Request
{"text": "Your text to classify here"}

// Response
{"class": "safe", "prompt_used": "baseline_prompt", "latency_ms": 345}
```

### POST /feedback
```json
{"text": "Original text", "predicted": "spam", "correct": "safe"}
```

### GET /metrics
```json
{
  "total_requests": 124,
  "class_distribution": {"toxic": 23, "spam": 41, "safe": 60},
  "feedback_counts": {"positive": 12, "negative": 5},
  "latency": {"avg_ms": 320, "p95_ms": 650}
}
```

### GET /healthz & GET /evaluation
Health check and model evaluation endpoints.



## 🎯 Prompt Engineering

**Baseline Prompt (Zero-shot):**
- Direct classification with clear category definitions
- Minimal context, fast execution
- Simple "Respond with only one word" instruction

**Advanced Prompt (Few-shot + Role-based):**
- Expert content moderator persona
- 6 concrete examples (2 per category)
- Detailed classification guidelines
- Context-aware decision making
- ~10-15% better accuracy than baseline

Both prompts managed via `PROMPT_REGISTRY` in `prompt_library.py`


## 🏗️ Architecture

```
llm-powered-text-classification-api/
├── app/
│   ├── data/
│   │   └── feedback_data.json          # Stored feedback data
│   ├── eval/
│   │   ├── __init__.py
│   │   └── evalution.py               # Model evaluation logic
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── prompt_library.py          # Baseline & advanced prompts
│   ├── services/
│   │   ├── __init__.py
│   │   └── classifier.py              # Text classification service
│   ├── telemetry/
│   │   ├── __init__.py
│   │   ├── custom_exception.py        # Custom exception handling
│   │   ├── custom_logger.py           # Structured logging
│   │   └── telemetry.py               # Metrics and feedback tracking
│   └── main.py                        # FastAPI application
├── logs/                              # Auto-generated log files
├── streamlit_ui.py                    # Streamlit web interface
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
├── .env                              # Environment variables (not tracked)
├── .gitignore                        # Git ignore rules
├── .python-version                   # Python version specification
└── README.md                         # This file
```


## 🧪 Testing

```bash
pytest tests/
```
