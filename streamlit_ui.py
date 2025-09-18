import streamlit as st
import asyncio
from app.services.classifier import TextClassifier
from app.telemetry.telemetry import TelemetryService
from app.eval.evalution import ModelEvaluator

classifier = TextClassifier()
if "telemetry" not in st.session_state:
    st.session_state.telemetry = TelemetryService()

telemetry = st.session_state.telemetry


st.title("Text-classifier")

# --- Classification Section ---
text = st.text_input("Enter the text:", key="classification_text")
execute = st.button("Execute Classification")

if execute and text:  # only run when button clicked and text is provided
    classification, prompt_used, latency_ms = asyncio.run(classifier.classify(text))

    telemetry.record_classification(classification, latency_ms)

    st.write(f"Class: {classification}")
    st.write(f"Prompt used: {prompt_used}")
    st.write(f"Time taken in the task: {latency_ms} ms")

# --- Feedback Section ---
st.title("Feedback")

feed_text = st.text_input("Enter the text:", key="feedback_text")
feed_predicted = st.text_input("Enter the predicted output:", key="feedback_predicted")
feed_actual = st.text_input("Enter the actual output:", key="feedback_actual")
feed_execute = st.button("Submit Feedback")

if feed_execute:
    telemetry.record_feedback(feed_text, feed_predicted, feed_actual)
    st.write("✅ Feedback received successfully")

# --- Metrics Section ---
st.title("Metrics of the Running")

metrics_execution = st.button("Click to Get Metrics")

if metrics_execution:
    st.write(telemetry.get_metrics())

# --- Evaluation Section ---
st.title("Evaluate the Project")

eval_execution = st.button("Click to Evaluate the Model")

if eval_execution:
    model_evaluator = ModelEvaluator()
    result = asyncio.run(model_evaluator.run_full_evaluation())
    st.write(result)
