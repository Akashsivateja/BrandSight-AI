# llm_analysis.py

from transformers import pipeline
import streamlit as st

# Use Streamlit's cache to load these models only once.
@st.cache_resource
def get_llm_pipelines():
    """Loads and caches the NLP models from Hugging Face."""
    # We pass truncation=True to the pipeline constructor.
    # This tells the pipeline to automatically cut down any text that is too long.
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", truncation=True)
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", truncation=True)
    return classifier, sentiment_pipe, summarizer

def analyze_post(text, classifier, sentiment_pipe, summarizer):
    """
    Analyzes a post using local Hugging Face models.
    The pipelines will now handle truncation automatically.
    """
    result = {"category": "N/A", "sentiment": "N/A", "summary": "N/A"}

    if not text or not text.strip():
        return result

    # --- NO MORE MANUAL TRUNCATION NEEDED ---
    # We can now safely pass the original text to the models.
    
    # --- 1. Category Classification ---
    candidate_labels = ["Bug Report", "Feature Request", "Competitor Mention", "Positive Feedback"]
    hypothesis_template = "This text is about a {}."
    classification_result = classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
    result["category"] = classification_result['labels'][0]

    # --- 2. Sentiment Analysis ---
    sentiment_result = sentiment_pipe(text)[0]
    result["sentiment"] = sentiment_result['label']

    # --- 3. Summarization ---
    if len(text.split()) > 40:
        summary_result = summarizer(text, max_length=60, min_length=20, do_sample=False)[0]
        result["summary"] = summary_result['summary_text']
    else:
        result["summary"] = text

    return result