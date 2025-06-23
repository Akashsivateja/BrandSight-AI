# llm_analysis.py
from transformers import pipeline
import streamlit as st

# --- LAZY LOADING FUNCTIONS ---
# Each model now has its own cached loading function.
# This means a model is only downloaded from Hugging Face the first time it is needed.

@st.cache_resource
def get_classifier():
    """Loads and caches the Zero-Shot Classification model."""
    st.write("Cache miss: Loading classification model...") # For debugging
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", truncation=True)

@st.cache_resource
def get_sentiment_analyzer():
    """Loads and caches the Sentiment Analysis model."""
    st.write("Cache miss: Loading sentiment model...") # For debugging
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", truncation=True)

@st.cache_resource
def get_summarizer():
    """Loads and caches the Summarization model."""
    st.write("Cache miss: Loading summarization model...") # For debugging
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", truncation=True)

# The analysis function now takes no models as arguments.
# It will call the loading functions itself.
def analyze_post(text):
    """
    Analyzes a post by calling the lazy-loading functions for each model.
    """
    result = {"category": "N/A", "sentiment": "N/A", "summary": "N/A"}

    if not text or not text.strip():
        return result
    
    # --- 1. Category Classification ---
    classifier = get_classifier() # Load the model (or get from cache)
    candidate_labels = ["Bug Report", "Feature Request", "Competitor Mention", "Positive Feedback"]
    hypothesis_template = "This text is about a {}."
    classification_result = classifier(text, candidate_labels, hypothesis_template=hypothesis_template)
    result["category"] = classification_result['labels'][0]

    # --- 2. Sentiment Analysis ---
    sentiment_pipe = get_sentiment_analyzer() # Load the model (or get from cache)
    sentiment_result = sentiment_pipe(text)[0]
    result["sentiment"] = sentiment_result['label']

    # --- 3. Summarization ---
    if len(text.split()) > 40:
        summarizer = get_summarizer() # Load the model (or get from cache)
        summary_result = summarizer(text, max_length=60, min_length=20, do_sample=False)[0]
        result["summary"] = summary_result['summary_text']
    else:
        result["summary"] = text

    return result
