# app.py

import streamlit as st
from reddit_client import search_reddit
from embedding import get_embeddings
from llm_analysis import get_llm_pipelines, analyze_post
from vector_search import build_faiss_index, search_similar

# --- Page Config ---
st.set_page_config(page_title="BrandSight AI - Mining Reddit for brand mentions, sentiment, and feedback - powered by LLMs.", layout="wide", page_icon="🔍")

# --- Load Models ---
# This happens once and is cached, showing a spinner on first load.
with st.spinner("Warming up the AI models... Please wait."):
    classifier, sentiment_pipe, summarizer = get_llm_pipelines()

# --- Main App UI ---
st.title("🔍 Proactive Brand Intelligence Monitor")
brand = st.text_input("Enter a brand or keyword to search on Reddit (e.g., 'Nvidia' or 'PlayStation 5')", key="brand_input")

if brand:
    # Use a session state to avoid re-fetching and re-analyzing data on every interaction
    if 'results' not in st.session_state or st.session_state.get('brand') != brand:
        st.session_state.brand = brand
        with st.spinner(f"Searching Reddit for posts mentioning **{brand}**..."):
            posts = search_reddit(brand)
        
        if not posts:
            st.warning("No recent posts found for this brand. Try another keyword.")
            st.stop() # Stop the script if no posts are found
        
        texts = [p.title + " " + (p.selftext or "") for p in posts]
        
        # Analyze posts with the local LLM pipelines
        results = []
        with st.spinner("Analyzing posts with local AI... This is much faster!"):
            for text in texts:
                res = analyze_post(text, classifier, sentiment_pipe, summarizer)
                results.append(res)
        
        # Store everything in the session state
        st.session_state.posts = posts
        st.session_state.results = results
        st.session_state.embeddings = get_embeddings(texts)
        st.session_state.index = build_faiss_index(st.session_state.embeddings)

    # --- Display Filters and Results ---
    st.header(f"Analysis for: {st.session_state.brand}")

    categories = ["Bug Report", "Feature Request", "Competitor Mention", "Positive Feedback"]
    category_filter = st.multiselect("Filter by category:", options=categories, default=categories)

    for i, post in enumerate(st.session_state.posts):
        result = st.session_state.results[i]
        if result["category"] in category_filter:
            with st.expander(f"**{post.title}** (r/{post.subreddit.display_name})"):
                st.markdown(f"**Category:** `{result['category']}` | **Sentiment:** `{result['sentiment']}`")
                st.markdown(f"**AI Summary:** *{result['summary']}*")
                st.write(f"🔗 [Link to Reddit post](https://reddit.com{post.permalink})")

    # --- Semantic Search UI ---
    st.header("Search for Similar Posts")
    query = st.text_input("Enter text to find posts with similar meaning (e.g., 'overheating issues')")
    
    if query:
        q_emb = get_embeddings([query])
        distances, indices = search_similar(st.session_state.index, q_emb)
        
        st.subheader("Top 5 similar posts found:")
        for rank, idx in enumerate(indices[0]):
            similar_post = st.session_state.posts[idx]
            similar_result = st.session_state.results[idx]
            st.markdown(f"**{rank+1}. {similar_post.title}** (r/{similar_post.subreddit.display_name})")
            st.markdown(f"> *{similar_result['summary']}*")
            st.write("---")
else:
    st.info("Enter a brand or keyword above to start monitoring.")
