import streamlit as st
from reddit_client import search_reddit
# We only import the main analysis function now
from llm_analysis import analyze_post
from embedding import get_embeddings
from vector_search import build_faiss_index, search_similar

# --- Page Config ---
st.set_page_config(page_title="BrandSight AI", layout="wide", page_icon="🔍")

# --- REMOVED THE INITIAL MODEL LOADING ---
# The models will now be loaded on-demand inside the analyze_post function.

# --- Main App UI ---
st.title("🔍 BrandSight AI: Reddit Intelligence Monitor")
brand = st.text_input("Enter a brand or keyword to search on Reddit (e.g., 'Nvidia' or 'PlayStation 5')", key="brand_input")

if brand:
    # Use session state to avoid re-fetching and re-analyzing data
    if 'results' not in st.session_state or st.session_state.get('brand') != brand:
        st.session_state.brand = brand
        with st.spinner(f"Searching Reddit for posts mentioning '{brand}'..."):
            posts = search_reddit(brand)
        
        if not posts:
            st.warning("No recent posts found for this brand. Try another keyword.")
            st.stop()
        
        # This is where the magic happens. The spinner will stay active while
        # the models are downloaded one by one for the first time.
        with st.spinner("Analyzing posts with AI... (First-time analysis may take several minutes to download models)"):
            results = []
            for post in posts:
                text_to_analyze = post.title + " " + (post.selftext or "")
                # The analyze_post function now handles loading its own models
                res = analyze_post(text_to_analyze)
                results.append(res)
        
        st.session_state.posts = posts
        st.session_state.results = results
        # For now, let's keep the semantic search part simple
        texts_for_embedding = [p.title for p in posts]
        st.session_state.embeddings = get_embeddings(texts_for_embedding)
        st.session_state.index = build_faiss_index(st.session_state.embeddings)

    # --- Display Filters and Results (This part remains the same) ---
    st.header(f"Analysis for: {st.session_state.brand}")
    # ... (the rest of your display code is perfect and needs no changes)

    categories = ["Bug Report", "Feature Request", "Competitor Mention", "Positive Feedback"]
    category_filter = st.multiselect("Filter by category:", options=categories, default=categories)

    for i, post in enumerate(st.session_state.posts):
        result = st.session_state.results[i]
        if result["category"] in category_filter:
            with st.expander(f"**{post.title}** (r/{post.subreddit.display_name})"):
                st.markdown(f"**Category:** `{result['category']}` | **Sentiment:** `{result['sentiment']}`")
                st.markdown(f"**AI Summary:** *{result['summary']}*")
                st.write(f"🔗 [Link to Reddit post](https://reddit.com{post.permalink})")

    # --- Semantic Search UI (This part remains the same) ---
    st.header("Search for Similar Posts")
    # ... (the rest of your semantic search code is perfect and needs no changes)
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
