"""
Task 2.8 — Browser search page (architecture section 3.5).

This is a thin display layer only — it renders whatever the /search backend
(task 2.7) returns and implements no ranking logic of its own, per the plan's
explicit constraint ("This interface does not change the retrieval
architecture. It is a serving layer around the same search methods used in
evaluation.").

Required elements (section 3.5), all present below:
  - a single search bar for free-form scene/mood/situation queries
  - an optional top-k control
  - a loading/error state
  - ranked result cards: book title, author, chapter, fragment, search
    method, score/rank, provenance
  - a low-confidence warning when evidence is weak

Run with:
    uvicorn backend:app --reload --port 8000        # in one terminal
    streamlit run frontend.py                        # in another
"""

from __future__ import annotations

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000/search"

st.set_page_config(page_title="Semantic Book Scene Search", page_icon="📖", layout="centered")

st.title("📖 Semantic Book Scene Search")
st.caption("Search public-domain fiction by scene, mood, or situation — not exact words.")

with st.form("search_form"):
    query = st.text_input(
        "What scene are you looking for?",
        placeholder='e.g. "a lonely person walking through a dark city"',
    )
    col1, col2 = st.columns([2, 1])
    with col1:
        mode = st.selectbox(
            "Search mode",
            options=["dense", "bm25", "hybrid", "refined"],
            index=0,
            help="dense = semantic (Role 2). bm25/hybrid/refined depend on other roles' modules.",
        )
    with col2:
        top_k = st.slider("Top-k", min_value=1, max_value=20, value=5)
    submitted = st.form_submit_button("Search")

if submitted:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            try:
                resp = requests.get(BACKEND_URL, params={"q": query, "mode": mode, "top_k": top_k}, timeout=10)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the search backend at "
                    f"`{BACKEND_URL}`. Is it running? (`uvicorn backend:app --reload --port 8000`)"
                )
                data = None
            except requests.exceptions.RequestException as e:
                st.error(f"Search failed: {e}")
                data = None

        if data is not None:
            if data.get("error"):
                st.error(f"Search mode '{mode}' is not available yet: {data['error']}")
            elif not data["results"]:
                st.info("No results found for this query.")
            else:
                if data["low_confidence"]:
                    st.warning(
                        "⚠️ Low confidence: the system could not find strong evidence for this query. "
                        "Showing the closest matches anyway."
                    )

                st.caption(f"Mode: **{data['mode']}** · Latency: {data['latency_ms']} ms")

                for card in data["results"]:
                    with st.container(border=True):
                        st.markdown(f"**{card['book_title']}** — {card['chapter']}")
                        st.caption(f"by {card['author']}")
                        st.write(card["fragment"])
                        meta_col1, meta_col2, meta_col3 = st.columns(3)
                        meta_col1.metric("Rank", card["rank"])
                        meta_col2.metric("Score", card["score"])
                        meta_col3.write(f"Method: `{card['method']}`")
                        st.caption(f"Provenance: {card['provenance']}")
