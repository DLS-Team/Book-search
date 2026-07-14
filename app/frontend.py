"""
Streamlit frontend for Semantic Book Scene Search.

Run from the repository root:
    python -m streamlit run app/frontend.py
"""

from __future__ import annotations

from typing import Any
import html

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000/search"
REQUEST_TIMEOUT_SECONDS = 120


def _clean_text(value: Any) -> str:
    """Return readable text without accidental 'None' values."""
    if value is None:
        return ""
    return str(value).strip()


def _extract_fragment(card: dict[str, Any]) -> str:
    """
    Extract matched text from any supported backend response shape.

    The current backend exposes `fragment`, while older search-engine responses
    may use `best_fragment` or `text`.
    """
    for key in ("fragment", "best_fragment", "text", "chunk_text", "matched_text"):
        fragment = _clean_text(card.get(key))
        if fragment and fragment.lower() not in {
            "full chapter returned",
            "none",
            "n/a",
        }:
            return fragment
    return ""


def _format_score(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if abs(score) >= 100:
        return f"{score:,.2f}"
    if abs(score) >= 1:
        return f"{score:.4f}"
    return f"{score:.6f}"


def _render_result_card(card: dict[str, Any], position: int) -> None:
    rank = card.get("rank", position)
    title = _clean_text(card.get("book_title")) or "Unknown Book"
    author = _clean_text(card.get("author")) or "Unknown Author"
    chapter = _clean_text(card.get("chapter")) or "Unknown Chapter"
    method = _clean_text(card.get("method")) or "unknown"
    provenance = _clean_text(card.get("provenance"))
    warning = _clean_text(card.get("warning"))
    fragment = _extract_fragment(card)

    with st.container(border=True):
        header_col, rank_col = st.columns([6, 1])

        with header_col:
            st.markdown(f"### {title}")
            st.caption(f"**Author:** {author} · **Chapter:** {chapter}")

        with rank_col:
            st.metric("Rank", rank)

        st.markdown("#### Matched chunk")

        if fragment:
            # Render matched text in a high-contrast reading panel.
            # html.escape prevents book text from being interpreted as HTML.
            safe_fragment = html.escape(fragment)
            st.markdown(
                f'<div class="matched-chunk">{safe_fragment}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "No matched chunk was returned for this result. "
                "The frontend can only display text provided by the backend. "
                "Make sure the search engine resolves chapter IDs to chunk text "
                "for this search mode."
            )

        score_col, method_col = st.columns(2)
        score_col.metric("Score", _format_score(card.get("score")))
        method_col.markdown(f"**Method:** `{method}`")

        if provenance:
            st.caption(f"**Provenance:** {provenance}")

        if card.get("low_confidence"):
            st.warning(warning or "This result has low confidence.")
        elif warning:
            st.info(warning)


st.set_page_config(
    page_title="Semantic Book Scene Search",
    page_icon="📖",
    layout="wide",
)

st.markdown(
    """
    <style>
    .matched-chunk {
        background: #f8fafc;
        color: #111827;
        border: 1px solid #cbd5e1;
        border-left: 5px solid #2563eb;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin: 0.35rem 0 1rem 0;
        line-height: 1.65;
        font-size: 1rem;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }

    @media (prefers-color-scheme: dark) {
        .matched-chunk {
            background: #111827;
            color: #f9fafb;
            border-color: #374151;
            border-left-color: #60a5fa;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📖 Semantic Book Scene Search")
st.caption(
    "Search public-domain fiction by scene, mood, or situation rather than exact wording."
)

with st.form("search_form"):
    query = st.text_input(
        "What scene are you looking for?",
        placeholder='e.g. "a lonely person walking through a dark city"',
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        mode = st.selectbox(
            "Search mode",
            options=["dense", "dense_ann", "bm25", "hybrid", "refined"],
            index=0,
            help=(
                "dense = exact dense retrieval; dense_ann = HNSW; "
                "bm25 = lexical retrieval; hybrid = fused retrieval; "
                "refined = hybrid retrieval with fragment refinement."
            ),
        )

    with col2:
        top_k = st.slider("Top-k", min_value=1, max_value=20, value=5)

    submitted = st.form_submit_button(
        "Search",
        type="primary",
        use_container_width=True,
    )

if submitted:
    normalized_query = query.strip()

    if not normalized_query:
        st.warning("Please enter a query.")
        st.stop()

    with st.spinner(f"Searching with {mode}..."):
        try:
            response = requests.get(
                BACKEND_URL,
                params={
                    "q": normalized_query,
                    "mode": mode,
                    "top_k": top_k,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the backend at "
                f"`{BACKEND_URL}`. Start it with:\n\n"
                "`python -m uvicorn app.backend:app --reload --port 8000`"
            )
            st.stop()

        except requests.exceptions.Timeout:
            st.error(
                f"The search request exceeded {REQUEST_TIMEOUT_SECONDS} seconds."
            )
            st.stop()

        except requests.exceptions.HTTPError as exc:
            try:
                details = response.json()
            except ValueError:
                details = response.text
            st.error(f"Backend returned an HTTP error: {exc}\n\n{details}")
            st.stop()

        except requests.exceptions.RequestException as exc:
            st.error(f"Search request failed: {exc}")
            st.stop()

        except ValueError:
            st.error("The backend returned invalid JSON.")
            st.stop()

    if data.get("error"):
        st.error(f"Search mode `{mode}` failed: {data['error']}")
        st.stop()

    results = data.get("results") or []

    if not results:
        st.info("No results were found for this query.")
        st.stop()

    if data.get("low_confidence"):
        st.warning(
            "Low confidence: the system did not find strong evidence for this query. "
            "The closest available matches are shown below."
        )

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Results", len(results))
    summary_col2.metric("Mode", data.get("mode", mode))
    summary_col3.metric("Latency", f"{data.get('latency_ms', 'N/A')} ms")

    st.divider()

    for position, card in enumerate(results, start=1):
        _render_result_card(card, position)
