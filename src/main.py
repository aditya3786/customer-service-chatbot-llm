import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_helper import get_qa_chain, create_vector_db
from sentiment_analyzer import analyze_sentiment
from medical_ner import extract_entities, highlight_entities, CATEGORY_COLORS
from medical_helper import get_medical_qa_chain, create_medical_vector_db
import knowledge_expander
import kb_scheduler
from arxiv_helper import create_arxiv_vector_db, search_papers, get_arxiv_qa_chain, ARXIV_VECTORDB_PATH
from cs_ner import extract_cs_entities, highlight_cs_entities, CATEGORY_COLORS as CS_COLORS
from summarizer import get_summary
from multimodal_assistant import run_pipeline

st.set_page_config(page_title="Customer Service Chatbot", layout="wide")
st.title("CUSTOMER SERVICE CHATBOT 🤖")

# --- Session state initialisation ---
if "feedback" not in st.session_state:
    st.session_state.feedback = {"positive": 0, "negative": 0}
if "history" not in st.session_state:
    st.session_state.history = []
if "arxiv_chat_history" not in st.session_state:
    st.session_state.arxiv_chat_history = []
if "arxiv_qa_history" not in st.session_state:
    st.session_state.arxiv_qa_history = []
if "multimodal_history" not in st.session_state:
    st.session_state.multimodal_history = []

# --- Tabs ---
tab_chat, tab_medical, tab_update, tab_research, tab_multimodal, tab_analytics = st.tabs(
    ["💬 Chat", "🏥 Medical Q&A", "🔄 Auto-Update", "🔬 Research", "🖼️ Visual AI", "📊 Analytics"]
)

# ── CHAT TAB ──────────────────────────────────────────────────────────────────
with tab_chat:
    btn = st.button("Create Knowledgebase")
    if btn:
        create_vector_db()

    question = st.text_input("Question: ", key="cs_question")

    if question:
        sentiment_result = analyze_sentiment(question)
        label = sentiment_result["label"]
        compound = sentiment_result["compound"]

        _BADGE_CONFIG = {
            "positive": ("green", "😊 Positive"),
            "negative": ("red",   "😟 Negative"),
            "neutral":  ("gray",  "😐 Neutral"),
        }
        badge_color, badge_text = _BADGE_CONFIG[label]

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.caption("Sentiment detected:")
        with col2:
            st.badge(badge_text, color=badge_color)
        with col3:
            st.caption(f"confidence: `{compound:.2f}`")

        _EMPATHY_MESSAGES = {
            "negative": "😟 We noticed you're frustrated — we're here to help and will do our best to resolve this.",
            "positive": "😊 Great to hear you're feeling good! Here's what we found for you.",
        }
        if label in _EMPATHY_MESSAGES:
            st.info(_EMPATHY_MESSAGES[label])

        chain = get_qa_chain(sentiment=label)
        response = chain(question)
        answer = response["result"]

        st.header("Answer")
        st.write(answer)

        st.session_state.history.append({
            "question": question,
            "sentiment": label,
            "compound": compound,
            "answer": answer,
        })

        st.divider()
        st.caption("Was this response helpful?")
        col_up, col_down, _ = st.columns([1, 1, 6])

        with col_up:
            if st.button("👍", key=f"up_{len(st.session_state.history)}"):
                st.session_state.feedback["positive"] += 1
                st.success("Thanks for the feedback!")
        with col_down:
            if st.button("👎", key=f"down_{len(st.session_state.history)}"):
                st.session_state.feedback["negative"] += 1
                st.error("Sorry about that. We'll improve!")

        total = st.session_state.feedback["positive"] + st.session_state.feedback["negative"]
        if total > 0:
            satisfaction = round(st.session_state.feedback["positive"] / total * 100)
            st.caption(
                f"Session satisfaction: **{satisfaction}%** "
                f"({st.session_state.feedback['positive']}👍 / "
                f"{st.session_state.feedback['negative']}👎)"
            )

# ── MEDICAL Q&A TAB ───────────────────────────────────────────────────────────
with tab_medical:
    st.subheader("Medical Q&A — Powered by MedQuAD (NIH)")
    st.caption("5,068 Q&A pairs from CancerGov, GHR, NIDDK, NINDS, NHLBI")

    if st.button("Build Medical Knowledge Base"):
        with st.spinner("Embedding 5,068 medical Q&A pairs... (this takes ~60s the first time)"):
            create_medical_vector_db()
        st.success("Medical knowledge base ready!")

    med_question = st.text_input("Ask a medical question:", key="med_question",
                                  placeholder="e.g. What are the symptoms of leukemia?")

    if med_question:
        # --- NER: extract and highlight entities ---
        entities = extract_entities(med_question)

        if entities:
            st.markdown("**Recognized medical entities:**")
            cols = st.columns(len(entities) if len(entities) <= 4 else 4)
            for i, ent in enumerate(entities[:4]):
                color = CATEGORY_COLORS.get(ent["category"], "#e0e0e0")
                cols[i % 4].markdown(
                    f'<span style="background:{color};padding:3px 8px;border-radius:4px;'
                    f'font-size:0.85em">{ent["term"]} — <em>{ent["category"]}</em></span>',
                    unsafe_allow_html=True,
                )
            st.markdown("")

            highlighted = highlight_entities(med_question, entities)
            st.markdown(
                f"<p style='font-size:1.05em'>{highlighted}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("No specific medical entities detected in the question.")

        # --- NER legend ---
        st.markdown(
            '<span style="font-size:0.8em">'
            '<mark style="background:#ffcccc;padding:1px 6px;border-radius:3px">Disease/Condition</mark> &nbsp;'
            '<mark style="background:#ffe0b2;padding:1px 6px;border-radius:3px">Symptom</mark> &nbsp;'
            '<mark style="background:#c8e6c9;padding:1px 6px;border-radius:3px">Treatment</mark>'
            "</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        # --- RAG answer ---
        try:
            med_chain = get_medical_qa_chain()
            med_response = med_chain(med_question)
            st.header("Answer")
            st.write(med_response["result"])

            # Source attribution
            if med_response.get("source_documents"):
                with st.expander("📄 Source documents"):
                    for doc in med_response["source_documents"][:3]:
                        st.markdown(f"- {doc.page_content[:200]}…")
        except Exception as e:
            if "medical_faiss_index" in str(e) or "No such file" in str(e):
                st.error("Please click **'Build Medical Knowledge Base'** first to create the index.")
            else:
                st.error(f"Error: {e}")

        # --- Safety disclaimer ---
        st.warning(
            "⚠️ **Medical Disclaimer:** This information is for educational purposes only and is sourced "
            "from NIH public databases. Always consult a qualified healthcare provider for personal "
            "medical advice, diagnosis, or treatment."
        )

# ── AUTO-UPDATE TAB ───────────────────────────────────────────────────────────
with tab_update:
    st.subheader("Dynamic Knowledge Base Expansion")
    st.caption("Add new sources to either knowledge base — manually or on a recurring schedule.")

    kb_choice_options = list(knowledge_expander.KB_PATHS.keys())

    col_size1, col_size2 = st.columns(2)
    for col, kb in zip([col_size1, col_size2], kb_choice_options):
        with col:
            count = knowledge_expander.get_index_doc_count(kb)
            st.metric(f"{kb} — vectors", count)

    st.divider()

    # --- Add a URL source ---
    st.markdown("**Add a URL Source**")
    with st.form("add_url_form"):
        url_kb = st.selectbox("Target knowledge base", kb_choice_options, key="url_kb")
        url_input = st.text_input("URL to ingest", placeholder="https://example.com/article")
        col_a, col_b = st.columns(2)
        submit_register = col_a.form_submit_button("Register for periodic refresh")
        submit_once = col_b.form_submit_button("Ingest once now")

        if submit_register and url_input:
            kb_scheduler.add_source(url_kb, url_input)
            st.success(f"Registered {url_input} for periodic refresh on {url_kb}.")

        if submit_once and url_input:
            with st.spinner("Fetching and embedding..."):
                result = knowledge_expander.ingest_url(url_kb, url_input)
            if result["status"] == "added":
                st.success(f"Added {result['chunks_added']} chunks to {url_kb}.")
            else:
                st.info("Skipped — this content was already ingested (no changes detected).")

    # --- Add manual Q&A ---
    st.markdown("**Add a Manual Q&A Entry**")
    with st.form("add_manual_form"):
        manual_kb = st.selectbox("Target knowledge base", kb_choice_options, key="manual_kb")
        manual_prompt = st.text_input("Question / prompt")
        manual_response = st.text_area("Answer / response")
        submit_manual = st.form_submit_button("Add to knowledge base")

        if submit_manual and manual_prompt and manual_response:
            with st.spinner("Embedding..."):
                result = knowledge_expander.ingest_manual_text(manual_kb, manual_prompt, manual_response)
            if result["status"] == "added":
                st.success(f"Added to {manual_kb}.")
            else:
                st.info("Skipped — identical entry already exists.")

    st.divider()

    # --- Periodic refresh controls ---
    st.markdown("**Periodic Auto-Refresh**")
    registered = kb_scheduler.get_sources()

    if registered:
        st.dataframe(pd.DataFrame(registered), use_container_width=True, hide_index=True)
    else:
        st.caption("No sources registered yet. Add one above with 'Register for periodic refresh'.")

    col_c, col_d, col_e = st.columns([1, 1, 2])
    with col_c:
        interval = st.number_input("Interval (minutes)", min_value=1, max_value=1440, value=60)
    with col_d:
        if not kb_scheduler.is_running():
            if st.button("▶ Start auto-refresh"):
                kb_scheduler.start_scheduler(interval_minutes=interval)
                st.rerun()
        else:
            if st.button("⏹ Stop auto-refresh"):
                kb_scheduler.stop_scheduler()
                st.rerun()
    with col_e:
        status = "🟢 Running" if kb_scheduler.is_running() else "⚪ Stopped"
        st.caption(f"Scheduler status: {status}")

    if st.button("Run refresh now (all registered sources)"):
        with st.spinner("Refreshing all registered sources..."):
            results = kb_scheduler.run_now()
        for r in results:
            if r["status"] == "added":
                st.success(f"{r['url']} → +{r['chunks_added']} chunks ({r['kb']})")
            elif r["status"] == "skipped":
                st.info(f"{r['url']} → no changes ({r['kb']})")
            else:
                st.error(f"{r['url']} → error: {r.get('reason')}")

    st.divider()

    # --- Ingestion history ---
    st.markdown("**Ingestion History**")
    history = knowledge_expander.get_ingestion_history()
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.caption("No ingestion history yet.")

# ── RESEARCH TAB ──────────────────────────────────────────────────────────────
import os as _os
with tab_research:
    st.subheader("🔬 arXiv Research Assistant")
    st.caption("Domain expert chatbot on ~3,000 CS papers (AI, ML, NLP, CV, NE) from arXiv.org")

    # ── Knowledge base setup ────────────────────────────────────────────────────
    _arxiv_ready = _os.path.exists(ARXIV_VECTORDB_PATH)
    col_rb1, col_rb2 = st.columns([3, 1])
    with col_rb1:
        if st.button("Build Research Knowledge Base"):
            _csv_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "dataset", "arxiv_cs_sample.csv")
            if not _os.path.exists(_csv_path):
                st.error("Dataset not found. Run `python dataset/fetch_arxiv.py` first to download papers.")
            else:
                with st.spinner("Embedding arXiv papers... (this may take ~90s the first time)"):
                    _n = create_arxiv_vector_db()
                st.success(f"Research knowledge base ready! ({_n} chunks indexed)")
                st.rerun()
    with col_rb2:
        if _arxiv_ready:
            st.success("KB: Ready ✅")
        else:
            st.warning("KB: Not built")

    if not _arxiv_ready:
        st.info("Click **'Build Research Knowledge Base'** to get started. The dataset CSV must exist at `dataset/arxiv_cs_sample.csv`.")
        st.stop()

    st.divider()

    # ── Paper Search ────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Paper Search")
    st.caption("Semantic search over paper titles and abstracts")

    _search_query = st.text_input(
        "Search for papers:",
        placeholder="e.g. attention mechanism in transformers",
        key="arxiv_search",
    )

    if _search_query:
        with st.spinner("Searching..."):
            _results = search_papers(_search_query, k=5)

        if not _results:
            st.info("No relevant papers found. Try a different query.")
        else:
            st.markdown(f"**Top {len(_results)} results:**")
            for _idx, (_doc, _score) in enumerate(_results):
                _meta = _doc.metadata
                _paper_key = _meta.get("arxiv_id", str(_idx))
                with st.container(border=True):
                    _col_title, _col_score = st.columns([5, 1])
                    with _col_title:
                        st.markdown(f"**{_meta.get('title', 'Untitled')}**")
                    with _col_score:
                        st.caption(f"Score: {_score:.2f}")

                    st.caption(
                        f"👤 {_meta.get('authors', 'N/A')} &nbsp;|&nbsp; "
                        f"📅 {_meta.get('year', 'N/A')} &nbsp;|&nbsp; "
                        f"🏷️ {_meta.get('categories', '')}"
                    )

                    _abstract_text = _doc.page_content
                    _cs_entities = extract_cs_entities(_abstract_text)
                    if _cs_entities:
                        _highlighted = highlight_cs_entities(_abstract_text[:600], _cs_entities[:8])
                        st.markdown(
                            f"<p style='font-size:0.9em;color:#444'>{_highlighted}…</p>",
                            unsafe_allow_html=True,
                        )
                        # NER legend
                        st.markdown(
                            '<span style="font-size:0.75em">'
                            + " &nbsp; ".join(
                                f'<mark style="background:{CS_COLORS[c]};padding:1px 5px;border-radius:3px">{c}</mark>'
                                for c in CS_COLORS
                            )
                            + "</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.caption(_abstract_text[:400] + "…")

                    _sum_state_key = f"arxiv_summary_{_paper_key}"
                    if st.button("📄 Summarize (open-source DistilBART)", key=f"sum_btn_{_paper_key}_{_idx}"):
                        with st.spinner("Summarizing with DistilBART (open-source LLM)..."):
                            st.session_state[_sum_state_key] = get_summary(_abstract_text)
                    if _sum_state_key in st.session_state:
                        st.info(f"**DistilBART Summary:** {st.session_state[_sum_state_key]}")

                    if _meta.get("url"):
                        st.markdown(f"[📎 View on arXiv]({_meta['url']})")

    st.divider()

    # ── Research Q&A ────────────────────────────────────────────────────────────
    st.markdown("### 💬 Research Q&A")
    st.caption("Ask complex questions — follow-up context is remembered within this session")

    _arxiv_question = st.text_input(
        "Ask a research question:",
        placeholder="e.g. What is BERT and how does it work?",
        key="arxiv_qa",
    )

    _col_qa1, _col_qa2 = st.columns([4, 1])
    with _col_qa2:
        if st.button("🗑 Clear conversation"):
            st.session_state.arxiv_chat_history = []
            st.session_state.arxiv_qa_history = []
            st.rerun()

    if _arxiv_question:
        with st.spinner("Searching papers and generating answer..."):
            try:
                _chain = get_arxiv_qa_chain()
                _response = _chain({
                    "question": _arxiv_question,
                    "chat_history": st.session_state.arxiv_chat_history,
                })
                _answer = _response.get("answer", _response.get("result", "No answer found."))
                _sources = _response.get("source_documents", [])
            except Exception as _e:
                _answer = f"Error: {_e}"
                _sources = []

        st.session_state.arxiv_chat_history.append((_arxiv_question, _answer))
        st.session_state.arxiv_qa_history.append({
            "question": _arxiv_question,
            "answer": _answer,
            "sources": [s.metadata.get("title", "") for s in _sources[:3]],
        })

        st.markdown("**Answer:**")
        st.write(_answer)

        if _sources:
            with st.expander("📄 Source papers"):
                for _s in _sources[:4]:
                    _sm = _s.metadata
                    st.markdown(
                        f"- **{_sm.get('title', 'Untitled')}** "
                        f"({_sm.get('year', '')}) — {_sm.get('authors', '')}"
                    )

    # Show last 3 conversation exchanges
    if st.session_state.arxiv_qa_history:
        st.markdown("**Recent conversation:**")
        for _item in reversed(st.session_state.arxiv_qa_history[-3:]):
            with st.expander(f"Q: {_item['question'][:80]}…"):
                st.write(_item["answer"])
                if _item["sources"]:
                    st.caption("Sources: " + " | ".join(filter(None, _item["sources"])))

    st.divider()

    # ── Concept Visualization ────────────────────────────────────────────────────
    st.markdown("### 📊 Concept Visualization")
    _csv_viz_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "dataset", "arxiv_cs_sample.csv")
    if not _os.path.exists(_csv_viz_path):
        st.caption("Run `python dataset/fetch_arxiv.py` to generate the dataset for visualizations.")
    else:
        @st.cache_data
        def _load_arxiv_df():
            return pd.read_csv(_csv_viz_path)

        _df = _load_arxiv_df()

        _col_v1, _col_v2 = st.columns(2)

        with _col_v1:
            st.markdown("**CS Subcategory Distribution**")
            _all_cats = []
            for _cat_str in _df["categories"].dropna():
                _all_cats.extend([c.strip() for c in _cat_str.split(",") if c.strip().startswith("cs.")])
            _cat_counts = pd.Series(_all_cats).value_counts().head(12).reset_index()
            _cat_counts.columns = ["Category", "Count"]
            _fig_cat = px.bar(_cat_counts, x="Count", y="Category", orientation="h",
                              color="Count", color_continuous_scale="Blues")
            _fig_cat.update_layout(showlegend=False, margin=dict(t=10, b=10), yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(_fig_cat, use_container_width=True)

        with _col_v2:
            st.markdown("**Papers by Year**")
            _year_counts = _df["year"].value_counts().sort_index().reset_index()
            _year_counts.columns = ["Year", "Count"]
            _fig_year = px.bar(_year_counts, x="Year", y="Count",
                               color="Count", color_continuous_scale="Teal")
            _fig_year.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(_fig_year, use_container_width=True)

        st.markdown("**Top 25 Abstract Keywords**")
        _STOPWORDS = {
            "the", "a", "an", "in", "of", "to", "and", "for", "is", "are", "that",
            "this", "with", "we", "our", "on", "by", "from", "as", "it", "be",
            "at", "or", "can", "not", "which", "have", "has", "was", "their",
            "also", "using", "show", "used", "based", "than", "more", "these",
            "such", "been", "its", "into", "two", "new", "paper", "method",
            "approach", "results", "proposed", "models", "model", "training",
            "data", "tasks", "task", "work", "large", "both", "between", "each",
        }
        _words: list[str] = []
        for _abstract in _df["abstract"].dropna():
            _words.extend(
                w.lower().strip(".,;:()[]\"'") for w in _abstract.split()
                if len(w) > 3 and w.lower().strip(".,;:()[]\"'") not in _STOPWORDS
            )
        from collections import Counter
        _top_words = pd.DataFrame(Counter(_words).most_common(25), columns=["Keyword", "Frequency"])
        _fig_kw = px.bar(_top_words, x="Frequency", y="Keyword", orientation="h",
                         color="Frequency", color_continuous_scale="Oranges")
        _fig_kw.update_layout(showlegend=False, margin=dict(t=10, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(_fig_kw, use_container_width=True)

# ── VISUAL AI TAB ─────────────────────────────────────────────────────────────
with tab_multimodal:
    st.subheader("🖼️ Visual AI — Multi-Modal Reasoning Assistant")
    st.caption(
        "Upload an image and ask a question. The assistant runs a 4-stage pipeline: "
        "image analysis → ambiguity detection → evidence-labelled response → self-critique validation."
    )

    # Evidence label legend
    st.markdown(
        '<span style="font-size:0.8em">'
        '<code>[seen in image]</code> directly observed &nbsp;·&nbsp; '
        '<code>[inferred]</code> logically derived &nbsp;·&nbsp; '
        '<code>[general knowledge]</code> background context'
        '</span>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Upload + question inputs ─────────────────────────────────────────────
    _mm_col1, _mm_col2 = st.columns([1, 2])

    with _mm_col1:
        _uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"],
            key="mm_upload",
            help="Supports JPEG, PNG, WebP",
        )
        if _uploaded:
            st.image(_uploaded, use_container_width=True)
            _mm_mime = f"image/{_uploaded.type.split('/')[-1].replace('jpg', 'jpeg')}"

    with _mm_col2:
        _mm_question = st.text_input(
            "Your question about the image:",
            placeholder="e.g. What is the main trend shown in this chart?",
            key="mm_question",
            help="Leave blank to run a pure image description.",
        )
        _mm_analyze = st.button(
            "🔍 Analyze",
            disabled=(_uploaded is None),
            key="mm_analyze_btn",
        )

    # ── Run pipeline ─────────────────────────────────────────────────────────
    if _mm_analyze and _uploaded is not None:
        _image_bytes = _uploaded.read()

        with st.status("Running 4-stage reasoning pipeline…", expanded=True) as _mm_status:
            # Stage 1
            st.write("🔍 Stage 1: Analyzing image structure and content…")
            from multimodal_assistant import analyze_image, detect_ambiguity, generate_response, validate_response
            import time as _time
            _t0 = _time.time()
            _img_analysis = analyze_image(_image_bytes, _mm_mime)
            st.write(f"   ✔ Image analyzed ({_time.time()-_t0:.1f}s) — scene: **{_img_analysis.get('scene_type','?')}**, confidence: **{_img_analysis.get('confidence','?')}**")

            # Stage 2
            st.write("🤔 Stage 2: Checking question for ambiguity…")
            _t0 = _time.time()
            _effective_q = _mm_question.strip() or "Describe this image in detail."
            _ambiguity = detect_ambiguity(_effective_q, _img_analysis)
            _amb_label = "ambiguous" if _ambiguity.get("is_ambiguous") else "clear"
            st.write(f"   ✔ Question is **{_amb_label}** ({_time.time()-_t0:.1f}s)")

            # Stage 3
            st.write("💬 Stage 3: Generating evidence-labelled response…")
            _t0 = _time.time()
            _mm_history = [(h["question"], h["final_answer"]) for h in st.session_state.multimodal_history]
            _answer = generate_response(_effective_q, _img_analysis, _mm_history, _ambiguity.get("assumptions", []))
            st.write(f"   ✔ Response generated ({_time.time()-_t0:.1f}s)")

            # Stage 4
            st.write("✅ Stage 4: Validating response accuracy…")
            _t0 = _time.time()
            _validation = validate_response(_effective_q, _img_analysis, _answer)
            _conf = _validation.get("confidence", 0.7)
            st.write(f"   ✔ Validation complete ({_time.time()-_t0:.1f}s) — confidence: **{_conf:.0%}**")

            _mm_status.update(label="Pipeline complete ✅", state="complete")

        _final_answer = _validation.get("corrected_answer") or _answer

        # ── Image analysis detail ────────────────────────────────────────────
        with st.expander("📊 Stage 1 — Image Analysis Details"):
            _ia_col1, _ia_col2 = st.columns(2)
            with _ia_col1:
                st.markdown(f"**Scene type:** {_img_analysis.get('scene_type','')}")
                st.markdown(f"**Confidence:** {_img_analysis.get('confidence','')}")
                st.markdown(f"**Objects detected:** {', '.join(_img_analysis.get('objects', [])) or 'none'}")
                st.markdown(f"**Dominant colors:** {', '.join(_img_analysis.get('colors', [])) or 'none'}")
            with _ia_col2:
                st.markdown(f"**Description:** {_img_analysis.get('description','')}")
                if _img_analysis.get("text_content"):
                    st.markdown(f"**Visible text (OCR):** {_img_analysis['text_content']}")
                if _img_analysis.get("spatial_info"):
                    st.markdown(f"**Spatial layout:** {_img_analysis['spatial_info']}")

        # ── Ambiguity notice ─────────────────────────────────────────────────
        if _ambiguity.get("is_ambiguous"):
            st.warning(
                f"⚠️ **Ambiguous question detected.** "
                f"{_ambiguity.get('reasoning','')}\n\n"
                + (f"**Clarifying question:** {_ambiguity['clarifying_question']}\n\n" if _ambiguity.get("clarifying_question") else "")
                + (f"**Proceeding with assumptions:** {'; '.join(_ambiguity.get('assumptions', []))}" if _ambiguity.get("assumptions") else "")
            )

        # ── Answer ───────────────────────────────────────────────────────────
        st.markdown("### Answer")
        st.markdown(_final_answer)

        # ── Validation badge ─────────────────────────────────────────────────
        _v_col1, _v_col2, _v_col3 = st.columns([1, 1, 3])
        with _v_col1:
            _conf_color = "green" if _conf >= 0.75 else ("orange" if _conf >= 0.5 else "red")
            st.badge(f"Confidence: {_conf:.0%}", color=_conf_color)
        with _v_col2:
            _valid_label = "✅ Valid" if _validation.get("is_valid") else "⚠️ Revised"
            st.caption(_valid_label)

        if _validation.get("caveats"):
            with st.expander("⚠️ Stage 4 — Validation Caveats"):
                for _cav in _validation["caveats"]:
                    st.markdown(f"- {_cav}")

        # ── Save to history ──────────────────────────────────────────────────
        st.session_state.multimodal_history.append({
            "question": _effective_q,
            "image_description": _img_analysis.get("description", ""),
            "scene_type": _img_analysis.get("scene_type", ""),
            "answer": _answer,
            "final_answer": _final_answer,
            "confidence": _conf,
            "is_valid": _validation.get("is_valid", True),
        })

    # ── Conversation history ─────────────────────────────────────────────────
    if st.session_state.multimodal_history:
        st.divider()
        _mm_hist_col1, _mm_hist_col2 = st.columns([4, 1])
        with _mm_hist_col1:
            st.markdown("**Conversation history** (last 3 turns)")
        with _mm_hist_col2:
            if st.button("🗑 Clear history", key="mm_clear"):
                st.session_state.multimodal_history = []
                st.rerun()

        for _turn in reversed(st.session_state.multimodal_history[-3:]):
            with st.chat_message("user"):
                st.markdown(f"**Q:** {_turn['question']}")
                if _turn.get("scene_type"):
                    st.caption(f"Image: {_turn['scene_type']}")
            with st.chat_message("assistant"):
                st.markdown(_turn["final_answer"][:600] + ("…" if len(_turn["final_answer"]) > 600 else ""))
                st.caption(f"Confidence: {_turn['confidence']:.0%} {'✅' if _turn['is_valid'] else '⚠️ revised'}")

# ── ANALYTICS TAB ─────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("Sentiment Analytics Dashboard")

    history = st.session_state.history

    if not history:
        st.info("No queries yet. Ask some questions in the Chat tab to see analytics.")
    else:
        df = pd.DataFrame(history)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Queries", len(df))
        col_b.metric("Avg Confidence", f"{df['compound'].abs().mean():.2f}")
        total_fb = st.session_state.feedback["positive"] + st.session_state.feedback["negative"]
        if total_fb > 0:
            sat_pct = round(st.session_state.feedback["positive"] / total_fb * 100)
            col_c.metric("Satisfaction", f"{sat_pct}%")
        else:
            col_c.metric("Satisfaction", "N/A")

        st.divider()

        col_left, col_right = st.columns(2)
        color_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}

        with col_left:
            st.markdown("**Sentiment Distribution**")
            counts = df["sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            fig_pie = px.pie(
                counts, names="Sentiment", values="Count",
                color="Sentiment", color_discrete_map=color_map, hole=0.4,
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.markdown("**Confidence Score per Query**")
            df["query_num"] = [f"Q{i+1}" for i in range(len(df))]
            fig_bar = px.bar(
                df, x="query_num", y="compound",
                color="sentiment", color_discrete_map=color_map,
                labels={"query_num": "Query", "compound": "Compound Score"},
                range_y=[-1, 1],
            )
            fig_bar.add_hline(y=0.05, line_dash="dot", line_color="green", annotation_text="positive threshold")
            fig_bar.add_hline(y=-0.05, line_dash="dot", line_color="red", annotation_text="negative threshold")
            fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("**Query History**")
        display_df = df[["query_num", "question", "sentiment", "compound"]].copy()
        display_df.columns = ["#", "Question", "Sentiment", "Compound Score"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
