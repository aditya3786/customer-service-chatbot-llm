import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_helper import get_qa_chain, create_vector_db
from sentiment_analyzer import analyze_sentiment
from medical_ner import extract_entities, highlight_entities, CATEGORY_COLORS
from medical_helper import get_medical_qa_chain, create_medical_vector_db

st.set_page_config(page_title="Customer Service Chatbot", layout="wide")
st.title("CUSTOMER SERVICE CHATBOT 🤖")

# --- Session state initialisation ---
if "feedback" not in st.session_state:
    st.session_state.feedback = {"positive": 0, "negative": 0}
if "history" not in st.session_state:
    st.session_state.history = []

# --- Tabs ---
tab_chat, tab_medical, tab_analytics = st.tabs(["💬 Chat", "🏥 Medical Q&A", "📊 Analytics"])

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
