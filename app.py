"""
app.py
------
DMHardware AI Chatbot — Streamlit chat UI.

A Gemini-powered assistant grounded in a curated Electronics / Robotics /
PCB Design / Digital Logic Design knowledge base via TF-IDF retrieval.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv

from rag.preprocess import load_knowledge_base
from rag.retriever import KnowledgeRetriever
from rag.gemini_client import get_client, generate_response, DEFAULT_MODEL

load_dotenv()

KB_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge_base.csv")

st.set_page_config(page_title="DMHardware AI Chatbot", page_icon="🔧", layout="centered")


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_retriever():
    df = load_knowledge_base(KB_PATH)
    return KnowledgeRetriever(df), len(df)


def main():
    st.title("🔧 DMHardware AI Chatbot")
    st.caption(
        "Gemini-powered assistant grounded in a curated knowledge base covering "
        "Electronics, Robotics, PCB Design, and Digital Logic Design."
    )

    retriever, kb_size = load_retriever()

    with st.sidebar:
        st.header("Settings")
        st.write(f"📚 Knowledge base: **{kb_size}** records")

        api_key_present = bool(os.environ.get("GEMINI_API_KEY"))
        if api_key_present:
            st.success("GEMINI_API_KEY loaded ✅")
        else:
            st.error("GEMINI_API_KEY not found. Add it to a .env file.")

        model = st.text_input("Gemini model", value=DEFAULT_MODEL)
        top_k = st.slider("Records to retrieve per question", 1, 10, 5)
        show_context = st.checkbox("Show retrieved context (debug)", value=False)

        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_message = st.chat_input("Ask about electronics, robotics, PCB design, or digital logic...")

    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):
            if not api_key_present:
                error_text = (
                    "I can't reach Gemini yet — set your `GEMINI_API_KEY` in a "
                    "`.env` file (see `.env.example`) and restart the app."
                )
                st.markdown(error_text)
                st.session_state.messages.append({"role": "assistant", "content": error_text})
            else:
                with st.spinner("Thinking..."):
                    records = retriever.retrieve(user_message, top_k=top_k)
                    context = retriever.format_context(records)

                    if show_context:
                        with st.expander("Retrieved context"):
                            st.text(context)

                    try:
                        client = get_client()
                        answer = generate_response(
                            client=client,
                            context=context,
                            chat_history=st.session_state.messages[:-1],
                            user_message=user_message,
                            model=model,
                        )
                    except Exception as e:
                        answer = f"Something went wrong calling Gemini: {e}"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
