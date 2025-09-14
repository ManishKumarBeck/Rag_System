# app.py

import streamlit as st
from rag_pipeline import ingest_urls

st.set_page_config(page_title="RAG Chatbot", layout="centered")

st.title("📚 Retrieval-Augmented Chatbot")
st.write("Ask questions based on content from one or more web pages.")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# URL input form
with st.form("url_form"):
    url_input = st.text_area("Enter one or more URLs (comma-separated):", 
                              placeholder="https://en.wikipedia.org/wiki/Marie_Curie, https://en.wikipedia.org/wiki/Volkswagen")
    submitted = st.form_submit_button("Ingest URLs")

    if submitted and url_input:
        with st.spinner("Loading and indexing documents..."):
            try:
                url_list = [url.strip() for url in url_input.split(",") if url.strip()]
                st.session_state.rag_chain = ingest_urls(url_list)
                st.session_state.chat_history = []
                st.success("✅ Documents loaded and indexed. Ask your questions below!")
            except Exception as e:
                st.error(f"Error processing URLs: {e}")

# Chat input
if st.session_state.rag_chain:
    user_input = st.chat_input("Ask a question...")

    if user_input:
        with st.spinner("Generating answer..."):
            try:
                response = st.session_state.rag_chain.invoke(user_input)
                st.session_state.chat_history.append(("user", user_input))
                st.session_state.chat_history.append(("bot", response))
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Display chat history
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(message)
        else:
            st.chat_message("assistant").markdown(message)
