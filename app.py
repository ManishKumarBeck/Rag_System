import streamlit as st
from rag_pipeline import ingest_urls, ingest_pdfs

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📚 Retrieval-Augmented Chatbot")

# Sidebar settings (simplified for brevity)
user_api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password", help="Enter your OpenAI API key or leave blank to use default.")

model_name = st.sidebar.selectbox("🤖 Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
temperature = st.sidebar.slider("🎨 Temperature", 0.0, 1.0, 0.2)
max_tokens = st.sidebar.slider("🧠 Max Tokens", 100, 2048, 500)
chunk_size = st.sidebar.number_input("📚 Chunk Size", 100, 2000, 1000, 100)
chunk_overlap = st.sidebar.number_input("🔁 Chunk Overlap", 0, 500, 200, 50)
top_k = st.sidebar.slider("🔍 Top K Retrieved Chunks", 1, 10, 4)

# Choose input type
input_type = st.radio("Choose input type:", ("URLs", "PDFs"))

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("input_form"):
    if input_type == "URLs":
        url_input = st.text_area("Enter one or more URLs (comma-separated):",
                                 placeholder="https://en.wikipedia.org/wiki/Marie_Curie, https://en.wikipedia.org/wiki/Volkswagen")
    else:
        pdf_files = st.file_uploader("Upload one or more PDF files:", type=["pdf"], accept_multiple_files=True)

    submitted = st.form_submit_button("Ingest")

    if submitted:
        if input_type == "URLs":
            if not url_input.strip():
                st.error("Please enter at least one URL.")
            else:
                try:
                    url_list = [u.strip() for u in url_input.split(",") if u.strip()]
                    with st.spinner("Loading and indexing URLs..."):
                        st.session_state.rag_chain = ingest_urls(
                            urls=url_list,
                            user_api_key=user_api_key,
                            model_name=model_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            top_k=top_k,
                        )
                        st.session_state.chat_history = []
                        st.success("✅ URLs ingested successfully. Ask your questions below!")
                except Exception as e:
                    st.error(f"Error ingesting URLs: {e}")

        else:  # PDFs
            if not pdf_files:
                st.error("Please upload at least one PDF file.")
            else:
                try:
                    with st.spinner("Loading and indexing PDFs..."):
                        st.session_state.rag_chain = ingest_pdfs(
                            pdf_files=pdf_files,
                            user_api_key=user_api_key,
                            model_name=model_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            top_k=top_k,
                        )
                        st.session_state.chat_history = []
                        st.success("✅ PDFs ingested successfully. Ask your questions below!")
                except Exception as e:
                    st.error(f"Error ingesting PDFs: {e}")

if st.session_state.rag_chain:
    user_question = st.chat_input("Ask a question...")

    if user_question:
        with st.spinner("Generating answer..."):
            try:
                answer = st.session_state.rag_chain.invoke(user_question)
                st.session_state.chat_history.append(("user", user_question))
                st.session_state.chat_history.append(("bot", answer))
            except Exception as e:
                st.error(f"Error generating answer: {e}")

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.chat_message("user").markdown(msg)
        else:
            st.chat_message("assistant").markdown(msg)
