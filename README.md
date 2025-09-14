# 🧠 RAG Chatbot with Streamlit, LangChain & OpenAI

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions over content sourced from **PDFs or Web URLs**, using OpenAI’s GPT models — through a clean Streamlit UI.

---

## 🚀 Live Demo

👉 [Click here to open the app](https://openai-rag-k1rc.onrender.com/)  
> ⚠️ **Note:** The app is hosted on a free Render instance, it may take **30–60 seconds to load**. Please be patient while the server wakes up.

---

## ✨ Features

- 📄 Upload **one or more PDFs**
- 🌐 Enter **one or more web URLs**
- 🔄 Choose between **PDF** or **URL** input
- 💬 Ask **natural language questions** via chat
- ⚙️ Customize settings:
  - OpenAI API Key (entered in UI)
  - Model (GPT-3.5 or GPT-4)
  - Temperature, max tokens, chunk size, etc.

---

## 🧰 Tech Stack

| Component       | Description                              |
|----------------|------------------------------------------|
| Streamlit       | UI & frontend                            |
| LangChain       | RAG pipeline + LLM chaining              |
| OpenAI API      | LLMs for answering                       |
| ChromaDB        | In-memory vector database                |
| PyPDF2 / pypdf  | PDF file parsing                         |
| BeautifulSoup4  | Web scraping for URLs                    |
