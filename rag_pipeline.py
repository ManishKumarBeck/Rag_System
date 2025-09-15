import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain.document_loaders import PyPDFLoader  # PDF loader
from langchain_community.document_loaders import WebBaseLoader  # URL loader

import tempfile
import os

load_dotenv()

DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")

def ingest_urls(urls, user_api_key=None, model_name="gpt-3.5-turbo", temperature=0.2, max_tokens=500,
                chunk_size=1000, chunk_overlap=200, top_k=4):
    api_key = user_api_key or DEFAULT_API_KEY
    if not api_key:
        raise ValueError("No OpenAI API key provided!")

    os.environ["OPENAI_API_KEY"] = api_key

    loader = WebBaseLoader(web_paths=urls)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(openai_api_key=api_key, model=model_name, temperature=temperature, max_tokens=max_tokens)

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | RunnableLambda(lambda x: x)
        | llm
        | StrOutputParser()
    )
    return rag_chain


def ingest_pdfs(pdf_files, user_api_key=None, model_name="gpt-3.5-turbo", temperature=0.2, max_tokens=500,
                chunk_size=1000, chunk_overlap=200, top_k=4):
    api_key = user_api_key or DEFAULT_API_KEY
    if not api_key:
        raise ValueError("No OpenAI API key provided!")

    os.environ["OPENAI_API_KEY"] = api_key

    documents = []

    for uploaded_file in pdf_files:
        # Save uploaded file to temp path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        # Now close the file and load it using PyPDFLoader
        try:
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            documents.extend(docs)
        finally:
            # Delete the temp file manually
            os.remove(temp_path)

    # Text splitting, embedding, retrieval, RAG chain 
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(openai_api_key=api_key, model=model_name, temperature=temperature, max_tokens=max_tokens)

    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | RunnableLambda(lambda x: x)
        | llm
        | StrOutputParser()
    )
    return rag_chain
