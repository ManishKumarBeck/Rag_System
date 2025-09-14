# rag_pipeline.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def ingest_urls(urls):
    # Load documents from all provided URLs
    loader = WebBaseLoader(web_paths=urls)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Generate embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    # Convert to retriever
    retriever = vectorstore.as_retriever()

    # Pull RAG prompt template
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI()

    # Helper function to format retrieved docs
    def format_docs(docs):
        return "\n".join(doc.page_content for doc in docs)

    # Define full RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | RunnableLambda(lambda x: x)  # optional for debugging prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
