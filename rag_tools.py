from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os
load_dotenv()

GPT_EMBEDDING_MODEL = str(os.environ.get("GPT_EMBEDDING_MODEL"))

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(
    model=GPT_EMBEDDING_MODEL
)

#Initialize vector store
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer answers Insurance agents' queries about Real-Time clause and Benefit lookup."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=int(os.environ.get("K")))

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs

rag_tools =[retrieve_context]