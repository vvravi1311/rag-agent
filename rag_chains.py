from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
from rag_tools import rag_tools
from langgraph.graph import MessagesState

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant that answers Insurance agents' questions on Real-Time clause and Benefit lookup. "
            "You have access to a tool that retrieves relevant context from Evidence of coverage and Summary of Benefits documents "
            "Use the tool to find relevant information before answering questions. "
            "Always cite the source and page_num you use in your answers. "
            "If you cannot find the answer in the retrieved documentation, say so."
        ),
        MessagesPlaceholder(variable_name="rag_messages"),
    ]
)

rag_llm = ChatOpenAI(model=os.environ.get("GPT_MODEL"), temperature=0, api_key=os.environ.get("OPENAI_API_KEY")).bind_tools(rag_tools)
rag_chain = rag_prompt | rag_llm