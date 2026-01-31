from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
import os
from rag_tools import rag_tools
from langgraph.graph import MessagesState

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Medicare Supplement underwriting assistant for internal agents."
            "Use tools to answer the questions; do not create new underwriting rules."
            "Never fabricate, assume, or hallucinate any values for tool inputs"
            "Only use values explicitly provided by the user."
        ),
        MessagesPlaceholder(variable_name="rag_messages"),
    ]
)

rag_llm = ChatOpenAI(model=os.environ.get("GPT_MODEL"), temperature=0, api_key=os.environ.get("OPENAI_API_KEY")).bind_tools(rag_tools)
rag_chain = rag_prompt | rag_llm