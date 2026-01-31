
from typing import TypedDict, Annotated, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState, StateGraph,END
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from rag_chains import rag_chain,rag_grounding_chain
from rag_tools import rag_tools
from langgraph.prebuilt import ToolNode
from pprint import pprint
load_dotenv()
import os

GPT_EMBEDDING_MODEL = str(os.environ.get("GPT_EMBEDDING_MODEL"))
RAG_AGENT_REASON="rag_agent_reason"
RAG_TOOL_NODE= "rag_tool_node"
RAG_GROUNDING_REASON = "rag_grounding_reason"

LAST = -1
SECOND_LAST = -2
FIRST = 0

# Initialize embeddings (same as ingestion.py)
embeddings = OpenAIEmbeddings(
    model=GPT_EMBEDDING_MODEL
)

#Initialize vector store
vectorstore = PineconeVectorStore(
    index_name=os.environ.get("INDEX_NAME"), embedding=embeddings
)


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    userQuery: str

def rag_agent_reason(state: MessageGraph):
    return {"messages": [rag_chain.invoke({"rag_messages": state["messages"]})]}

def rag_grounding_reason(state: MessageGraph):
    question = state["messages"][FIRST].content
    answer = state["messages"][LAST].content
    context = ""
    if has_tool_message(state):
        tool_msg = next(msg for msg in state["messages"] if isinstance(msg, ToolMessage))
        context = tool_msg.content
    print("***********************      question, answer, context     *****************************")
    print(question, answer, context)

    return {
        "messages": [rag_grounding_chain.invoke({
            "question": state["messages"],
            "context": context,
            "answer": answer})]
    }

rag_tool_node = ToolNode(rag_tools)

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return RAG_GROUNDING_REASON
    return RAG_TOOL_NODE

flow = StateGraph(MessagesState)
flow.add_node(RAG_AGENT_REASON, rag_agent_reason)
flow.add_node(RAG_TOOL_NODE, rag_tool_node)
flow.add_node(RAG_GROUNDING_REASON, rag_grounding_reason)

flow.set_entry_point(RAG_AGENT_REASON)
flow.add_conditional_edges(RAG_AGENT_REASON, should_continue, {
    RAG_GROUNDING_REASON:RAG_GROUNDING_REASON,
    RAG_TOOL_NODE:RAG_TOOL_NODE})
flow.add_edge(RAG_TOOL_NODE, RAG_AGENT_REASON)
flow.add_edge(RAG_GROUNDING_REASON, END)

uw_flow = flow.compile()
uw_flow.get_graph().draw_mermaid_png(output_file_path="rag_agent_graph.png")

def has_tool_message(result):
    # Case 1: result is a single message
    if hasattr(result, "tool_calls") and result.tool_calls:
        return True

    # Case 2: result is a dict with messages
    if isinstance(result, dict) and "messages" in result:
        for msg in result["messages"]:
            if getattr(msg, "type", None) == "tool":
                return True
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return True
    return False

def run_rag_graph(query: str) -> Dict[str, Any]:
    result = uw_flow.invoke({"messages": [HumanMessage(
        content=query)]})
    answer = result["messages"][SECOND_LAST].content
    audit = []
    if has_tool_message(result):
        # print("*************** A tool Message is present   **********************")
        answer = result["messages"][SECOND_LAST ].content
        print(answer)
        tool_msg = next(msg for msg in result["messages"] if isinstance(msg, ToolMessage))
        # print(tool_msg)
        for artifact in tool_msg.artifact:
            # print(artifact.metadata["source"])
            # print(artifact.metadata["MY_page_number"])
            audit.append({
                "source": artifact.metadata["source"],
                "page_number": artifact.metadata["MY_page_number"]
            })

    return {
        "answer": answer,
        "audit": audit,
        "grounded_agent_info" : result["messages"][LAST].content
    }

if __name__ == "__main__":
    print("Hello from ReAct RAG Agent")
    # query = "If I have Plan N, how much do I pay for an emergency room visit if I am admitted to the hospital"
    # query = "I’m looking at Plan N and Plan G. If my doctor does not accept 'assignment' and charges more than the Medicare-approved amount, which plan protects me from the balance bill"
    # query = "what is Plan N"
    query = "Can you tell me about Medicare plans in Pakistan"
    result = run_rag_graph(query)
    pprint(result)