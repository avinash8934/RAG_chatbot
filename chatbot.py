from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_core.tools import tool

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import TypedDict, Annotated, Literal
from uuid import uuid4
from config import *

# === LangGraph State ===
class ChatState(TypedDict):
	messages: Annotated[list, add_messages]

# === Load VectorStore ===
print("Getting embeddings and vector_store may take some time...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDER_MODEL)
vectorstore = Chroma(
	persist_directory=DB_URI,
	collection_name=DB_COLLECTION,
	embedding_function=embedding_model
)

# ===  Tool Node: Retrieval ===
@tool
def retrieve_context(query: str):
	"""Search for the question in medical documents.
It should be used only when the question is related to medical topics or healthcare information.
The input should be a question or query that needs to be searched in the documents.
It will return relevant document results.
Example: "What are the symptoms of type 2 diabetes?
	"""
	
	results = vectorstore.similarity_search(query, k=4)
	print("Search Completed, retrieved results:", len(results))

	result = "\n".join( f"{res.page_content} \n Relevant Options: {res.metadata.get('options')}" for res in results)
	return result

# === Prompt + LLM ===
llm = ChatOllama(model=LLM_MODEL)

prompt = ChatPromptTemplate.from_messages([
	("system", SYSTEM_PROMPT),
	("user", "{input}")
])
def go_back(state: MessagesState) :
	messages = state['messages']
	last_message = messages[-1]
	if last_message.tool_calls:
		return "search"
	return END

# === Model Node ===
def call_model(state: ChatState):
	last_user_msg = state["messages"]
	rag_chain = prompt | llm
	response = rag_chain.invoke(last_user_msg)
	return {"messages": [response]}

# === LangGraph ===
graph_builder = StateGraph(ChatState)
graph_builder.add_node("model", call_model)
graph_builder.add_node("search", retrieve_context)
graph_builder.add_edge(START, "model")
graph_builder.add_conditional_edges("model",go_back)
graph_builder.add_edge("search", "model")
graph = graph_builder.compile(checkpointer=MemorySaver())
print("Graph created Successfully...")

convo_state = {}
# ===  CLI Chat Loop ===
def chat():
	print("Bot Started...")
	print("Type 'exit' to quit.\n")

	session_id = str(uuid4())

	while True:
		user_input = input("\nYou: ")
		if user_input.strip().lower() in ["exit", "quit"]:
			break
		if session_id in convo_state:
			state = convo_state[session_id]
			state['messages'].append(('user', user_input))
			config = {"configurable": {"thread_id": session_id}}
		else:
			state = {"messages": [('user', user_input)]}
			config = {"configurable": {"thread_id": session_id}}

		state = graph.invoke(state, config = config)

		print(f"\033[92mBot: {state['messages'][-1].content}\033[0m")

if __name__ == "__main__":
	chat()
