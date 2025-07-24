from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.schema import Document
from typing import List
from pydantic import BaseModel, Field
from utils.rag_utils import get_vectorstore

RETRIEVER_K = 4

# Definición del esquema del estado con Pydantic
class AgentStateModel(BaseModel):
    input: str
    history: List[dict] = Field(default_factory=list)  # [{"role": "user"/"assistant", "content": str}]
    docs: List[Document] = Field(default_factory=list)
    output: str = ""

# Nodo 1: Recuperación de contexto (RAG)
def get_relevant_docs_node():
    retriever = get_vectorstore().as_retriever(search_type="similarity", k=RETRIEVER_K)

    def retrieve(state: AgentStateModel) -> AgentStateModel:
        query = state.input
        docs = retriever.get_relevant_documents(query)
        state.docs = docs
        return state

    return RunnableLambda(retrieve)

# Nodo 2: Generación de respuesta usando contexto + memoria
def get_generation_node():
    llm = ChatOllama(model="llama3.2:3b-instruct-q4_K_M")
    prompt = PromptTemplate(
        template="""
Eres un asistente experto en investigación académica. 
Responde con base en los siguientes documentos:

{context}

Pregunta del usuario:
{question}
""",
        input_variables=["context", "question"]
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    def generate(state: AgentStateModel) -> AgentStateModel:
        context = "\n\n".join(doc.page_content for doc in state.docs)
        question = state.input
        result = chain.run({"context": context, "question": question})
        state.output = result
        return state

    return RunnableLambda(generate)

# Crear el grafo con el esquema correcto
graph = StateGraph(state_schema=AgentStateModel)
graph.add_node("retrieve_docs", get_relevant_docs_node())
graph.add_node("generate_response", get_generation_node())

graph.set_entry_point("retrieve_docs")
graph.add_edge("retrieve_docs", "generate_response")
graph.add_edge("generate_response", END)

agent_graph = graph.compile()

# Función para conversar
def chat_with_agent(user_input: str, history: List[dict]):
    result_state = agent_graph.invoke({
        "input": user_input,
        "history": history,
        "docs": [],
        "output": ""
    })
    return result_state["output"]

