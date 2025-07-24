from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

VECTORDB_PATH = "data/vectordb"

def load_and_index_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # o llama2 si usas otro modelo
    vectordb = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=VECTORDB_PATH)
    vectordb.persist()
    return vectordb

def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=VECTORDB_PATH, embedding_function=embeddings)
