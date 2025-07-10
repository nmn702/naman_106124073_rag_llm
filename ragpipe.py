import getpass
import os
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()

os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#defining which llm to use
llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
##defining the vector base
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  #Where to save data locally
)
prompt = hub.pull("rlm/rag-prompt")

#loading the documents fucntion
def load_documents():
    pdf_files = ["bert.pdf", "graphs.pdf", "attention.pdf", "gpt3.pdf", "llama.pdf"]
    pages = []
    for file in pdf_files:
        loader = PyPDFLoader(file)
        pages.extend(loader.load())
    return pages

#building the vector store and loading the documents to vectorstore
def build_vectorstore():
    pages = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
    all_splits=text_splitter.split_documents(pages)
    vector_store.add_documents(all_splits)

class State(TypedDict):
    question:str
    context:List[Document]
    answer:str

#getting relevant information from pdfs
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=5)
    return {"context": retrieved_docs}

#generating the answer
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def build_graph():
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()