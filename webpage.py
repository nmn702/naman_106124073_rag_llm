import streamlit as st
import os
import getpass
from ragpipe import build_vectorstore,build_graph

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = st.text_input("Enter your Google Gemini API key:", type="password")


if "graph" not in st.session_state:
    with st.spinner("Loading vector store and building graph..."):
        build_vectorstore()
        st.session_state.graph = build_graph()

st.title("RAG Question Answering")

question=st.text_input("Ask a question based on the loaded PDFs:")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        response=st.session_state.graph.invoke({"question": question})
        st.success(response["answer"])
