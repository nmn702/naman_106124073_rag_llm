This project is a Retrieval-Augmented Generation application that allows users to ask questions based on the contents of several academic PDFs. It uses:

1. LangChain for orchestration
2. Google Gemini 2.0 Flash for LLM responses
3. Chroma as the vector database
4. HuggingFace Sentence Transformers for embeddings
5. Streamlit for a user friendly web interface


Features:
1. Loads and parses multiple PDF documents
2. Splits documents into chunks and stores them in a Chroma vector database
3. Accepts user queries and retrieves the most relevant chunks
4. Uses Google Gemini to generate accurate answers based on retrieved context
