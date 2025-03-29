Medical Chatbot -
Overview
This project builds an AI-powered chatbot capable of answering medical-related queries using Retrieval-Augmented Generation (RAG). It utilizes FAISS as a vector database for efficient retrieval and Mistral-7B-Instruct as the LLM for generating responses. The chatbot is deployed using Streamlit.

🚀 Features
Document Ingestion: Parses and processes medical PDFs.

Vector Database: Embeds and stores document chunks in FAISS.

LLM Integration: Uses Mistral-7B-Instruct via Hugging Face.

RAG-based Querying: Retrieves relevant document chunks before answering.

Streamlit UI: Provides an interactive chatbot experience.

📁 Project Layout
Phase 1: Setup Memory for LLM (Vector Database)
Load raw PDFs: Extracts text from medical PDFs.

Create Chunks: Splits extracted text into smaller chunks.

Create Vector Embeddings: Converts text chunks into embeddings using sentence-transformers/all-MiniLM-L6-v2.

Store Embeddings in FAISS: Saves embeddings in a FAISS vector database for efficient retrieval.

Phase 2: Connect Memory with LLM
Setup LLM: Uses Mistral-7B-Instruct via Hugging Face for natural language processing.

Connect LLM with FAISS: Retrieves relevant text chunks based on user queries.

Create a Retrieval-Augmented Generation (RAG) Chain: Ensures responses are contextually accurate by retrieving information before generating answers.

Phase 3: Setup UI for the Chatbot
Build Chatbot with Streamlit: Develops an interactive UI for user queries.

Load FAISS Vector Store in Cache: Ensures fast retrieval using Streamlit’s caching mechanism.

Enable RAG-based Retrieval: Enhances chatbot responses by grounding them in retrieved documents.

