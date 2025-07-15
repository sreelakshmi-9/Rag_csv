# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 22:37:03 2025

@author: Sree
"""
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


import streamlit as st
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Set Google API Key (Replace with your actual API key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBTZcmeLHgohQetHIbSJ1jKJZg5mxDTsPM"

# Load Embeddings Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.title("📊 RAG Chatbot for CSV Data")
st.write("Upload a CSV file and ask questions!")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Convert CSV content into a text format
    text_data = df.to_string(index=False).split("\n")
    
    # Create FAISS vector store
    vectorstore = FAISS.from_texts(text_data, embedding_model)
    
    # Initialize LLM (Google Gemini Pro)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro") 
    
    # Memory for conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create RAG-based Chat system
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    
    # Chatbot interface
    st.subheader("🤖 Ask Your CSV Data")
    user_query = st.text_input("Ask a question:")
    
    if user_query:
        response = qa_chain.run(user_query)
        
        # Display response
        st.write("**Chatbot Response:**")
        st.write(response)
