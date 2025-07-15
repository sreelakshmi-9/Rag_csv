# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 22:37:03 2025

@author: Sree
"""
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings


# import streamlit as st
# import pandas as pd
# import os
# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_google_genai import ChatGoogleGenerativeAI

# # Set Google API Key (Replace with your actual API key)
# os.environ["GOOGLE_API_KEY"] = "AIzaSyBTZcmeLHgohQetHIbSJ1jKJZg5mxDTsPM"

# # Load Embeddings Model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Streamlit UI
# st.title("📊 RAG Chatbot for CSV Data")
# st.write("Upload a CSV file and ask questions!")

# # Upload CSV file
# uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

# if uploaded_file is not None:
#     # Read CSV
#     df = pd.read_csv(uploaded_file)
    
#     # Convert CSV content into a text format
#     text_data = df.to_string(index=False).split("\n")
    
#     # Create FAISS vector store
#     vectorstore = FAISS.from_texts(text_data, embedding_model)
    
#     # Initialize LLM (Google Gemini Pro)
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro") 
    
#     # Memory for conversation
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
#     # Create RAG-based Chat system
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm, retriever=vectorstore.as_retriever(), memory=memory
#     )
    
#     # Chatbot interface
#     st.subheader("🤖 Ask Your CSV Data")
#     user_query = st.text_input("Ask a question:")
    
#     if user_query:
#         response = qa_chain.run(user_query)
        
#         # Display response
#         st.write("**Chatbot Response:**")
#         st.write(response)



# -*- coding: utf-8 -*-
"""
Created on Mar 21, 2025
Author: Sree
"""

import streamlit as st
import pandas as pd
import os

# Correct imports for newer langchain versions (v0.1.16+)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Load API key from Streamlit secrets (secure method)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="CSV Q&A Chatbot", layout="wide")
st.title("📊 RAG Chatbot for CSV Data")
st.write("Upload a CSV file and ask questions about its content using Google Gemini!")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded successfully!")

    # Display the DataFrame (optional)
    with st.expander("📄 View Data"):
        st.dataframe(df)

    # Convert CSV rows to string chunks (1 row per chunk)
    text_data = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()

    # Create FAISS Vector Store
    vectorstore = FAISS.from_texts(text_data, embedding_model)

    # Initialize Google Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    # Memory (disable messages, since Gemini cannot handle LangChain chat message types)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

    # Conversational RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # Chat interface
    st.subheader("🤖 Ask Your CSV a Question")
    user_query = st.text_input("🔍 Your question:")

    if user_query:
        try:
            response = qa_chain.run(user_query)
            st.write("**💬 Chatbot Response:**")
            st.success(response)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

