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
import streamlit as st
st.set_page_config(page_title="CSV Q&A Chatbot", layout="wide")

import pandas as pd
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI


# ✅ HARD-CODED API KEY (⚠️ Do NOT use in public repositories)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBTZcmeLHgohQetHIbSJ1jKJZg5mxDTsPM"  # Replace with your actual key

# ✅ Imports for LangChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Configure Streamlit UI
st.set_page_config(page_title="CSV Q&A Chatbot", layout="wide")
st.title("📊 RAG Chatbot for CSV Data")
st.markdown("Upload a CSV file and ask questions using **Google Gemini**.")

# ✅ Upload CSV file
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # ✅ Read the file
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV loaded successfully!")

    # Show data preview
    with st.expander("📄 View Data"):
        st.dataframe(df)

    # ✅ Convert CSV rows into text format
    text_data = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()

    # ✅ Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Create FAISS vectorstore
    vectorstore = FAISS.from_texts(text_data, embedding_model)

    # ✅ Load Gemini model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    # ✅ Use simple memory (Gemini doesn't support full LangChain memory messages)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

    # ✅ Set up Conversational Retrieval Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # ✅ User input interface
    st.subheader("💬 Ask Your CSV a Question")
    user_query = st.text_input("🔍 Type your question:")

    if user_query:
        try:
            response = qa_chain.run(user_query)
            st.markdown("**🤖 Chatbot Response:**")
            st.success(response)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
