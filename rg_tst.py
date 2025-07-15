# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 22:37:03 2025

@author: Sree
"""






import streamlit as st

# ✅ PAGE CONFIG MUST BE THE FIRST STREAMLIT CALL
st.set_page_config(page_title="CSV Q&A Chatbot", layout="wide")

import pandas as pd
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Hardcoded API key (for local/dev use only)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBTZcmeLHgohQetHIbSJ1jKJZg5mxDTsPM"

# ✅ App title
st.title("📊 RAG Chatbot for CSV Data")
st.markdown("Upload a CSV file and ask questions using **Google Gemini 1.5 Pro**.")

# ✅ File uploader
uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # ✅ Load the CSV file
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV file loaded!")

    # ✅ Preview data
    with st.expander("📄 View Data"):
        st.dataframe(df)

    # ✅ Prepare text chunks
    text_data = df.astype(str).apply(lambda row: " | ".join(row), axis=1).tolist()

    # ✅ Load embedding model and FAISS
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_data, embedding_model)

    # ✅ Load Gemini LLM
   # llm = ChatGoogleGenerativeAI(model="gemini-pro")
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    convert_system_message_to_human=True
)


    # ✅ Simple memory (Gemini does not support full message history)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

    # ✅ Set up the RAG pipeline
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    # ✅ Chat input
    st.subheader("💬 Ask your CSV a question")
    user_query = st.text_input("🔍 Enter your question:")

    if user_query:
        try:
            response = qa_chain.run(user_query)
            st.markdown("**🤖 Chatbot Response:**")
            st.success(response)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

