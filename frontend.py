import streamlit as st
import tempfile
import os
import backend  # Import backend functions

st.set_page_config(page_title="Chat with PDF", layout="wide")

st.title("📄💬 Chat with Your PDF")

# Sidebar for PDF upload
st.sidebar.header("Upload a PDF to Start")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file:
    st.sidebar.success("PDF uploaded successfully!")

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Process PDF
    backend.create_vector_store(temp_path)

    st.subheader("Ask Questions About Your PDF:")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask me anything about the document...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)

        # Query FAISS
        response = backend.query_pdf(user_input)

        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
