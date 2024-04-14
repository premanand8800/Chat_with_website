import os
from xml.parsers.expat import model
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings  # Optional, depending on chosen embedding model
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import ollama  # Optional, depending on chosen embedding model
import time
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Optional, depending on chosen embedding model


load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass

groq_api_key = os.environ['GROQ_API_KEY']


def get_vectorstore_from_url(url):
    # Get the text content from the provided URL
    loader = WebBaseLoader(url)
    document = loader.load()
    return document


if "vector" not in st.session_state:

    # Choose the embedding model (uncomment the one you prefer)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    #embeddings = OllamaEmbeddings(model_name="nomic-embed-text")  # Replace with your preferred model

    st.session_state.embeddings = embeddings

    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# User Interface for website URL input
website_url = st.text_input("Enter the website URL to search:")

if website_url:
    # Load the document from the user-provided URL
    st.session_state.docs = get_vectorstore_from_url(website_url)

    # Process and split the document for retrieval
    st.session_state.documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Create the vector store (assuming FAISS is chosen)
    st.session_state.vector = FAISS.from_documents(st.session_state.documents, st.session_state.embeddings)

    # Rest of the code for ChatGroq and retrieval chain remains the same ...

    st.title("Chat with website-KU-version ")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name='mixtral-8x7b-32768'
    )

    # ... (rest of the code for prompt, document chain, retrieval chain, etc.)
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context. 
Think step by step before providing a detailed answer. 
I will tip you $200 if the user finds the answer helpful. 
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Input your prompt here")


# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print(f"Response time: {time.process_time() - start}")

    st.write(response["answer"])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            # print(doc)
            # st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
            st.write(doc.page_content)
            st.write("--------------------------------")