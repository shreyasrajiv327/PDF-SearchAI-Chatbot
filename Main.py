import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import chromadb
os.environ['OPENAI_API_KEY']="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"

def load_chunk_persist_pdf() -> Chroma:
    pdf_folder_path = "/Users/shreyasr/Documents/Pdf_GenAI/Data"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(),
        persist_directory="/Users/shreyasr/Documents/Pdf_GenAI/CromaDB"
    )
    vectordb.persist()
    return vectordb

def create_agent_chain():
  
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain
def get_llm_response(query):
    vectordb = load_chunk_persist_pdf()
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


# Streamlit UI
# ===============
st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
st.header("Query PDF Source")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(get_llm_response(form_input))