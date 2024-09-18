# import libraries
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


@st.cache_resource
def load_pdf_and_create_vector_db(local_path):
    # Load the PDF data
    loader = PyPDFLoader(file_path=local_path)
    data = loader.load()
    # Split and chunk 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    
    # Add to vector database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
        collection_name="local-rag"
    )

    return vector_db

# Funtion for Similarity search
def retrieve_info(query, vector_db):
    similar_info = vector_db.similarity_search(query, k=3)
    
    page_contents_array = [doc.page_content for doc in similar_info]
    
    page_metadata_array = []
    for sim in similar_info:
        source = sim.metadata.get("source")
        page = sim.metadata.get("page")
        page_id = f"{source}:{page}"
        page_metadata_array.append(page_id)

    print("\n\n".join(page_metadata_array))

    return page_contents_array


# Set up the LLMChain & prompt

local_model = "llama3.1"
llm = ChatOllama(model=local_model)

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""
prompt = PromptTemplate(
    input_variables= ["question", "context"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Retrival Augmented generation
def generate_response(question, vector_db):
    context = retrieve_info(question,vector_db)
    response = chain.run(question=question, context=context)
    return response



# Build an app with streamlit
def main():
    st.set_page_config(
        page_title="Local knowledge base ChatBot", page_icon=":bird:")
    
    st.header("Local knowledge base ChatBot :bird:")

    # define the path to the PDF file
    local_path = "C:\\Users\\alael\\Rag_from_scratch\\Data\\incose_vision_2035.pdf"

    # load the vector database
    vector_db = load_pdf_and_create_vector_db(local_path)

    question = st.text_area("Question prompt")

    if question:
        st.write("Generating response based on knowledge base...")

        result = generate_response(question, vector_db)

        st.info(result)


if __name__ == '__main__':
    main()