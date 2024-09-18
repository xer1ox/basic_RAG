# import libraries
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load the PDF data
local_path = "C:\\Users\\alael\\Rag_from_scratch\\Data\\Beyond PLM Blog.pdf"
loader = PyPDFLoader(file_path=local_path)
data = loader.load()

print (len(data))

# Split and chunk 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

# Add to vector database
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
    collection_name="local-rag"
)

# Funtion for Similarity search
def retrieve_info(query):
    similar_info = vector_db.similarity_search(query, k=3)
    
    page_contents_array = [doc.page_content for doc in similar_info]

    page_metadata_array = []
    for sim in similar_info:
        source = sim.metadata.get("source")
        page = sim.metadata.get("page")
        page_id = f"{source}:{page}"
        page_metadata_array.append(page_id)

    print("\n\n".join(page_metadata_array))
    #print("\n\n".join(doc.page_content for doc in similar_info))
    #print(page_contents_array)

    return page_contents_array


query = "What is the PLM End Game"
results = retrieve_info(query)
print(results)


# # Set up the LLMChain & prompt

# local_model = "llama3.1"
# llm = ChatOllama(model=local_model)

# template = """Answer the question based ONLY on the following context:
# {context}
# Question: {question}
# """
# prompt = PromptTemplate(
#     input_variables= ["question", "context"],
#     template=template
# )

# chain = LLMChain(llm=llm, prompt=prompt)

# # Retrival Augmented generation
# def generate_response(question):
#     context = retrieve_info(question)
#     response = chain.run(question=question, context=context)
#     return response



# # Build an app with streamlit
# def main():
#     st.set_page_config(
#         page_title="Local knowledge base ChatBot", page_icon=":bird:")
    
#     st.header("Local knowledge base ChatBot :bird:")
#     question = st.text_area("Question prompt")

#     if question:
#         st.write("Generating response based on knowledge base...")

#         result = generate_response(question)

#         st.info(result)


# if __name__ == '__main__':
#     main()