import streamlit as st
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
import bs4  # Import BeautifulSoup for parsing
import langchain
import os
import chromadb
from chromadb import Settings


client = chromadb.Client(Settings(is_persistent=True,
                                    persist_directory="/Users/thibaut/Documents/myfolder/chroma_db",
                                ))
coll = client.get_collection("reddit")
print(coll.get()) # Gets all the data

langchain.debug=True
os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Initialize Streamlit app
st.title("RAG Chatbot")

embedding_function = OpenAIEmbeddings()

# Specify Chroma specified directory used by Amphi pipeline and ChrombaDB collection
vectorstore = Chroma(persist_directory="/myfolder/chroma_db", embedding_function=embedding_function, collection_name = 'reddit')

# Set up the retriever and prompt
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

llm = OpenAI()  # Initialize the language model

def format_docs(docs):
    formatted_docs = "\n\n".join(doc.page_content for doc in docs)
    st.write("Documents sent to the LLM:")
    for doc in docs:
        st.code(doc.page_content)
    return formatted_docs

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit interface for chatbot
use_rag = st.checkbox("Use RAG", value=True)
user_question = st.text_input("Ask a question:")

if user_question:
    if use_rag:
        # Retrieve documents
        retrieved_docs = retriever.get_relevant_documents(user_question)
        
        # Display the retrieved documents
        #st.write("Retrieved Documents:")
        #for doc in retrieved_docs:
           # st.write(doc.page_content)
        
        # Format and process the documents
        formatted_docs = format_docs(retrieved_docs)
        
        # Create the input for the RAG chain
        rag_input = f"Context:\n{formatted_docs}\n\nQuestion: {user_question}"
        
        # Get the answer
        answer = rag_chain.invoke(rag_input)
    else:
        # Directly use the LLM without RAG
        answer = llm(user_question)
    
    st.write("Answer:")
    st.write(answer)