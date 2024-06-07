import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Function to create a vector store from a website URL
def get_vectore_store_url(url):
    # Load the document from the provided URL
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents=document)
    
    # Create a vector store using the document chunks and OpenAI embeddings
    vectore_store = Chroma.from_documents(documents=document_chunks, embedding=OpenAIEmbeddings())
    return vectore_store

# Function to create a context-aware retriever
def get_context_retriever(vectore_store):
    # Define the prompt template for the retriever
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', "{input}"),
        ('user', "Given the above conversation, generate a search query to look up in order to get the relevant information based on the conversation")
    ])
    
    # Create a chat model using OpenAI
    llm = ChatOpenAI()
    
    # Create a retriever from the vector store
    retriever = vectore_store.as_retriever()
    
    # Create a history-aware retriever chain
    history_aware_retriever_chain = create_history_aware_retriever(
        llm, retriever, prompt
    )
    return history_aware_retriever_chain

# Function to create a retrieval-augmented generation (RAG) chain
def get_rag_chain(history_aware_retriever_chain):
    # Define the prompt template for the RAG chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on below context:\n\n{context}"),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', "{input}"),
    ])
    
    # Create a chat model using OpenAI GPT-3.5 Turbo
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    # Create a document chain using the LLM and prompt
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever_chain, stuff_documents_chain)
    return rag_chain

# Function to get a response from the RAG chain based on user query
def get_response(user_query):
    # Get the context retriever chain
    context_retriever_chain = get_context_retriever(vectore_store=st.session_state.vector_store)
    
    # Get the RAG chain
    rag_chain = get_rag_chain(context_retriever_chain)
    
    # Get the response from the RAG chain
    response = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    return response['answer']

# Streamlit app configuration
st.set_page_config(page_icon='🌐', page_title='WebChatAI')
st.title('WebChatAI')
with st.expander("💡 Tips", expanded=False):
        st.write(
        """
           **WebChatAI** is a website featuring an AI-powered chat interface that interacts with specified websites, allowing users to ask questions about the content. It works by loading and processing website content, converting it into vector embeddings, and using them to generate responses. The application is built using Streamlit and utilizes various tools and libraries such as LangChain Core, OpenAIEmbeddings, and Chroma for document processing and AI interaction.

            To use WebChatAI, simply enter a website URL into the provided text input field, ask your question in the chat interface, and the AI will provide relevant responses based on the website's content.
        """
        )
# Sidebar for settings
with st.sidebar:
    st.title('Settings')
    website_url = st.text_input('Enter your website URL')
    st.info('example of a URL : https://www.langchain.com/langsmith')

# Main logic for the app
if website_url != "" and website_url is not None:
    
    # Initialize chat history if not present
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content='Hi there, how can I help you!')
        ]
    
    # Initialize vector store if not present
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = get_vectore_store_url(website_url)

    # Get user query from chat input
    user_query = st.chat_input('Enter your query..!')
    
    if user_query != "" and user_query is not None:
        # Append user query to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        # Get response from RAG chain and append to chat history
        st.session_state.chat_history.append(AIMessage(content=get_response(user_query=user_query)))
    
        # Display chat history
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message('Human'):
                    st.write(message.content)
            if isinstance(message, AIMessage):
                with st.chat_message('AI'):
                    st.write(message.content)
else:
    # Display an information message if website URL is not provided
    st.info(' <-- Provide your website URL to proceed')
