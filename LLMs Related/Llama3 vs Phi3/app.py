from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.llms.ollama import Ollama
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import streamlit as st
import uuid
import os
import base64
import gc
import tempfile

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None
    gc.collect()


def display_pdf(file):

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:

    selected_model = st.selectbox(
            "Select your LLM:",
            ("Phi-3", "Llama-3"),
            index=0,
            key='selected_model'  
        )
    

    st.header(f"Add your documents!")

    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
    
    if uploaded_file:
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
                st.session_state.current_model = selected_model
                st.session_state.file_cache.pop(file_key, None)  
                st.experimental_rerun()  

            if st.session_state.current_model == "Llama-3":
                # llm = Ollama(model="llama3")
                llm = ChatGroq(
                    groq_api_key="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", model_name="llama3-8b-8192"
                    )


                llm = ChatGroq()

            elif st.session_state.current_model == "Phi-3":
                llm = Ollama(model="phi3")

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                            loader = PyPDFLoader(
                                file_path = temp_dir+"/"+uploaded_file.name
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    
                    docs = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter()
                    documents = text_splitter.split_documents(docs)

                    # setup embedding model
                    embed_model = HuggingFaceEmbeddings(model_name="embedder", model_kwargs={"device": "cpu"})

                    vectorstore = FAISS.from_documents(documents, embed_model)

                    retriever = vectorstore.as_retriever(search_kwargs={"k":3})

                    prompt = PromptTemplate.from_template(
                        """Answer the following question based only on the provided context:
                        <context>
                        {context}
                        </context>

                        Question: {input}
                        """)

                    document_chain = create_stuff_documents_chain(llm, prompt)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
                st.success("Ready to Chat!")
                display_pdf(uploaded_file)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with your Docs! 📄")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

if "messages" not in st.session_state:
    reset_chat()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        response = retrieval_chain.invoke({"input": prompt})

        message_placeholder.markdown(response["answer"])

    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})