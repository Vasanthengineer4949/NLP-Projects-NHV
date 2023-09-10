import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from ui import css, bot_template, user_template
from langchain.llms import CTransformers
from langchain import PromptTemplate


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    hfembeddings = HuggingFaceEmbeddings(
                            model_name="thenlper/gte-large", 
                            model_kwargs={'device': 'cpu'}
                        )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=hfembeddings)
    vectorstore.save_local("faiss/medical")

def create_medbot():
    prompt_temp = '''
With the information provided try to answer the question. 
If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
This is related to medical domain. So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

Context: {context}
Question: {question}
Do provide only correct answers

Correct answer:
    '''
    custom_prompt_temp = PromptTemplate(template=prompt_temp,
                            input_variables=['context', 'question'])
    llm = CTransformers(
                model = "res\llama-2-7b-chat.ggmlv3.q4_1.bin",
                model_type="llama",
                max_new_tokens = 512,
                temperature = 0.9
            )

    hfembeddings = HuggingFaceEmbeddings(
                            model_name="thenlper/gte-large", 
                            model_kwargs={'device': 'cpu'}
                        )

    vectorstore = FAISS.load_local("faiss/medical", hfembeddings)
    
    retrieval_qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type="stuff",
                                retriever=vectorstore.as_retriever(search_kwargs={'k': 2}),
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": custom_prompt_temp}
                            )
    return retrieval_qa_chain

def display_conversation(history):

    for i in range(len(history["assistant"])):
        st.write(user_template.replace(
                "{{MSG}}", history["user"][i]), unsafe_allow_html=True)
        st.write(bot_template.replace(
                "{{MSG}}", history["assistant"][i]), unsafe_allow_html=True)
        
def vectorize_pdf():
    st.sidebar.subheader("Your documents")
    pdf_docs = st.sidebar.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.sidebar.button("Process"):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vectorstore(text_chunks)
        st.sidebar.write("Vector Store creation completed")
    return True


def main():
    st.set_page_config(page_title="Chat with a personal doctor - Medbot for personal remedies",
                    page_icon=":heart:")
    st.write(css, unsafe_allow_html=True)

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ["I am ready to help you"]
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hey there!"]

    st.title("Chat with Medbot")
    st.header("Your Home Remedy Doctor :heart:")
    flag = vectorize_pdf()
    user_question = st.text_input("Ask a question about your documents:")
    while flag:
        medbot = create_medbot()
        break         
    if st.button("Answer"):
        answer = medbot({"query": user_question})["result"]
        st.session_state["user"].append(user_question)
        st.session_state["assistant"].append(answer)

    if st.session_state["assistant"]:
        display_conversation(st.session_state)

if __name__ == '__main__':
    main()