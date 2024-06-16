import bs4
from langchain import hub
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os

os.environ["GROQ_API_KEY"] = "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"

from langchain_groq import ChatGroq

llm = ChatGroq(model="llama3-8b-8192")

# Load, chunk and index the contents of the blog.
loader = PyPDFLoader("AI THUMBNAIL PROMPTS.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = FAISS.from_documents(documents=splits, embedding=HuggingFaceBgeEmbeddings())
# vectorstore.save_local("faiss/10min")
vectorstore.load_local("faiss/10min", HuggingFaceBgeEmbeddings(), allow_dangerous_deserialization=True)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

input_s = input("Q:")
print(rag_chain.invoke(input_s))