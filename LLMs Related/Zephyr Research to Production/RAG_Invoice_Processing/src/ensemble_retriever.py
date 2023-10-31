from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import *

def retriever_creation():
    dir_loader = DirectoryLoader(
                            DATA_DIR_PATH,
                            glob='*.pdf',
                            loader_cls=PyPDFLoader
                        )
    docs = dir_loader.load()
    print("PDFs Loaded")
    
    txt_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=CHUNK_SIZE, 
                            chunk_overlap=CHUNK_OVERLAP
                        )
    inp_txt = txt_splitter.split_documents(docs)
    print("Data Chunks Created")
    print(len(inp_txt))

    hfembeddings = HuggingFaceEmbeddings(
                            model_name=EMBEDDER, 
                            model_kwargs={'device': 'cpu'}
                        )

    faiss_vectorstore = FAISS.from_documents(inp_txt, hfembeddings)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k":2})
    bm25_retriever = BM25Retriever.from_documents(inp_txt)
    ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.5, 0.5])
    print("Vector Store Creation Completed")
    return ensemble_retriever