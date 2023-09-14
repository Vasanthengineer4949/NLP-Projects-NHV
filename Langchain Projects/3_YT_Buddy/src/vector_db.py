from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import * 

def faiss_vector_db():

    dir_loader = DirectoryLoader(
                            DATA_DIR_PATH,
                            glob='*.txt',
                            loader_cls=TextLoader
                        )
    docs = dir_loader.load()
    
    txt_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=CHUNK_SIZE, 
                            chunk_overlap=CHUNK_OVERLAP
                        )
    inp_txt = txt_splitter.split_documents(docs)

    hfembeddings = HuggingFaceEmbeddings(
                            model_name=EMBEDDER, 
                            model_kwargs={'device': 'cpu'}
                        )

    db = FAISS.from_documents(inp_txt, hfembeddings)
    db.save_local(VECTOR_DB_PATH)
    print(1)
    return "Vector Store Creation Completed"