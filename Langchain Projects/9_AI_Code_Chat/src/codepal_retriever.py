from config import *
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def create_codepal_retriever():
    loader = GenericLoader.from_filesystem(
                                            REPO_PATH,
                                            glob=GLOB,
                                            suffixes=SUFFIXES,
                                            parser=LanguageParser(
                                                language=Language.PYTHON, 
                                                parser_threshold=PARSER_THRESHOLD
                                            )
                                        )
    documents = loader.load()
    print(len(documents))

    code_splitter = RecursiveCharacterTextSplitter.from_language(
                                                                language=Language.PYTHON, 
                                                                chunk_size=CHUNK_SIZE, 
                                                                chunk_overlap=CHUNK_OVERLAP
                                                            )
    code_chunks = code_splitter.split_documents(documents)
    print(len(code_chunks))
    
    embeddings = HuggingFaceEmbeddings(
                                        model_name=EMBEDDER, 
                                        model_kwargs=EMBEDDER_KWARGS
                                    )
    db = FAISS.from_documents(code_chunks, embeddings)

    retriever = db.as_retriever(search_kwargs=SEARCH_KWARGS)

    return retriever


    

