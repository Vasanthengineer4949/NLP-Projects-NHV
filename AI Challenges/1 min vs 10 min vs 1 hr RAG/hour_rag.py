from langchain_community.document_loaders.pdf import OnlinePDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
import os
import time

load_dotenv(find_dotenv())

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_ap_keyi"
os.environ['LANGCHAIN_PROJECT'] = '1hour-rag'

class RAG:

    def __init__(self, web_pdf_url):
        
        self.vectorstore_index_name = "1hour-rag"
        self.loader = OnlinePDFLoader(web_pdf_url)
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=os.getenv("HF_INFERENCE_API_KEY"), model="mixedbread-ai/mxbai-embed-large-v1"
        )
        self.groq_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"), 
            model="llama3-8b-8192", 
            temperature=0
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=100
        )
        self.create_pinecone_index(self.vectorstore_index_name)
        self.vectorstore = PineconeVectorStore(
            index_name=self.vectorstore_index_name,
            embedding=self.embeddings,
            pinecone_api_key=os.getenv("PINECONE_API_KEY")
        )
        self.rag_prompt = hub.pull(
            "rlm/rag-prompt", 
            api_key=os.getenv("LANGSMITH_API_KEY")
        )
        config = RailsConfig.from_path("./config")

        self.guardrails = RunnableRails(config=config,llm=self.groq_llm)


    def create_pinecone_index(self, vectorstore_index_name):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  
        spec = ServerlessSpec(cloud='aws', region='us-east-1')  
        if vectorstore_index_name in pc.list_indexes().names():  
            pc.delete_index(vectorstore_index_name)  
        pc.create_index(  
            vectorstore_index_name,  
            dimension=384,
            metric='dotproduct',  
            spec=spec  
        )  
        while not pc.describe_index(vectorstore_index_name).status['ready']:  
            time.sleep(1) 

    def load_docs_into_vectorstore_chain(self):
        docs = self.loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        self.vectorstore.add_documents(split_docs)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_retrieval_chain(self):
        self.load_docs_into_vectorstore_chain()
        self.retriever = self.vectorstore.as_retriever()
        self.rag_chain = (
                        {
                            "context": self.retriever | self.format_docs, "question": RunnablePassthrough()
                        }
                        | self.rag_prompt
                        | self.groq_llm
                        | StrOutputParser()
                    )
        self.rag_chain = self.guardrails | self.rag_chain
    def qa(self, query, vectorstore_created):
        if vectorstore_created:
            pass
        else:
            self.create_retrieval_chain()
        return self.rag_chain.invoke(query), True

if __name__=="__main__":
    rag = RAG("https://files.eric.ed.gov/fulltext/EJ1172284.pdf")
    print(rag.qa("what is the university name", False))
    