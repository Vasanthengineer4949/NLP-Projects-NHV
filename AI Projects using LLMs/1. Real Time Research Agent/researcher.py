from config import *
import os
from dotenv import load_dotenv, find_dotenv
import json
import requests
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
load_dotenv(find_dotenv())

class Researcher:

    def __init__(self):
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=INPUT_VARIABLES
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.llm = ChatGroq(temperature=0.5, model_name="mixtral-8x7b-32768", groq_api_key=self.groq_api_key)
        self.hfembeddings = HuggingFaceEmbeddings(
                            model_name=EMBEDDER, 
                            model_kwargs={'device': 'cuda'}
                        )

    def search_articles(self, query):

        url = "https://google.serper.dev/search"
        data = json.dumps({"q":query})

        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=data)

        return response.json()
    
    def research_answerer(self):
    
        research_qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=CHAIN_TYPE,
                retriever= self.db.as_retriever(search_kwargs=SEARCH_KWARGS),
                return_source_documents=True,
                verbose=True,
                chain_type_kwargs={"prompt": self.prompt_template}
            )
        return research_qa_chain

    def get_urls(self, articles):
        urls = []
        try:
            urls.append(articles["answerBox"]["link"])
        except:
            pass
        for i in range(0, min(3, len(articles["organic"]))):
            urls.append(articles["organic"][i]["link"])
        return urls
    
    def get_content_from_urls(self, urls):
        loader = UnstructuredURLLoader(urls=urls)
        research_content = loader.load()
        return research_content
    
    def research_given_query(self, research_objective, research_content):
        
        docs = self.text_splitter.split_documents(research_content)
        self.db = FAISS.from_documents(documents=docs, embedding=self.hfembeddings)
        bot = self.research_answerer()
        research_out =bot({"query": research_objective})
        return research_out["result"]

    def research(self, query):
        search_articles = self.search_articles(query)
        urls = self.get_urls(search_articles)
        research_content = self.get_content_from_urls(urls)
        answer = self.research_given_query(query, research_content)
        return answer