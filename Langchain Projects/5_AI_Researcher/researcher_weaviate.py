from config import *
import os
from dotenv import load_dotenv, find_dotenv
import json
import requests
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
import os
import weaviate
load_dotenv(find_dotenv())

class Researcher:

    def __init__(self):
        self.serper_api_key = os.getenv("SERPER_API_KEY")
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
        auth_config = weaviate.AuthApiKey(api_key=self.weaviate_api_key)
        self.client = weaviate.Client(
            url="https://ai-researcher-cluster-8nct9dkc.weaviate.network",
            auth_client_secret=auth_config
        )
        self.prompt_template = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=INPUT_VARIABLES
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=SEPARATORS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.llm = CTransformers(
            model=MODEL,
            model_type=MODEL_TYPE,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE
        ) 
        self.hfembeddings = HuggingFaceEmbeddings(
                            model_name=EMBEDDER, 
                            model_kwargs={'device': 'cpu'}
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
        self.db = Weaviate.from_documents(docs, self.hfembeddings, client=self.client, by_text=False)
        bot = self.research_answerer()
        research_out =bot({"query": research_objective})
        return research_out["result"]

    def research(self, query):
        search_articles = self.search_articles(query)
        urls = self.get_urls(search_articles)
        research_content = self.get_content_from_urls(urls)
        answer = self.research_given_query(query, research_content)
        return answer
