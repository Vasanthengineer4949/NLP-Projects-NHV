from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from config import *

class EduBotCreator:

    def __init__(self):
        self.prompt_temp = PROMPT_TEMPLATE
        self.input_variables = INP_VARS
        self.chain_type = CHAIN_TYPE
        self.search_kwargs = SEARCH_KWARGS
        self.embedder = EMBEDDER
        self.vector_db_path = VECTOR_DB_PATH
        self.model_ckpt = MODEL_CKPT
        self.model_type = MODEL_TYPE
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE

    def create_custom_prompt(self):
        custom_prompt_temp = PromptTemplate(template=self.prompt_temp,
                            input_variables=self.input_variables)
        return custom_prompt_temp
    
    def load_llm(self):
        llm = CTransformers(
                model = self.model_ckpt,
                model_type=self.model_type,
                max_new_tokens = self.max_new_tokens,
                temperature = self.temperature
            )
        return llm
    
    def load_vectordb(self):
        hfembeddings = HuggingFaceEmbeddings(
                            model_name=self.embedder, 
                            model_kwargs={'device': 'cpu'}
                        )

        vector_db = FAISS.load_local(self.vector_db_path, hfembeddings)
        return vector_db

    def create_bot(self, custom_prompt, vectordb, llm):
        retrieval_qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type=self.chain_type,
                                retriever=vectordb.as_retriever(search_kwargs=self.search_kwargs),
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": custom_prompt}
                            )
        return retrieval_qa_chain
    
    def create_edubot(self):
        self.custom_prompt = self.create_custom_prompt()
        self.vector_db = self.load_vectordb()
        self.llm = self.load_llm()
        self.bot = self.create_bot(self.custom_prompt, self.vector_db, self.llm)
        return self.bot