DATA_DIR_PATH = "data/"
VECTOR_DB_PATH = "faiss/education"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 200
EMBEDDER = "thenlper/gte-large"
DEVICE = "cpu"
PROMPT_TEMPLATE = '''
With the information provided try to answer the question. 
If you cant answer the question based on the information either say you cant find an answer or unable to find an answer.
So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

Context: {context}
Question: {question}
Do provide only helpful answers

Helpful answer:
'''
INP_VARS = ['context', 'question']
CHAIN_TYPE = "stuff"
SEARCH_KWARGS = {'k': 2}
MODEL_CKPT = "res/llama-2-7b-chat.ggmlv3.q4_1.bin"
MODEL_TYPE = "llama"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.9
