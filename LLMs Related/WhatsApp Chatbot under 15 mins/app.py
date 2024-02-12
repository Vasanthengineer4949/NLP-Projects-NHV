from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def response(question, llm):
    
    template = """Question: {question}

    Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    response = llm_chain.run(question)

    return response

