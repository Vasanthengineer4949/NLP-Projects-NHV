from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
import os
import pandas as pd
import streamlit as st
from langchain.prompts import PromptTemplate
from pandasql import sqldf
from groq import Groq


template = """You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. Dont add \n characters.

You must output the SQL query that answers the question in a single line.

### Input:
`{question}`

### Context:
`{context}`

### Response:
"""


prompt = PromptTemplate.from_template(template=template)

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

def groq_infer(prompt):
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="mixtral-8x7b-32768",
)
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

# 1. Create cache_resource - To load the model
# infer - pipeline -> pipe()
def main():
    st.set_page_config(page_title="Database Administrator", page_icon="📊", layout="wide")
    st.title("Mixtral Database Administrator ft. Groq")

    col1, col2 = st.columns([2, 3])

    with col1:
        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, encoding="latin1")
            df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "", regex=True)
            st.write("Here's a preview of your uploaded file:")
            st.dataframe(df)

            context = pd.io.sql.get_schema(df.reset_index(), "df").replace('"', "")
            st.write("SQL Schema:")
            st.code(context)

    with col2:
        if uploaded_file is not None:
            question = st.text_input("Write a question about the data", key="question")

            if st.button("Get Answer", key="get_answer"):
                if question:
                    attempt = 0
                    max_attempts = 5
                    while attempt < max_attempts:
                        try:
                            input = {"context": context, "question": question}
                            formatted_prompt = prompt.invoke(input=input).text
                            response = groq_infer(formatted_prompt)
                            final = response.replace("`", "").replace("sql", "").strip()
                            result = sqldf(final, locals())
                            st.write("Answer:")
                            st.dataframe(result)
                            break
                        except Exception as e:
                            attempt += 1
                            st.error(
                                f"Attempt {attempt}/{max_attempts} failed. Retrying..."
                            )
                            if attempt == max_attempts:
                                st.error(
                                    "Unable to get the correct query, refresh app or try again later."
                                )
                            continue

                else:
                    st.warning("Please enter a question before clicking 'Get Answer'.")


if __name__ == "__main__":
    main()