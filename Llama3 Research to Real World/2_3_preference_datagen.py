import json
from datasets import load_dataset
import groq
from rich import print
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQQ_API_KEY")

# Load a real dataset from Hugging Face
dataset = load_dataset("squad_v2", split="train")

# Convert dataset to list of dictionaries
json_data = dataset.select(range(100)).to_dict()

print("Dataset Rows:", len(json_data['id']))
# print({key: json_data[key][0] for key in json_data})

# Function to format the input for the model
def format_input(context, question):
    return (
        "### Context:\n" + context +
        ("\n\n### Question:\n" + question if question else "")
    )

# Initialize the Ollama client
client = groq.Groq()  # Replace with your actual API key if required

# Initialize new keys in the json_data dictionary
json_data['rejected'] = [''] * len(json_data['id'])
json_data['chosen'] = [''] * len(json_data['id'])

# Process each entry in the dataset
for i in range(len(json_data['id'])):
    context = json_data['context'][i]
    question = json_data['question'][i]
    answer = json_data['answers'][i]['text'][0] if json_data['answers'][i]['text'] else "No answer"

    print("Rejected Answer:", answer)

    prompt_text = format_input(context, question)

    prompt = (
        f"Rewrite `{prompt_text}` output to be concise and clear: {answer}. "
        "Ensure the response is easy to understand, professional and as a full sentense. Just respond only with the Answer"
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.1-8b-instant",
    )

    response = chat_completion.choices[0].message.content
    chosen_answer = response
    print("Chosen Answer:", chosen_answer)

    json_data['rejected'][i] = answer
    json_data['chosen'][i] = chosen_answer

# Convert back to dictionary format expected by json.dump
final_data = [{key: json_data[key][i] for key in json_data} for i in range(len(json_data['id']))]

with open("preference_dataset.json", "w") as file:
    json.dump(final_data, file, indent=4)

new_data = load_dataset("json", data_files=["preference_dataset.json"])
new_data.push_to_hub("Vasanth/squad-demo-preference-dataset")