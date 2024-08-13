from langchain_core.prompts import ChatPromptTemplate

def chat_history_parser(chat_history):

    chats = []
    for chat in chat_history:
        if chat["role"] == "user":
            chats.append("User: " + chat["content"])
        else:
            chats.append("Assistant: " + chat["content"])
    
    chat_history_formatted = "\n".join(chats)
    return chat_history_formatted


def general_exec(general_model, question, chat_history):

    chat_history_formatted = chat_history_parser(chat_history)

    general_template = """You are an assistant created by Neural Hacks with Vasanth. Here are your previous conversations.
    
    {chat_history_formatted}

    Answer the question given by the user.
    User: {question}
    Assistant: """
    
    question_formatted = general_template.format(chat_history_formatted=chat_history_formatted, question=question)

    assistant_answer = general_model.invoke(question_formatted).content

    current_chats = []
    current_chats.append({"role": "user", "content": question})
    current_chats.append({"role": "assistant", "content": assistant_answer})

    chat_history.extend(current_chats)
    return chat_history

def math_exec(math_model, question, chat_history):

    chat_history_formatted = chat_history_parser(chat_history)

    math_template = """You are an assistant created by Neural Hacks with Vasanth. Here are your previous conversations.
    
    {chat_history_formatted}

    Solve this math question of the user.
    User: {question}
    Assistant: """

    question_formatted = math_template.format(chat_history_formatted=chat_history_formatted, question=question)

    assistant_answer = math_model.invoke(question_formatted)

    current_chats = []
    current_chats.append({"role": "user", "content": question})
    current_chats.append({"role": "assistant", "content": assistant_answer})

    chat_history.extend(current_chats)
    return chat_history

