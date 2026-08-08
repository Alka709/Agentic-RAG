from langchain_ollama import ChatOllama

def create_llm(model_name):
    return ChatOllama(model=model_name,temperature=0)

def generate_answer(llm,prompt,question,context):
    messages=prompt.invoke({
        "question":question,
        "context" :context
    })

    response=llm.invoke(messages)

    return response.content