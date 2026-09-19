from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_API_KEY


def create_llm(model_name: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )


def generate_answer(llm, prompt, question, context):
    messages = prompt.invoke({
        "question": question,
        "context": context
    })

    response = llm.invoke(messages)

    return response.content