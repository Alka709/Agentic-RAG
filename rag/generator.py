from langchain_google_genai import ChatGoogleGenerativeAI

from config import GEMINI_API_KEY


def create_llm(model_name: str):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        temperature=0,
    )


def _format_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", content))
    return str(content)


def generate_answer(llm, prompt, question, context):
    messages = prompt.invoke({
        "question": question,
        "context": context
    })

    response = llm.invoke(messages)

    return _format_content(response.content)