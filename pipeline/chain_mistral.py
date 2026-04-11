import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

SYSTEM_PROMPT = """Ты — помощник студента ОмГТУ. Отвечай на основе предоставленного контекста.
Если в контексте есть частичная информация — используй её и укажи что данные могут быть неполными.
Если контекст совсем не относится к вопросу — честно скажи что такой информации нет.
Не придумывай факты.

Контекст:
{context}"""


def ask_ollama(prompt: str, model: str = "mistral") -> str:
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.1}
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def build_chain_mistral(collection, top_k: int = 3, model: str = "mistral"):

    def retrieve(query: str) -> str:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        docs = results["documents"][0]
        return "\n\n".join(docs)

    def chain_fn(question: str) -> str:
        context = retrieve(question)
        prompt = f"{SYSTEM_PROMPT.format(context=context)}\n\nВопрос: {question}"
        return ask_ollama(prompt, model)

    return chain_fn