from langchain_gigachat import GigaChat
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

SYSTEM_PROMPT = """Ты — помощник студента ОмГТУ. Отвечай строго на основе предоставленного контекста.
Если ответа в контексте нет — так и скажи, не придумывай.

Контекст:
{context}"""

def build_chain(collection, gigachat_credentials: str, top_k: int = 3):
    llm = GigaChat(
        credentials=gigachat_credentials,
        verify_ssl_certs=False,
        model="GigaChat"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])

    def retrieve(query: str) -> str:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        docs = results["documents"][0]
        return "\n\n".join(docs)

    chain = (
        {"context": RunnableLambda(retrieve), "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain

if __name__ == "__main__":
    collection = get_collection()
    chain = build_chain(collection, os.getenv("GIGACHAT_CREDENTIALS"))
    print(chain.invoke("Что такое академический отпуск?"))