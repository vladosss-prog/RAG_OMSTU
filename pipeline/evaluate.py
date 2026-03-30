import json
import time
from pipeline.retriever import get_collection, retrieve
from pipeline.chain import build_chain

TEST_QUESTIONS = [
    # Академический отпуск
    "Как получить академический отпуск?",

    # Учебный процесс
    "Когда начинается сессия?",
    "Какой проходной балл на бюджет?",
    "Как оформить справку об обучении?",
    "Какие электронные библиотеки доступны студентам ОмГТУ?",
    "Чем бакалавриат отличается от специалитета?",

    # Стипендии
    "Как получить стипендию?",
    "Платят ли стипендию студентам первого курса до первой сессии?",
    "Зависит ли социальная стипендия от оценок?",
    "Есть ли стипендия для студентов военного учебного центра ОмГТУ?",
    "Может ли студент обжаловать решение стипендиальной комиссии?",
    "Получат ли студенты, чьи родители служат в зоне СВО, материальную помощь?",

    # ПГАС
    "Что такое ПГАС и как её получить?",
    "Лишат ли меня ПГАС за пересдачу экзамена?",
    "Дают ли баллы ПГАС за платные олимпиады и онлайн-курсы?",
    "Сколько баллов даёт патент на изобретение для ПГАС?",
    "Получают ли ПГАС студенты-спортсмены с президентской стипендией?",

    # Деканаты и руководство
    "Кто ректор ОмГТУ?",
    "Где находится деканат ФИТиКС?",
    "Кто декан ФТНГ?",
    "Кто декан РТФ?",
    "Кто директор колледжа ОмГТУ?",

    # Навигация
    "Где находится корпус 8 ОмГТУ?",
    "Где находятся корпуса в Городке нефтяников?",
    "Где корпус 12 ОмГТУ?",

    # Питание и инфраструктура
    "Где поесть в университете?",
    "Где распечатать документы в ОмГТУ?",

    # Транспорт
    "Как добраться до 6 корпуса?",

    # Контакты и отделы
    "Как связаться с отделом социальной защиты студентов ОмГТУ?",
    "Контакты центра карьеры ОмГТУ?",
    "Кто завкафедрой КЗИ в ОмГТУ?",

    # Организации и активности
    "Что такое СтудЛаб и как туда попасть?",
    "Выдаётся ли сертификат в Студенческой ИТ-лаборатории?",
]


def ask_without_rag(llm, question: str) -> tuple[str, float]:
    from langchain.schema import HumanMessage
    start = time.time()
    response = llm.invoke([HumanMessage(content=question)])
    elapsed = time.time() - start
    return response.content, round(elapsed, 2)


def ask_with_rag(chain, question: str) -> tuple[str, float]:
    start = time.time()
    response = chain.invoke(question)
    elapsed = time.time() - start
    return response.content, round(elapsed, 2)


def score_answer(answer: str, question: str) -> dict:
    # question зарезервирован для LLM-as-judge (пока не используется)
    answer_len = len(answer)
    has_content = answer_len > 50
    not_refused = not any(p in answer.lower() for p in [
        "не знаю", "нет информации", "не могу ответить", "уточните"
    ])
    return {
        "length": answer_len,
        "has_content": int(has_content),
        "not_refused": int(not_refused),
    }


def run_evaluation(gigachat_credentials: str, output_path: str = "results.json"):
    from langchain_community.chat_models.gigachat import GigaChat

    llm = GigaChat(
        credentials=gigachat_credentials,
        verify_ssl_certs=False,
        model="GigaChat"
    )
    collection = get_collection()
    chain = build_chain(collection, gigachat_credentials)

    results = []

    for question in TEST_QUESTIONS:
        print(f"\nВопрос: {question}")

        ans_base, t_base = ask_without_rag(llm, question)
        ans_rag, t_rag = ask_with_rag(chain, question)

        score_base = score_answer(ans_base, question)
        score_rag = score_answer(ans_rag, question)

        results.append({
            "question": question,
            "baseline": {
                "answer": ans_base,
                "time_sec": t_base,
                **score_base
            },
            "rag": {
                "answer": ans_rag,
                "time_sec": t_rag,
                **score_rag
            }
        })

        print(f"  Baseline ({t_base}s): {ans_base[:80]}...")
        print(f"  RAG     ({t_rag}s): {ans_rag[:80]}...")

    n = len(results)
    summary = {
        "total_questions": n,
        "baseline_avg_time": round(sum(r["baseline"]["time_sec"] for r in results) / n, 2),
        "rag_avg_time": round(sum(r["rag"]["time_sec"] for r in results) / n, 2),
        "baseline_has_content": sum(r["baseline"]["has_content"] for r in results),
        "rag_has_content": sum(r["rag"]["has_content"] for r in results),
        "baseline_not_refused": sum(r["baseline"]["not_refused"] for r in results),
        "rag_not_refused": sum(r["rag"]["not_refused"] for r in results),
    }

    output = {"summary": summary, "details": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== ИТОГО ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nРезультаты сохранены в {output_path}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    run_evaluation(os.getenv("GIGACHAT_CREDENTIALS"))