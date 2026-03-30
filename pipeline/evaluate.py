import json
import time
from pipeline.retriever import get_collection
from pipeline.chain import build_chain

TEST_QUESTIONS = [
    # Академический отпуск
    "Как получить академический отпуск в ОмГТУ?",

    # Учебный процесс
    "Когда начинается сессия в ОмГТУ?",
    "Какой проходной балл на бюджет в ОмГТУ?",
    "Как оформить справку об обучении в ОмГТУ?",
    "Какие электронные библиотеки доступны студентам ОмГТУ?",
    "Чем бакалавриат отличается от специалитета в ОмГТУ?",

    # Стипендии
    "Как получить стипендию в ОмГТУ?",
    "Платят ли стипендию студентам первого курса до первой сессии в ОмГТУ?",
    "Зависит ли социальная стипендия от оценок в ОмГТУ?",
    "Есть ли стипендия для студентов военного учебного центра ОмГТУ?",
    "Может ли студент обжаловать решение стипендиальной комиссии в ОмГТУ?",

    # ПГАС
    "Что такое ПГАС и как её получить в ОмГТУ?",
    "Лишат ли меня ПГАС за пересдачу экзамена в ОмГТУ?",
    "Дают ли баллы ПГАС за платные олимпиады и онлайн-курсы в ОмГТУ?",
    "Сколько баллов даёт патент на изобретение для ПГАС в ОмГТУ?",
    "Получают ли ПГАС студенты-спортсмены с президентской стипендией в ОмГТУ?",

    # Деканаты и руководство
    "Кто ректор ОмГТУ?",
    "Где находится деканат ФИТиКС?",
    "Кто декан ФТНГ?",
    "Кто декан РТФ?",
    "Кто директор колледжа ОмГТУ?",

    # Навигация
    "Где находится корпус 8 ОмГТУ?",
    "Где находятся корпуса ОмГТУ в Городке нефтяников?",
    "Где корпус 12 ОмГТУ?",

    # Питание и инфраструктура
    "Где поесть в ОмГТУ?",
    "Где распечатать документы в ОмГТУ?",

    # Транспорт
    "Как добраться до 6 корпуса ОмГТУ?",

    # Контакты и отделы
    "Как связаться с отделом социальной защиты студентов ОмГТУ?",
    "Контакты центра карьеры ОмГТУ?",
    "Кто завкафедрой КЗИ в ОмГТУ?",

    # Организации и активности
    "Что такое СтудЛаб в Омске и как туда попасть?",
    "Выдаётся ли сертификат в Студенческой ИТ-лаборатории ОмГТУ?",
]


def ask_without_rag(llm, question: str) -> tuple[str, float]:
    from langchain_core.messages import HumanMessage
    start = time.time()
    response = llm.invoke([HumanMessage(content=question)])
    elapsed = time.time() - start
    return response.content, round(elapsed, 2)


def ask_with_rag(chain, question: str) -> tuple[str, float]:
    start = time.time()
    response = chain.invoke(question)
    elapsed = time.time() - start
    return response.content, round(elapsed, 2)


def run_evaluation(gigachat_credentials: str, output_path: str = "results.json"):
    from langchain_gigachat import GigaChat

    llm = GigaChat(
        credentials=gigachat_credentials,
        verify_ssl_certs=False,
        model="GigaChat"
    )
    collection = get_collection()
    chain = build_chain(collection, gigachat_credentials)

    results = []

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] {question}")

        ans_base, t_base = ask_without_rag(llm, question)
        ans_rag, t_rag = ask_with_rag(chain, question)

        results.append({
            "id": i + 1,
            "question": question,
            "baseline": {
                "answer": ans_base,
                "time_sec": t_base,
                "manual_score": None,  # 0 или 1 — заполнить вручную
            },
            "rag": {
                "answer": ans_rag,
                "time_sec": t_rag,
                "manual_score": None,  # 0 или 1 — заполнить вручную
            }
        })

        print(f"  Baseline ({t_base}s): {ans_base[:80]}...")
        print(f"  RAG     ({t_rag}s): {ans_rag[:80]}...")

    # Автоматические метрики скорости
    n = len(results)
    summary = {
        "total_questions": n,
        "baseline_avg_time_sec": round(sum(r["baseline"]["time_sec"] for r in results) / n, 2),
        "rag_avg_time_sec": round(sum(r["rag"]["time_sec"] for r in results) / n, 2),
        # Заполнятся после ручной разметки
        "baseline_manual_score_sum": None,
        "rag_manual_score_sum": None,
        "baseline_accuracy_pct": None,
        "rag_accuracy_pct": None,
    }

    output = {"summary": summary, "details": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== ИТОГО (автоматические метрики) ===")
    print(f"  Вопросов: {n}")
    print(f"  Baseline avg time: {summary['baseline_avg_time_sec']}s")
    print(f"  RAG avg time: {summary['rag_avg_time_sec']}s")
    print(f"\nРезультаты сохранены в {output_path}")
    print("Заполни manual_score (0 или 1) в results.json и запусти score.py")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    run_evaluation(os.getenv("GIGACHAT_CREDENTIALS"))