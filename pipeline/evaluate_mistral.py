import json
import time
from pipeline.retriever import get_collection
from pipeline.chain_mistral import build_chain_mistral

# Те же вопросы что в evaluate.py — не трогать порядок, id должны совпадать
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

    # Косвенные вопросы
    "Можно ли взять перерыв в учёбе по состоянию здоровья в ОмГТУ?",        # → академический отпуск
    "Получу ли я деньги от университета если сдам сессию на отлично?",        # → стипендия
    "Где в ОмГТУ можно перекусить между парами?",                             # → питание
    "Как мне найти деканат моего факультета?",                                # → навигация
    "Влияет ли участие в олимпиадах на повышенную стипендию?",               # → ПГАС
    "Кто руководит университетом?",                                           # → ректор
    "Есть ли в ОмГТУ место где студенты занимаются IT-проектами?",           # → СтудЛаб
    "Могу ли я оспорить решение комиссии по стипендии?",                      # → обжалование
    "Где взять официальную бумагу что я учусь в ОмГТУ?",                     # → справка
    "Как мне добраться до корпуса где учат нефтяников?"
]


def ask_with_rag(chain, question: str) -> tuple[str, float]:
    start = time.time()
    response = chain(question)  # теперь chain это просто функция
    elapsed = time.time() - start
    return response, round(elapsed, 2)

def run_evaluation_mistral(output_path: str = "results/results_mistral.json"):
    collection = get_collection()
    chain = build_chain_mistral(collection)

    results = []

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/{len(TEST_QUESTIONS)}] {question}")

        ans_rag, t_rag = ask_with_rag(chain, question)

        results.append({
            "id": i + 1,
            "question": question,
            "mistral_rag": {
                "answer": ans_rag,
                "time_sec": t_rag,
                "manual_score": None,   # 0 или 1 — заполнить вручную (Андрей)
                "hallucination": None,  # 0 или 1
                "relevance": None,      # 0 или 1
            }
        })

        print(f"  Mistral+RAG ({t_rag}s): {ans_rag[:120]}...")

    n = len(results)
    times = [r["mistral_rag"]["time_sec"] for r in results]
    summary = {
        "model": "mistral",
        "total_questions": n,
        "avg_time_sec": round(sum(times) / n, 2),
        "min_time_sec": round(min(times), 2),
        "max_time_sec": round(max(times), 2),
        # Заполнятся после ручной разметки:
        "manual_score_sum": None,
        "accuracy_pct": None,
        "hallucination_count": None,
        "hallucination_pct": None,
        "relevance_pct": None,
    }

    output = {"summary": summary, "details": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== ИТОГО (Mistral + RAG) ===")
    print(f"  Вопросов:      {n}")
    print(f"  Avg time:      {summary['avg_time_sec']}s")
    print(f"  Min/Max time:  {summary['min_time_sec']}s / {summary['max_time_sec']}s")
    print(f"\nРезультаты сохранены в {output_path}")
    print("Заполни manual_score, hallucination, relevance и запусти compare.py")


if __name__ == "__main__":
    run_evaluation_mistral()
