import json

with open("results.json", encoding="utf-8") as f:
    data = json.load(f)

details = data["details"]
n = len(details)

fields = ["manual_score", "hallucination", "relevance"]

# Проверка что всё заполнено
for field in fields:
    base = [r["baseline"].get(field) for r in details]
    rag = [r["rag"].get(field) for r in details]
    if None in base or None in rag:
        print(f"Не заполнено поле '{field}'!")
        exit()

base_acc = sum(r["baseline"]["manual_score"] for r in details)
rag_acc = sum(r["rag"]["manual_score"] for r in details)

base_hall = sum(r["baseline"]["hallucination"] for r in details)
rag_hall = sum(r["rag"]["hallucination"] for r in details)

base_rel = sum(r["baseline"]["relevance"] for r in details)
rag_rel = sum(r["rag"]["relevance"] for r in details)

data["summary"]["baseline_accuracy_pct"] = round(base_acc / n * 100, 1)
data["summary"]["rag_accuracy_pct"] = round(rag_acc / n * 100, 1)
data["summary"]["baseline_hallucinations"] = base_hall
data["summary"]["rag_hallucinations"] = rag_hall
data["summary"]["baseline_relevance_pct"] = round(base_rel / n * 100, 1)
data["summary"]["rag_relevance_pct"] = round(rag_rel / n * 100, 1)

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("=== ИТОГОВЫЕ МЕТРИКИ ===")
print(f"Вопросов: {n}")
print(f"Точность:      Baseline {data['summary']['baseline_accuracy_pct']}% | RAG {data['summary']['rag_accuracy_pct']}%")
print(f"Галлюцинации:  Baseline {base_hall} | RAG {rag_hall}")
print(f"Релевантность: Baseline {data['summary']['baseline_relevance_pct']}% | RAG {data['summary']['rag_relevance_pct']}%")
print(f"Время (avg):   Baseline {data['summary']['baseline_avg_time_sec']}s | RAG {data['summary']['rag_avg_time_sec']}s")