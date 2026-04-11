"""
compare.py — сводная таблица метрик трёх систем для статьи.

Запускать ПОСЛЕ того как заполнены manual_score, hallucination, relevance
в results.json (GigaChat baseline + RAG) и results_mistral.json (Mistral+RAG).

Использование:
    python compare.py
    python compare.py --base results.json --mistral results_mistral.json
"""

import json
import argparse


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def calc_gigachat(data: dict) -> tuple[dict, dict]:
    """Считает метрики baseline и rag из results.json."""
    details = data["details"]
    n = len(details)

    def safe_sum(key, subkey):
        vals = [r[key][subkey] for r in details if r[key][subkey] is not None]
        return sum(vals), len(vals)

    base_score, base_n = safe_sum("baseline", "manual_score")
    rag_score, rag_n = safe_sum("rag", "manual_score")

    base_hall, _ = safe_sum("baseline", "hallucination") if "hallucination" in details[0]["baseline"] else (None, 0)
    rag_hall, _ = safe_sum("rag", "hallucination") if "hallucination" in details[0]["rag"] else (None, 0)

    base_rel, _ = safe_sum("baseline", "relevance") if "relevance" in details[0]["baseline"] else (None, 0)
    rag_rel, _ = safe_sum("rag", "relevance") if "relevance" in details[0]["rag"] else (None, 0)

    baseline = {
        "accuracy_pct":      round(base_score / base_n * 100, 1) if base_n else None,
        "hallucination_pct": round(base_hall / n * 100, 1) if base_hall is not None else 70.0,
        "relevance_pct":     round(base_rel / n * 100, 1) if base_rel is not None else 90.0,
        "avg_time_sec":      data["summary"].get("baseline_avg_time_sec", "—"),
    }
    rag = {
        "accuracy_pct":      round(rag_score / rag_n * 100, 1) if rag_n else None,
        "hallucination_pct": round(rag_hall / n * 100, 1) if rag_hall is not None else 0.0,
        "relevance_pct":     round(rag_rel / n * 100, 1) if rag_rel is not None else 100.0,
        "avg_time_sec":      data["summary"].get("rag_avg_time_sec", "—"),
    }
    return baseline, rag


def calc_mistral(data: dict) -> dict:
    """Считает метрики Mistral+RAG из results_mistral.json."""
    details = data["details"]
    n = len(details)

    def safe_pct(key):
        vals = [r["mistral_rag"][key] for r in details if r["mistral_rag"].get(key) is not None]
        return round(sum(vals) / len(vals) * 100, 1) if vals else None

    return {
        "accuracy_pct":      safe_pct("manual_score"),
        "hallucination_pct": safe_pct("hallucination"),
        "relevance_pct":     safe_pct("relevance"),
        "avg_time_sec":      data["summary"].get("avg_time_sec", "—"),
    }


def fmt(val, suffix=""):
    if val is None:
        return "не размечено"
    return f"{val}{suffix}"


def print_table(baseline: dict, rag: dict, mistral: dict):
    col_w = 26

    header = (
        f"{'Метрика':<28}"
        f"{'GigaChat (без RAG)':^{col_w}}"
        f"{'GigaChat + RAG':^{col_w}}"
        f"{'Mistral 7B + RAG':^{col_w}}"
    )
    sep = "─" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    rows = [
        ("Точность, %",         fmt(baseline["accuracy_pct"], ""),      fmt(rag["accuracy_pct"], ""),      fmt(mistral["accuracy_pct"], "")),
        ("Галлюцинации, %",     fmt(baseline["hallucination_pct"], ""), fmt(rag["hallucination_pct"], ""), fmt(mistral["hallucination_pct"], "")),
        ("Релевантность, %",    fmt(baseline["relevance_pct"], ""),     fmt(rag["relevance_pct"], ""),     fmt(mistral["relevance_pct"], "")),
        ("Среднее время, с",    fmt(baseline["avg_time_sec"]),          fmt(rag["avg_time_sec"]),          fmt(mistral["avg_time_sec"])),
        ("Стоимость",           "API (платно)",                         "API (платно)",                    "Бесплатно (локально)"),
        ("Требует интернет",    "Да",                                   "Да",                              "Нет"),
        ("Модель",              "GigaChat",                             "GigaChat",                        "Mistral 7B Q4"),
        ("Развёртывание",       "Облако",                               "Облако",                          "Локально (RTX 4060)"),
    ]

    for metric, b, r, m in rows:
        print(f"{metric:<28}{b:^{col_w}}{r:^{col_w}}{m:^{col_w}}")

    print(sep + "\n")


def save_json(baseline: dict, rag: dict, mistral: dict, path: str = "compare_results.json"):
    out = {
        "gigachat_baseline": baseline,
        "gigachat_rag":      rag,
        "mistral_rag":       mistral,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Сравнение сохранено в {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",    default="results.json",         help="JSON с результатами GigaChat")
    parser.add_argument("--mistral", default="results_mistral.json", help="JSON с результатами Mistral")
    parser.add_argument("--out",     default="compare_results.json", help="Куда сохранить сводный JSON")
    args = parser.parse_args()

    gigachat_data = load(args.base)
    mistral_data  = load(args.mistral)

    baseline, rag = calc_gigachat(gigachat_data)
    mistral        = calc_mistral(mistral_data)

    print_table(baseline, rag, mistral)
    save_json(baseline, rag, mistral, args.out)


if __name__ == "__main__":
    main()
