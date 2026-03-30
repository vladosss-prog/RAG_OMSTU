import json
import chromadb
from chromadb.utils import embedding_functions

def load_knowledge_base(json_path: str, collection_name: str = "omstu_kb"):
    client = chromadb.PersistentClient(path="./chroma_db")
    
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    # Добавляем только если коллекция пустая
    if collection.count() == 0:
        documents = [item["question"] + " " + item["answer"] for item in data]
        metadatas = [{
            "category": item["category"],
            "category_label": item.get("category_label", ""),
            "source": item.get("source", ""),
            "question": item["question"]
        } for item in data]
        ids = [str(item["id"]) for item in data]
        
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Загружено {len(data)} чанков в ChromaDB")
    else:
        print(f"Коллекция уже содержит {collection.count()} чанков")
    
    return collection

if __name__ == "__main__":
    load_knowledge_base("data/rag_omstu.json")