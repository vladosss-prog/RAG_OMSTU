import chromadb
from chromadb.utils import embedding_functions


def get_collection(collection_name: str = "omstu_kb"):
    client = chromadb.PersistentClient(path="./chroma_db")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    return client.get_collection(name=collection_name, embedding_function=emb_fn)


def retrieve(collection, query: str, top_k: int = 3) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    hits = []
    for i in range(len(results["documents"][0])):
        hits.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    return hits

if __name__ == "__main__":
    collection = get_collection()
    print(retrieve(collection, "Что такое академический отпуск?"))