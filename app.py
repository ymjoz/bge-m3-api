from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import faiss
import numpy as np

app = FastAPI()

# 加載本地模型
model_path = "/app/models/bge-m3"
try:
    model = SentenceTransformer(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

class TextInput(BaseModel):
    text: str

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

# 設置 FAISS 索引 (L2 距離)
embedding_dim = 1024  # bge-m3 產生 1024 維的向量
index = faiss.IndexFlatL2(embedding_dim)  # L2 距離 (歐幾里得距離)
vector_store = {}  # 存放 ID -> Text 對應關係 (僅作簡單示範)


@app.post("/embed")
async def get_embedding(input: TextInput):
    embedding = model.encode(input.text).tolist()
    return {"embedding": embedding}

@app.post("/add/")
async def add_text(texts: list[str]):
    """新增文本到 FAISS 向量庫"""
    global index
    embeddings = model.encode(texts)

    start_id = len(vector_store)
    for i, text in enumerate(texts):
        vector_store[start_id + i] = text  # 儲存文本
    index.add(np.array(embeddings, dtype=np.float32))  # 新增到 FAISS

    return {"message": "Added", "count": len(texts)}

@app.post("/search/")
async def search_text(search_query: SearchQuery):
    """執行相似性搜索"""
    if index.ntotal == 0:
        return {"error": "FAISS index is empty"}

    query_embedding = model.encode([search_query.query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), search_query.top_k)

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx in vector_store:
            results.append({"text": vector_store[idx], "distance": float(distances[0][i])})

    return {"query": search_query.query, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

