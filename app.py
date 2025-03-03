from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List
import faiss
import numpy as np

app = FastAPI()

# 加載本地模型
model_path = "/app/models/bge-m3"
try:
    model = SentenceTransformer(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")


# 定義請求體格式
class TextInput(BaseModel):
    text: str

class TextListInput(BaseModel):
    texts: List[str]  # 修正 FastAPI 解析 JSON 問題

class SearchInput(BaseModel):
    query: str
    top_k: int = 5

# 設置 FAISS 索引 (L2 距離)
embedding_dim = 1024  # bge-m3 產生 1024 維的向量
index = faiss.IndexFlatL2(embedding_dim)  # L2 距離 (歐幾里得距離)
vector_store = {}  # 存放 ID -> Text 對應關係 (簡單示範)


@app.post("/embed")
async def get_embedding(input: TextInput):
    """獲取文本的嵌入向量"""
    embedding = model.encode(input.text).astype(np.float32).tolist()  # 確保是 float32
    return {"embedding": embedding}

@app.get("/index_info")
async def get_index_info():
    """獲取 FAISS 索引資訊"""
    return {"total_vectors": index.ntotal, "d": index.d}

@app.post("/add/")
async def add_text(data: TextListInput):
    """新增文本到 FAISS 向量庫"""
    global index
    texts = data.texts  # 解析 JSON
    embeddings = model.encode(texts).astype(np.float32)  # 確保是 float32

    start_id = len(vector_store)
    for i, text in enumerate(texts):
        vector_store[start_id + i] = text  # 儲存文本

    index.add(embeddings)  # 新增到 FAISS
    return {"message": "Added", "count": len(texts)}


@app.post("/search/")
async def search_text(data: SearchInput):
    """執行相似性搜索"""
    if index.ntotal == 0:
        return {"error": "FAISS index is empty"}

    query_embedding = model.encode([data.query]).astype(np.float32)  # 確保是 float32
    distances, indices = index.search(query_embedding, data.top_k)

    results = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        if idx in vector_store:
            results.append({"text": vector_store[idx], "distance": float(distances[0][i])})

    return {"query": data.query, "results": results}




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

