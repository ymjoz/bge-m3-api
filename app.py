from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

app = FastAPI()

# 加載本地模型
model_path = "/app/models/bge-m3"
try:
    model = SentenceTransformer(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

class TextInput(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(input: TextInput):
    embedding = model.encode(input.text).tolist()
    return {"embedding": embedding}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
