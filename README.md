# Test BGE-M3 embedding

### build docker image
```sh
cd ~/dev/bge-m3-api
docker build -t tom/bge-m3-api:1.0.1 .
docker build -t tom/bge-m3-faiss:1.2 .
```

### 容器化執行
```sh
docker run -d --gpus device=0 \
	-p 11437:8000 \
	--name bge-m3-tom-v1 \
	-v /home/B30098/llm/models/bge-m3:/app/models/bge-m3 \
	-v /home/B30098/dev/bge-m3-api/vector_data:/app/vector_data \
	tom/bge-m3-faiss:1.5
```

### 測試/add API
```sh
curl -X POST "http://localhost:11437/add/" \
     -H "Content-Type: application/json" \
     -d '{"texts":["這是一個測試句子。", "機器學習是人工智慧的一部分。", "自然語言處理是一個熱門領域。"]}'
```
### 應回傳
```sh
{"message":"Added","count":3}
```

### 測試/search API
```sh
curl -X POST "http://localhost:11437/search/" \
     -H "Content-Type: application/json" \
     -d '{"query": "機器學習", "top_k": 5}'
```

### 測試/index_info API,回傳 FAISS 內部的向量數據庫大小
```sh
curl -X GET "http://localhost:11437/index_info"
```