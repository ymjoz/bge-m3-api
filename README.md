# Test BGE-M3 embedding


```sh
cd ~/dev/bge-m3-api
docker build -t tom/bge-m3-api:1.0.1 .
docker build -t tom/bge-m3-faiss:1.0 .
```


```sh
docker run -d --gpus device=0 \
	-p 11436:8000 \
	--name bge-m3-tom \
	tom/bge-m3-api:1.0.1
```

```sh
curl -X POST "http://localhost:11436/embed" \
	-H "Content-Type: application/json" \ 
	-d '{"text": "這是一個測試句子"}'
```
