from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import argparse
import uvicorn

app = FastAPI()

class EmbeddingRequest(BaseModel):
    texts: list[str]

class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]

class KoSBERTModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def get_embeddings(self, request: EmbeddingRequest):
        embeddings = self.model.encode(request.texts)
        return {"embeddings": embeddings.tolist()}
    
    @app.get("/v1/health")
    async def health(self):
        return {"status": "ok"}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()

    model = KoSBERTModel(model_name=args.model_name)
    uvicorn.run(app, host=args.host, port=args.port)
