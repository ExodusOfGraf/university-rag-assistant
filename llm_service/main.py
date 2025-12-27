import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()

# Настройки из переменных окружения
MODEL_PATH = os.getenv("MODEL_PATH", "/models/llm/qwen2.5-7b-instruct-q3_k_m.gguf")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """Ты — справочный помощник университета. Отвечай на русском языке, вежливо и по делу.""")

print("--- ЗАГРУЗКА МОДЕЛИ НА GPU ---")
llm = Llama(
    model_path=MODEL_PATH,
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=True
)

class QueryRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(request: QueryRequest):
    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt}
        ],
        temperature=0.1,
        max_tokens=1024
    )
    return {"response": output["choices"][0]["message"]["content"]}
