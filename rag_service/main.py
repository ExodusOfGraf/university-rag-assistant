"""
RAG API Service — универсальный бэкенд для справочной системы.
Может использоваться с любым фронтендом: Telegram, Web, Mobile.
"""

import os
import json
import requests
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

# --- НАСТРОЙКИ ---
LLM_API_URL = os.getenv("LLM_API_URL", "http://llm_service:8000/generate")
EMBED_MODEL_PATH = os.getenv("EMBED_MODEL_PATH", "/models/embed/sbert_large_nlu_ru")
CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/chroma_db")
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
API_KEY = os.getenv("RAG_API_KEY", "")  # Опциональная защита API

app = FastAPI(
    title="University RAG API",
    description="API для интеллектуальной справочной системы университета",
    version="1.0.0"
)

# CORS для веб-клиентов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- МОДЕЛИ ДАННЫХ ---
class ChatRequest(BaseModel):
    message: str
    user_id: str
    group: Optional[str] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[dict] = []
    intent: str

class SearchRequest(BaseModel):
    query: str
    collection: Optional[str] = None  # schedules, departments, general_info
    filters: Optional[dict] = None
    limit: int = 5

class SearchResponse(BaseModel):
    results: List[dict]
    total: int

class UserCreate(BaseModel):
    user_id: str
    username: Optional[str] = None
    platform: str = "unknown"  # telegram, web, mobile

class UserResponse(BaseModel):
    user_id: str
    username: Optional[str]
    group: Optional[str]
    is_blocked: bool
    warnings: int

class GroupsResponse(BaseModel):
    groups: List[str]

class ModerationReport(BaseModel):
    user_id: str
    message: str
    violation_type: Optional[str] = None

# --- ИНИЦИАЛИЗАЦИЯ ---
print("Загрузка модели эмбеддингов...")
embed_model = SentenceTransformer(EMBED_MODEL_PATH)

print("Инициализация ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_or_create_collection(name):
    try:
        return chroma_client.get_collection(name)
    except:
        return chroma_client.create_collection(name, metadata={"hnsw:space": "cosine"})

collections = {
    "schedules": get_or_create_collection("schedules"),
    "departments": get_or_create_collection("departments"),
    "general_info": get_or_create_collection("general_info")
}

# Простое хранилище пользователей (в проде заменить на БД)
users_db: dict = {}
user_groups: dict = {}

# --- ЗАГРУЗКА ДАННЫХ ---
def load_initial_data():
    total_docs = sum(c.count() for c in collections.values())
    if total_docs > 0:
        print(f"База содержит {total_docs} документов")
        return
    
    print("Загрузка данных...")
    
    # Расписания
    schedules_dir = os.path.join(DATA_DIR, "schedules")
    if os.path.exists(schedules_dir):
        docs, metas, ids = [], [], []
        for fn in os.listdir(schedules_dir):
            if fn.endswith(".json"):
                with open(os.path.join(schedules_dir, fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
                group = data["group"]
                for i, exam in enumerate(data["exams"]):
                    doc = f"Расписание группы {group}. {exam['subject']}: {exam['date']} {exam['time']}, ауд. {exam['room']} ({exam['building']}), преп. {exam['teacher']}."
                    docs.append(doc)
                    metas.append({"type": "schedule", "group": group, "subject": exam["subject"]})
                    ids.append(f"sched_{group}_{i}")
        if docs:
            embs = embed_model.encode(docs).tolist()
            collections["schedules"].add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
            print(f"Загружено {len(docs)} расписаний")
    
    # Кафедры
    dept_file = os.path.join(DATA_DIR, "departments", "departments.json")
    if os.path.exists(dept_file):
        with open(dept_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs, metas, ids = [], [], []
        for i, d in enumerate(data["departments"]):
            doc = f"Кафедра {d['name']} ({d['short_name']}). {d['building']}, каб. {d['room']}. Тел: {d['phone']}. Зав: {d['head']}."
            docs.append(doc)
            metas.append({"type": "department", "short_name": d["short_name"]})
            ids.append(f"dept_{i}")
        if docs:
            embs = embed_model.encode(docs).tolist()
            collections["departments"].add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
            print(f"Загружено {len(docs)} кафедр")
    
    # Общая информация
    info_file = os.path.join(DATA_DIR, "info", "general_info.json")
    if os.path.exists(info_file):
        with open(info_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs, metas, ids = [], [], []
        
        d = data["deanery"]
        docs.append(f"Деканат {d['faculty']}. {d['building']}, каб. {d['room']}. Тел: {d['phone']}. Часы: {d['work_hours']}.")
        metas.append({"type": "deanery"})
        ids.append("deanery")
        
        lib = data["library"]
        docs.append(f"Библиотека. {lib['building']}, {lib['room']}. Часы: {lib['work_hours']}.")
        metas.append({"type": "library"})
        ids.append("library")
        
        for i, faq in enumerate(data["faq"]):
            docs.append(f"Q: {faq['question']} A: {faq['answer']}")
            metas.append({"type": "faq"})
            ids.append(f"faq_{i}")
        
        if docs:
            embs = embed_model.encode(docs).tolist()
            collections["general_info"].add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
            print(f"Загружено {len(docs)} общей информации")
    
    print("Загрузка завершена!")

load_initial_data()

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def verify_api_key(x_api_key: str = Header(None)):
    """Проверка API ключа (опционально)"""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

def query_llm(prompt: str) -> str:
    try:
        resp = requests.post(LLM_API_URL, json={"prompt": prompt}, timeout=120)
        return resp.json().get("response", "Ошибка генерации")
    except Exception as e:
        return f"Ошибка LLM: {e}"

def search_documents(query: str, collection_name: str = None, filters: dict = None, limit: int = 5) -> list:
    query_emb = embed_model.encode([query]).tolist()
    results = []
    
    target_collections = [collection_name] if collection_name else collections.keys()
    
    for name in target_collections:
        if name not in collections:
            continue
        coll = collections[name]
        if coll.count() == 0:
            continue
        
        where = filters if filters else None
        res = coll.query(query_embeddings=query_emb, n_results=limit, where=where)
        
        if res["documents"] and res["documents"][0]:
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
                results.append({
                    "text": doc,
                    "metadata": meta,
                    "collection": name,
                    "score": 1 - dist  # Конвертируем расстояние в score
                })
    
    # Сортируем по релевантности
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:limit]

def detect_intent(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in ["расписание", "экзамен", "сессия", "когда", "зачет"]):
        return "schedule"
    elif any(kw in q for kw in ["кафедра", "преподаватель", "деканат"]):
        return "department"
    return "general"

def check_moderation(text: str) -> Optional[str]:
    """Проверка на нарушения"""
    import re
    text_lower = text.lower()
    
    profanity = [r'\b(бля|хуй|пизд|ебать|сука|мудак)\b']
    aggression = [r'\b(убью|сдохни|урою|ненавижу)\b']
    
    for p in profanity:
        if re.search(p, text_lower):
            return "profanity"
    for p in aggression:
        if re.search(p, text_lower):
            return "aggression"
    return None

# --- API ENDPOINTS ---

@app.get("/")
async def root():
    return {"service": "University RAG API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "collections": {name: coll.count() for name, coll in collections.items()},
        "llm_service": LLM_API_URL
    }

# --- CHAT ENDPOINT (главный) ---

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Основной endpoint для чата.
    Принимает сообщение, ищет контекст в RAG, генерирует ответ через LLM.
    """
    # Проверка блокировки
    user = users_db.get(req.user_id, {})
    if user.get("is_blocked"):
        raise HTTPException(status_code=403, detail="User is blocked")
    
    # Проверка модерации
    violation = check_moderation(req.message)
    if violation:
        # Увеличиваем счётчик предупреждений
        if req.user_id not in users_db:
            users_db[req.user_id] = {"warnings": 0, "is_blocked": False}
        users_db[req.user_id]["warnings"] = users_db[req.user_id].get("warnings", 0) + 1
        
        if users_db[req.user_id]["warnings"] >= 3:
            users_db[req.user_id]["is_blocked"] = True
            raise HTTPException(status_code=403, detail="User blocked for violations")
        
        raise HTTPException(
            status_code=400, 
            detail=f"Message violates rules. Warnings: {users_db[req.user_id]['warnings']}/3"
        )
    
    intent = detect_intent(req.message)
    group = req.group or user_groups.get(req.user_id)
    
    # Поиск контекста
    filters = {"group": group} if intent == "schedule" and group else None
    collection = "schedules" if intent == "schedule" else None
    
    results = search_documents(req.message, collection_name=collection, filters=filters, limit=3)
    
    # Формируем промпт
    if results:
        context = "\n".join([r["text"] for r in results])
        prompt = f"Контекст:\n{context}\n\nВопрос: {req.message}\n\nОтветь на основе контекста."
    else:
        prompt = f"Вопрос: {req.message}\n\nВ базе нет информации. Дай общий совет."
    
    response = query_llm(prompt)
    
    return ChatResponse(
        response=response,
        sources=[{"text": r["text"][:100], "collection": r["collection"]} for r in results],
        intent=intent
    )

# --- SEARCH ENDPOINT ---

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Поиск по базе знаний без генерации ответа.
    Полезно для отображения списка результатов.
    """
    results = search_documents(
        req.query, 
        collection_name=req.collection, 
        filters=req.filters, 
        limit=req.limit
    )
    
    return SearchResponse(
        results=results,
        total=len(results)
    )

# --- USER MANAGEMENT ---

@app.post("/users", response_model=UserResponse)
async def create_or_get_user(req: UserCreate):
    """Создать или получить пользователя"""
    if req.user_id not in users_db:
        users_db[req.user_id] = {
            "username": req.username,
            "platform": req.platform,
            "is_blocked": False,
            "warnings": 0,
            "created_at": datetime.now().isoformat()
        }
    
    user = users_db[req.user_id]
    return UserResponse(
        user_id=req.user_id,
        username=user.get("username"),
        group=user_groups.get(req.user_id),
        is_blocked=user.get("is_blocked", False),
        warnings=user.get("warnings", 0)
    )

@app.put("/users/{user_id}/group")
async def set_user_group(user_id: str, group: str):
    """Установить группу пользователя"""
    user_groups[user_id] = group
    return {"user_id": user_id, "group": group}

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Получить информацию о пользователе"""
    user = users_db.get(user_id, {})
    return UserResponse(
        user_id=user_id,
        username=user.get("username"),
        group=user_groups.get(user_id),
        is_blocked=user.get("is_blocked", False),
        warnings=user.get("warnings", 0)
    )

# --- GROUPS ---

@app.get("/groups", response_model=GroupsResponse)
async def get_groups():
    """Получить список доступных групп"""
    groups = set()
    coll = collections["schedules"]
    if coll.count() > 0:
        results = coll.get()
        for meta in results["metadatas"]:
            if "group" in meta:
                groups.add(meta["group"])
    return GroupsResponse(groups=sorted(list(groups)))

# --- ADMIN ENDPOINTS ---

@app.get("/admin/stats")
async def admin_stats(x_api_key: str = Header(None)):
    """Статистика системы (требует API ключ)"""
    verify_api_key(x_api_key)
    
    blocked_count = sum(1 for u in users_db.values() if u.get("is_blocked"))
    
    return {
        "total_users": len(users_db),
        "blocked_users": blocked_count,
        "collections": {name: coll.count() for name, coll in collections.items()},
        "users_with_warnings": sum(1 for u in users_db.values() if u.get("warnings", 0) > 0)
    }

@app.get("/admin/users")
async def admin_list_users(x_api_key: str = Header(None)):
    """Список пользователей"""
    verify_api_key(x_api_key)
    return {
        "users": [
            {
                "user_id": uid,
                "username": u.get("username"),
                "group": user_groups.get(uid),
                "is_blocked": u.get("is_blocked", False),
                "warnings": u.get("warnings", 0),
                "platform": u.get("platform")
            }
            for uid, u in users_db.items()
        ]
    }

@app.post("/admin/users/{user_id}/block")
async def admin_block_user(user_id: str, reason: str = "Admin action", x_api_key: str = Header(None)):
    """Заблокировать пользователя"""
    verify_api_key(x_api_key)
    if user_id not in users_db:
        users_db[user_id] = {}
    users_db[user_id]["is_blocked"] = True
    users_db[user_id]["block_reason"] = reason
    return {"user_id": user_id, "is_blocked": True, "reason": reason}

@app.post("/admin/users/{user_id}/unblock")
async def admin_unblock_user(user_id: str, x_api_key: str = Header(None)):
    """Разблокировать пользователя"""
    verify_api_key(x_api_key)
    if user_id in users_db:
        users_db[user_id]["is_blocked"] = False
        users_db[user_id]["warnings"] = 0
    return {"user_id": user_id, "is_blocked": False}

# --- DATA MANAGEMENT ---

@app.post("/admin/reload")
async def admin_reload_data(x_api_key: str = Header(None)):
    """Перезагрузить данные из файлов"""
    verify_api_key(x_api_key)
    
    # Очищаем коллекции
    for name in collections:
        try:
            chroma_client.delete_collection(name)
        except:
            pass
        collections[name] = chroma_client.create_collection(name, metadata={"hnsw:space": "cosine"})
    
    load_initial_data()
    
    return {
        "status": "reloaded",
        "collections": {name: coll.count() for name, coll in collections.items()}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
