"""
Скрипт загрузки данных в ChromaDB.
Запускать из корня проекта: python scripts/load_data.py

В будущем замени JSON файлы на реальные данные:
- PDF парсинг: PyPDF2 или pdfplumber
- Excel: pandas.read_excel()
- Парсинг сайта: requests + BeautifulSoup
"""

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Пути
DATA_DIR = "data"
CHROMA_PATH = "chroma_db"
EMBED_MODEL_PATH = "models/embed/sbert_large_nlu_ru"

print("Загрузка модели эмбеддингов...")
embed_model = SentenceTransformer(EMBED_MODEL_PATH)

print("Инициализация ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Удаляем старые коллекции если есть
for name in ["schedules", "departments", "general_info"]:
    try:
        client.delete_collection(name)
    except:
        pass

def load_schedules():
    """Загрузка расписаний"""
    collection = client.create_collection(
        name="schedules",
        metadata={"hnsw:space": "cosine"}
    )
    
    schedules_dir = os.path.join(DATA_DIR, "schedules")
    documents = []
    metadatas = []
    ids = []
    
    for filename in os.listdir(schedules_dir):
        if filename.endswith(".json"):
            with open(os.path.join(schedules_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
            
            group = data["group"]
            faculty = data["faculty"]
            course = data["course"]
            session = data["session"]
            
            for i, exam in enumerate(data["exams"]):
                # Формируем текст для поиска
                doc_text = (
                    f"Расписание экзамена для группы {group}. "
                    f"Предмет: {exam['subject']}. "
                    f"Дата: {exam['date']}, время: {exam['time']}. "
                    f"Аудитория: {exam['room']}, {exam['building']}. "
                    f"Преподаватель: {exam['teacher']}."
                )
                
                documents.append(doc_text)
                metadatas.append({
                    "type": "schedule",
                    "group": group,
                    "faculty": faculty,
                    "course": str(course),
                    "subject": exam["subject"],
                    "date": exam["date"]
                })
                ids.append(f"schedule_{group}_{i}")
    
    if documents:
        embeddings = embed_model.encode(documents).tolist()
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Загружено {len(documents)} записей расписания")

def load_departments():
    """Загрузка информации о кафедрах"""
    collection = client.create_collection(
        name="departments",
        metadata={"hnsw:space": "cosine"}
    )
    
    with open(os.path.join(DATA_DIR, "departments", "departments.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    metadatas = []
    ids = []
    
    for i, dept in enumerate(data["departments"]):
        doc_text = (
            f"Кафедра: {dept['name']} ({dept['short_name']}). "
            f"Факультет: {dept['faculty']}. "
            f"Расположение: {dept['building']}, кабинет {dept['room']}. "
            f"Телефон: {dept['phone']}, email: {dept['email']}. "
            f"Заведующий: {dept['head']}, {dept['head_title']}. "
            f"Часы работы: {dept['work_hours']}."
        )
        
        documents.append(doc_text)
        metadatas.append({
            "type": "department",
            "name": dept["name"],
            "short_name": dept["short_name"],
            "faculty": dept["faculty"]
        })
        ids.append(f"dept_{i}")
    
    if documents:
        embeddings = embed_model.encode(documents).tolist()
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Загружено {len(documents)} кафедр")

def load_general_info():
    """Загрузка общей информации"""
    collection = client.create_collection(
        name="general_info",
        metadata={"hnsw:space": "cosine"}
    )
    
    with open(os.path.join(DATA_DIR, "info", "general_info.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    metadatas = []
    ids = []
    
    # Деканат
    deanery = data["deanery"]
    doc_text = (
        f"Деканат {deanery['faculty']}. "
        f"Расположение: {deanery['building']}, кабинет {deanery['room']}. "
        f"Телефон: {deanery['phone']}, email: {deanery['email']}. "
        f"Часы работы: {deanery['work_hours']}. "
        f"Услуги: {', '.join(deanery['services'])}."
    )
    documents.append(doc_text)
    metadatas.append({"type": "deanery", "faculty": deanery["faculty"]})
    ids.append("deanery_fit")
    
    # Библиотека
    library = data["library"]
    doc_text = (
        f"Библиотека университета. "
        f"Расположение: {library['building']}, {library['room']}. "
        f"Телефон: {library['phone']}. "
        f"Часы работы: {library['work_hours']}. "
        f"Услуги: {', '.join(library['services'])}."
    )
    documents.append(doc_text)
    metadatas.append({"type": "library"})
    ids.append("library")
    
    # FAQ
    for i, faq in enumerate(data["faq"]):
        doc_text = f"Вопрос: {faq['question']} Ответ: {faq['answer']}"
        documents.append(doc_text)
        metadatas.append({"type": "faq"})
        ids.append(f"faq_{i}")
    
    if documents:
        embeddings = embed_model.encode(documents).tolist()
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Загружено {len(documents)} записей общей информации")

if __name__ == "__main__":
    print("=" * 50)
    print("Загрузка данных в ChromaDB")
    print("=" * 50)
    
    load_schedules()
    load_departments()
    load_general_info()
    
    print("=" * 50)
    print("Загрузка завершена!")
    print("=" * 50)
