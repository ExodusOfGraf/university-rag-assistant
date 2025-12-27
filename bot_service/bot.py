import os
import json
import asyncio
import requests
import chromadb
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
)
from sentence_transformers import SentenceTransformer

from moderation import moderation, ViolationType

# --- НАСТРОЙКИ ---
LLM_API_URL = "http://llm_service:8000/generate"
EMBED_MODEL_PATH = "/models/embed/sbert_large_nlu_ru"
CHROMA_PATH = "/app/chroma_db"
DATA_DIR = "/app/data"

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "").lower()
AVAILABLE_GROUPS = ["ИВТ-21", "ПИ-22"]

# --- СОСТОЯНИЯ ---
class DialogStates(StatesGroup):
    in_chat = State()              # В диалоге с LLM
    waiting_for_group = State()    # Выбор группы
    waiting_block_reason = State() # Причина блокировки
    waiting_support_msg = State()  # Сообщение в поддержку

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

# --- АВТОЗАГРУЗКА ДАННЫХ ---
def load_initial_data():
    """Загрузка тестовых данных если база пустая"""
    total_docs = sum(c.count() for c in collections.values())
    if total_docs > 0:
        print(f"База уже содержит {total_docs} документов")
        return

    print("База пустая, загружаем данные...")

    # Загрузка расписаний
    schedules_dir = os.path.join(DATA_DIR, "schedules")
    if os.path.exists(schedules_dir):
        documents, metadatas, ids = [], [], []
        for filename in os.listdir(schedules_dir):
            if filename.endswith(".json"):
                with open(os.path.join(schedules_dir, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                group = data["group"]
                for i, exam in enumerate(data["exams"]):
                    doc_text = (
                        f"Расписание экзамена для группы {group}. "
                        f"Предмет: {exam['subject']}. Дата: {exam['date']}, время: {exam['time']}. "
                        f"Аудитория: {exam['room']}, {exam['building']}. Преподаватель: {exam['teacher']}."
                    )
                    documents.append(doc_text)
                    metadatas.append({"type": "schedule", "group": group, "subject": exam["subject"]})
                    ids.append(f"schedule_{group}_{i}")
        if documents:
            embeddings = embed_model.encode(documents).tolist()
            collections["schedules"].add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
            print(f"Загружено {len(documents)} записей расписания")

    # Загрузка кафедр
    dept_file = os.path.join(DATA_DIR, "departments", "departments.json")
    if os.path.exists(dept_file):
        with open(dept_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        documents, metadatas, ids = [], [], []
        for i, dept in enumerate(data["departments"]):
            doc_text = (
                f"Кафедра: {dept['name']} ({dept['short_name']}). Факультет: {dept['faculty']}. "
                f"Расположение: {dept['building']}, кабинет {dept['room']}. "
                f"Телефон: {dept['phone']}, email: {dept['email']}. "
                f"Заведующий: {dept['head']}, {dept['head_title']}. Часы работы: {dept['work_hours']}."
            )
            documents.append(doc_text)
            metadatas.append({"type": "department", "name": dept["name"], "short_name": dept["short_name"]})
            ids.append(f"dept_{i}")
        if documents:
            embeddings = embed_model.encode(documents).tolist()
            collections["departments"].add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
            print(f"Загружено {len(documents)} кафедр")

    # Загрузка общей информации
    info_file = os.path.join(DATA_DIR, "info", "general_info.json")
    if os.path.exists(info_file):
        with open(info_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        documents, metadatas, ids = [], [], []
        
        deanery = data["deanery"]
        doc_text = (
            f"Деканат {deanery['faculty']}. Расположение: {deanery['building']}, кабинет {deanery['room']}. "
            f"Телефон: {deanery['phone']}, email: {deanery['email']}. Часы работы: {deanery['work_hours']}. "
            f"Услуги: {', '.join(deanery['services'])}."
        )
        documents.append(doc_text)
        metadatas.append({"type": "deanery"})
        ids.append("deanery")
        
        library = data["library"]
        doc_text = (
            f"Библиотека университета. Расположение: {library['building']}, {library['room']}. "
            f"Телефон: {library['phone']}. Часы работы: {library['work_hours']}. "
            f"Услуги: {', '.join(library['services'])}."
        )
        documents.append(doc_text)
        metadatas.append({"type": "library"})
        ids.append("library")
        
        for i, faq in enumerate(data["faq"]):
            documents.append(f"Вопрос: {faq['question']} Ответ: {faq['answer']}")
            metadatas.append({"type": "faq"})
            ids.append(f"faq_{i}")
        
        if documents:
            embeddings = embed_model.encode(documents).tolist()
            collections["general_info"].add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
            print(f"Загружено {len(documents)} записей общей информации")
    
    print("Загрузка данных завершена!")

load_initial_data()

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def is_admin(username: str) -> bool:
    if not username or not ADMIN_USERNAME:
        return False
    return username.lower() == ADMIN_USERNAME

def query_llm(prompt: str) -> str:
    try:
        response = requests.post(LLM_API_URL, json={"prompt": prompt}, timeout=120)
        return response.json().get("response", "Ошибка получения ответа")
    except Exception as e:
        return f"Ошибка связи с LLM: {e}"

def search_all_collections(query: str, n_results: int = 3) -> list:
    query_embedding = embed_model.encode([query]).tolist()
    all_results = []
    for name, collection in collections.items():
        if collection.count() > 0:
            results = collection.query(query_embeddings=query_embedding, n_results=n_results)
            if results["documents"] and results["documents"][0]:
                for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                    all_results.append({"text": doc, "metadata": metadata, "collection": name})
    return all_results

def search_schedule_by_group(group: str) -> list:
    collection = collections["schedules"]
    if collection.count() == 0:
        return []
    results = collection.get(where={"group": group})
    return results["documents"] if results["documents"] else []

def detect_intent(query: str) -> str:
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["расписание", "экзамен", "сессия", "когда", "зачет", "пара"]):
        return "schedule"
    elif any(kw in query_lower for kw in ["кафедра", "преподаватель", "деканат", "факультет"]):
        return "department"
    return "general"

def rag_query(user_question: str, group: str = None) -> str:
    intent = detect_intent(user_question)
    
    if intent == "schedule" and group:
        relevant_docs = search_schedule_by_group(group)
        if relevant_docs:
            context = "\n".join(relevant_docs)
            prompt = f"Информация о расписании группы {group}:\n{context}\n\nВопрос: {user_question}\n\nДай точный ответ."
            return query_llm(prompt)
    
    results = search_all_collections(user_question)
    if results:
        context = "\n".join([r["text"] for r in results])
        prompt = f"Контекст из базы знаний:\n{context}\n\nВопрос: {user_question}\n\nОтветь на основе контекста."
    else:
        prompt = f"Вопрос: {user_question}\n\nВ базе нет информации. Помоги общим советом."
    return query_llm(prompt)

# --- КЛАВИАТУРЫ ---
def get_main_menu_keyboard(username: str = None) -> InlineKeyboardMarkup:
    """Главное меню с кнопками"""
    buttons = [
        [InlineKeyboardButton(text="📅 Указать группу", callback_data="menu_group")],
        [InlineKeyboardButton(text="💬 Начать диалог", callback_data="menu_chat")],
        [InlineKeyboardButton(text="🆘 Тех. поддержка", callback_data="menu_support")]
    ]
    if is_admin(username):
        buttons.append([InlineKeyboardButton(text="🔧 Админ-панель", callback_data="menu_admin")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_chat_keyboard() -> ReplyKeyboardMarkup:
    """Reply-клавиатура для диалога с LLM"""
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="🔚 Завершить диалог")]],
        resize_keyboard=True,
        one_time_keyboard=False
    )

def get_group_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура выбора группы"""
    buttons = [[InlineKeyboardButton(text=g, callback_data=f"group_{g}")] for g in AVAILABLE_GROUPS]
    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="back_menu")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_admin_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📋 Открытые тикеты", callback_data="admin_tickets")],
        [InlineKeyboardButton(text="🚫 Заблокированные", callback_data="admin_blocked")],
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="◀️ В главное меню", callback_data="back_menu")]
    ])

def get_ticket_keyboard(ticket_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="⚠️ Предупредить", callback_data=f"ticket_warn_{ticket_id}"),
            InlineKeyboardButton(text="🚫 Заблокировать", callback_data=f"ticket_block_{ticket_id}")
        ],
        [
            InlineKeyboardButton(text="✅ Закрыть тикет", callback_data=f"ticket_close_{ticket_id}"),
            InlineKeyboardButton(text="◀️ Назад", callback_data="admin_tickets")
        ]
    ])

# --- TELEGRAM БОТ ---
storage = MemoryStorage()
bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher(storage=storage)

user_groups = {}
pending_actions = {}

WELCOME_TEXT = """👋 <b>Добро пожаловать!</b>

Я — справочный бот университета. Помогу найти информацию о:
• 📅 Расписании экзаменов и сессии
• 🏛 Кафедрах и преподавателях  
• 📋 Деканате, библиотеке и других службах

<b>Как пользоваться:</b>
1. Укажи свою группу для получения расписания
2. Начни диалог и задавай вопросы
3. При проблемах — обратись в поддержку

Выбери действие:"""

# ==================== ГЛАВНОЕ МЕНЮ ====================

@dp.message(CommandStart())
async def cmd_start(msg: types.Message, state: FSMContext):
    await state.clear()
    if moderation.is_user_blocked(msg.from_user.id):
        await msg.answer("🚫 Ваш аккаунт заблокирован.")
        return
    moderation.get_or_create_user(msg.from_user.id, msg.from_user.username)
    
    await msg.answer(
        WELCOME_TEXT,
        reply_markup=get_main_menu_keyboard(msg.from_user.username),
        parse_mode="HTML"
    )

@dp.callback_query(F.data == "back_menu")
async def callback_back_menu(callback: CallbackQuery, state: FSMContext):
    await state.clear()
    await callback.message.edit_text(
        WELCOME_TEXT,
        reply_markup=get_main_menu_keyboard(callback.from_user.username),
        parse_mode="HTML"
    )

async def show_main_menu(msg: types.Message, state: FSMContext):
    """Показать главное меню (новым сообщением)"""
    await state.clear()
    await msg.answer(
        WELCOME_TEXT,
        reply_markup=get_main_menu_keyboard(msg.from_user.username),
        parse_mode="HTML"
    )

# ==================== ВЫБОР ГРУППЫ ====================

@dp.callback_query(F.data == "menu_group")
async def callback_menu_group(callback: CallbackQuery):
    current_group = user_groups.get(callback.from_user.id, "не указана")
    await callback.message.edit_text(
        f"📅 <b>Выбор группы</b>\n\nТекущая группа: <b>{current_group}</b>\n\nВыбери свою группу:",
        reply_markup=get_group_keyboard(),
        parse_mode="HTML"
    )

@dp.callback_query(F.data.startswith("group_"))
async def callback_select_group(callback: CallbackQuery):
    group = callback.data.replace("group_", "")
    user_groups[callback.from_user.id] = group
    await callback.answer(f"✅ Группа {group} сохранена!", show_alert=True)
    await callback.message.edit_text(
        WELCOME_TEXT,
        reply_markup=get_main_menu_keyboard(callback.from_user.username),
        parse_mode="HTML"
    )

# ==================== ДИАЛОГ С LLM ====================

@dp.callback_query(F.data == "menu_chat")
async def callback_menu_chat(callback: CallbackQuery, state: FSMContext):
    await state.set_state(DialogStates.in_chat)
    await callback.message.delete()
    await callback.message.answer(
        "💬 <b>Диалог начат!</b>\n\n"
        "Задавай вопросы о расписании, кафедрах, деканате и т.д.\n"
        "Для завершения нажми кнопку ниже.",
        reply_markup=get_chat_keyboard(),
        parse_mode="HTML"
    )

@dp.message(F.text == "🔚 Завершить диалог")
async def end_chat(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer(
        "✅ Диалог завершён.",
        reply_markup=ReplyKeyboardRemove()
    )
    await show_main_menu(msg, state)

@dp.message(DialogStates.in_chat)
async def handle_chat_message(msg: types.Message, state: FSMContext):
    if moderation.is_user_blocked(msg.from_user.id):
        await msg.answer("🚫 Ваш аккаунт заблокирован.", reply_markup=ReplyKeyboardRemove())
        await state.clear()
        return
    
    # Проверка на нарушения
    violation = moderation.check_message(msg.text)
    if violation:
        ticket = moderation.create_ticket(msg.from_user.id, msg.from_user.username, violation, msg.text)
        user = moderation.get_user_stats(msg.from_user.id)
        if user.warnings >= 3:
            moderation.block_user(msg.from_user.id, "Автоблокировка: 3 нарушения")
            await msg.answer("🚫 Аккаунт заблокирован за нарушения.", reply_markup=ReplyKeyboardRemove())
            await state.clear()
            return
        await msg.answer(f"⚠️ Сообщение нарушает правила.\nПредупреждений: {user.warnings}/3")
        return
    
    await bot.send_chat_action(chat_id=msg.chat.id, action="typing")
    
    user_group = user_groups.get(msg.from_user.id)
    intent = detect_intent(msg.text)
    
    if intent == "schedule" and not user_group:
        await msg.answer("📅 Для расписания укажи группу.\nЗаверши диалог и выбери группу в меню.")
        return
    
    response = rag_query(msg.text, group=user_group)
    await msg.answer(response)

# ==================== ТЕХ. ПОДДЕРЖКА ====================

@dp.callback_query(F.data == "menu_support")
async def callback_menu_support(callback: CallbackQuery, state: FSMContext):
    await state.set_state(DialogStates.waiting_support_msg)
    await callback.message.edit_text(
        "🆘 <b>Тех. поддержка</b>\n\n"
        "Опишите вашу проблему или вопрос.\n"
        "Сообщение будет передано администратору.\n\n"
        "Для отмены отправьте /cancel",
        parse_mode="HTML"
    )

@dp.message(DialogStates.waiting_support_msg)
async def handle_support_message(msg: types.Message, state: FSMContext):
    # Создаём тикет поддержки
    from moderation import Ticket, ViolationType
    from datetime import datetime
    
    moderation.ticket_counter += 1
    ticket = Ticket(
        id=moderation.ticket_counter,
        user_id=msg.from_user.id,
        username=msg.from_user.username or "unknown",
        violation_type=ViolationType.SPAM,  # Используем как тип "обращение"
        message_text=f"[ПОДДЕРЖКА] {msg.text[:500]}",
        timestamp=datetime.now()
    )
    moderation.tickets.append(ticket)
    
    await state.clear()
    await msg.answer(
        f"✅ <b>Обращение #{ticket.id} создано!</b>\n\n"
        "Администратор рассмотрит его в ближайшее время.",
        parse_mode="HTML"
    )
    await show_main_menu(msg, state)

@dp.message(Command("cancel"))
async def cmd_cancel(msg: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state:
        await state.clear()
        await msg.answer("❌ Действие отменено.", reply_markup=ReplyKeyboardRemove())
    await show_main_menu(msg, state)

# ==================== АДМИН-ПАНЕЛЬ ====================

@dp.callback_query(F.data == "menu_admin")
async def callback_menu_admin(callback: CallbackQuery):
    if not is_admin(callback.from_user.username):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    
    open_tickets = len(moderation.get_open_tickets())
    blocked_users = len(moderation.get_all_blocked_users())
    
    await callback.message.edit_text(
        f"🔧 <b>Админ-панель</b>\n\n"
        f"📋 Открытых тикетов: {open_tickets}\n"
        f"🚫 Заблокировано: {blocked_users}\n"
        f"👥 Всего пользователей: {len(moderation.users)}",
        reply_markup=get_admin_menu_keyboard(),
        parse_mode="HTML"
    )

@dp.message(Command("admin"))
async def cmd_admin(msg: types.Message):
    if not is_admin(msg.from_user.username):
        await msg.answer("⛔ Нет доступа к админ-панели.")
        return
    
    open_tickets = len(moderation.get_open_tickets())
    blocked_users = len(moderation.get_all_blocked_users())
    
    await msg.answer(
        f"🔧 <b>Админ-панель</b>\n\n"
        f"📋 Открытых тикетов: {open_tickets}\n"
        f"🚫 Заблокировано: {blocked_users}\n"
        f"👥 Всего пользователей: {len(moderation.users)}",
        reply_markup=get_admin_menu_keyboard(),
        parse_mode="HTML"
    )

@dp.callback_query(F.data == "admin_tickets")
async def callback_admin_tickets(callback: CallbackQuery):
    if not is_admin(callback.from_user.username):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    
    tickets = moderation.get_open_tickets()
    if not tickets:
        await callback.message.edit_text(
            "✅ Нет открытых тикетов!",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_admin")]
            ])
        )
        return
    
    keyboard = []
    for ticket in tickets[-10:]:
        emoji = "🆘" if "[ПОДДЕРЖКА]" in ticket.message_text else {
            ViolationType.PROFANITY: "🤬", ViolationType.AGGRESSION: "😡",
            ViolationType.DANGEROUS: "⚠️", ViolationType.SPAM: "📢"
        }.get(ticket.violation_type, "❓")
        keyboard.append([InlineKeyboardButton(
            text=f"{emoji} #{ticket.id} @{ticket.username}",
            callback_data=f"ticket_view_{ticket.id}"
        )])
    keyboard.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_admin")])
    
    await callback.message.edit_text(
        f"📋 <b>Открытые тикеты ({len(tickets)})</b>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
        parse_mode="HTML"
    )

@dp.callback_query(F.data.startswith("ticket_view_"))
async def callback_ticket_view(callback: CallbackQuery):
    if not is_admin(callback.from_user.username):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    
    ticket_id = int(callback.data.split("_")[2])
    ticket = moderation.get_ticket_by_id(ticket_id)
    if not ticket:
        await callback.answer("Тикет не найден", show_alert=True)
        return
    
    vtype_name = moderation.get_violation_type_name(ticket.violation_type)
    if "[ПОДДЕРЖКА]" in ticket.message_text:
        vtype_name = "🆘 Обращение в поддержку"
    
    user = moderation.get_user_stats(ticket.user_id)
    await callback.message.edit_text(
        f"🎫 <b>Тикет #{ticket.id}</b>\n\n"
        f"👤 @{ticket.username} (ID: <code>{ticket.user_id}</code>)\n"
        f"⚠️ Тип: {vtype_name}\n"
        f"📅 {ticket.timestamp.strftime('%d.%m.%Y %H:%M')}\n"
        f"⚡ Предупреждений: {user.warnings if user else 0}\n\n"
        f"💬 <b>Сообщение:</b>\n<i>{ticket.message_text[:300]}</i>",
        reply_markup=get_ticket_keyboard(ticket_id),
        parse_mode="HTML"
    )

@dp.callback_query(F.data.startswith("ticket_warn_"))
async def callback_ticket_warn(callback: CallbackQuery):
    if not is_admin(callback.from_user.username):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    
    ticket_id = int(callback.data.split("_")[2])
    ticket = moderation.get_ticket_by_id(ticket_id)
    if not ticket:
        await callback.answer("Тикет не найден", show_alert=True)
        return
    
    try:
        user = moderation.get_user_stats(ticket.user_id)
        await bot.send_message(
            ticket.user_id,
            f"⚠️ <b>Предупреждение</b>\n\nВаше сообщение нарушает правила.\nПредупреждений: {user.warnings}/3",
            parse_mode="HTML"
        )
    except:
        pass
    
    moderation.resolve_ticket(ticket_id, "Предупреждение выдано")
    await callback.answer("✅ Предупреждение отправлено", show_alert=True)
    await callback_admin_tickets(callback)

@dp.callback_query(F.data.startswith("ticket_block_"))
async def callback_ticket_block(callback: CallbackQuery, state: FSMContext):
    if not is_admin(callback.from_user.username):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    
    ticket_id = int(callback.data.split("_")[2])
    ticket = moderation.get_ticket_by_id(ticket_id)
    if not ticket:
        await callback.answer("Тикет не найден", show_alert=True)
        return
    
    pending_actions[callback.from_user.id] = {"user_id": ticket.user_id, "ticket_id": ticket_id}
    await callback.message.edit_text(
        f"🚫 <b>Блокировка @{ticket.username}</b>\n\nВведите причину:",
        parse_mode="HTML"
    )
    await state.set_state(DialogStates.waiting_block_reason)

@dp.callback_query(F.data.startswith("ticket_close_"))
async def callback_ticket_close(callback: CallbackQuery):
    if not is_admin(callback.from_user.username):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    
    ticket_id = int(callback.data.split("_")[2])
    ticket = moderation.get_ticket_by_id(ticket_id)
    
    # Если это обращение в поддержку — отправляем ответ
    if ticket and "[ПОДДЕРЖКА]" in ticket.message_text:
        try:
            await bot.send_message(ticket.user_id, "✅ Ваше обращение рассмотрено администратором.")
        except:
            pass
    
    moderation.resolve_ticket(ticket_id, "Закрыт")
    await callback.answer("✅ Тикет закрыт", show_alert=True)
    await callback_admin_tickets(callback)

@dp.message(DialogStates.waiting_block_reason)
async def process_block_reason(msg: types.Message, state: FSMContext):
    if not is_admin(msg.from_user.username):
        await state.clear()
        return
    
    action = pending_actions.get(msg.from_user.id)
    if not action:
        await state.clear()
        return
    
    moderation.block_user(action["user_id"], msg.text)
    if action.get("ticket_id"):
        moderation.resolve_ticket(action["ticket_id"], f"Заблокирован: {msg.text}")
    
    try:
        await bot.send_message(action["user_id"], f"🚫 <b>Аккаунт заблокирован</b>\n\nПричина: {msg.text}", parse_mode="HTML")
    except:
        pass
    
    del pending_actions[msg.from_user.id]
    await state.clear()
    await msg.answer(f"✅ Пользователь заблокирован.", reply_markup=get_admin_menu_keyboard())

@dp.callback_query(F.data == "admin_blocked")
async def callback_admin_blocked(callback: CallbackQuery):
    if not is_admin(callback.from_user.username):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    
    blocked = moderation.get_all_blocked_users()
    if not blocked:
        await callback.message.edit_text(
            "✅ Нет заблокированных!",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_admin")]
            ])
        )
        return
    
    keyboard = [[InlineKeyboardButton(text=f"🚫 @{u.username}", callback_data=f"user_unblock_{u.user_id}")] for u in blocked]
    keyboard.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_admin")])
    
    await callback.message.edit_text(
        f"🚫 <b>Заблокированные ({len(blocked)})</b>\n\nНажмите для разблокировки:",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard),
        parse_mode="HTML"
    )

@dp.callback_query(F.data.startswith("user_unblock_"))
async def callback_user_unblock(callback: CallbackQuery):
    if not is_admin(callback.from_user.username):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    
    user_id = int(callback.data.split("_")[2])
    if moderation.unblock_user(user_id):
        try:
            await bot.send_message(user_id, "✅ Ваш аккаунт разблокирован.")
        except:
            pass
        await callback.answer("✅ Разблокирован", show_alert=True)
    await callback_admin_blocked(callback)

@dp.callback_query(F.data == "admin_stats")
async def callback_admin_stats(callback: CallbackQuery):
    if not is_admin(callback.from_user.username):
        await callback.answer("⛔ Нет доступа", show_alert=True)
        return
    
    total_tickets = len(moderation.tickets)
    open_tickets = len(moderation.get_open_tickets())
    blocked_users = len(moderation.get_all_blocked_users())
    
    type_stats = {}
    for t in moderation.tickets:
        k = "support" if "[ПОДДЕРЖКА]" in t.message_text else t.violation_type.value
        type_stats[k] = type_stats.get(k, 0) + 1
    stats_text = "\n".join([f"  • {k}: {v}" for k, v in type_stats.items()]) or "  Нет данных"
    
    await callback.message.edit_text(
        f"📊 <b>Статистика</b>\n\n"
        f"👥 Пользователей: {len(moderation.users)}\n"
        f"🚫 Заблокировано: {blocked_users}\n"
        f"🎫 Тикетов: {total_tickets} (открыто: {open_tickets})\n\n"
        f"<b>По типам:</b>\n{stats_text}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_admin")]
        ]),
        parse_mode="HTML"
    )

# ==================== FALLBACK ====================

@dp.message()
async def fallback_handler(msg: types.Message, state: FSMContext):
    """Обработка сообщений вне диалога"""
    if moderation.is_user_blocked(msg.from_user.id):
        await msg.answer("🚫 Ваш аккаунт заблокирован.")
        return
    await msg.answer(
        "ℹ️ Используй меню для навигации.\nНажми /start для открытия меню.",
        reply_markup=ReplyKeyboardRemove()
    )

if __name__ == "__main__":
    print(f"Бот запущен! Админ: @{ADMIN_USERNAME}")
    asyncio.run(dp.start_polling(bot))
