"""
Telegram Bot — клиент для RAG API.
Вся логика RAG вынесена в отдельный сервис.
"""

import os
import asyncio
import requests
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery,
    ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
)

# --- НАСТРОЙКИ ---
RAG_API_URL = os.getenv("RAG_API_URL", "http://rag_service:8001")
RAG_API_KEY = os.getenv("RAG_API_KEY", "")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "").lower()

# --- СОСТОЯНИЯ ---
class DialogStates(StatesGroup):
    in_chat = State()
    waiting_support_msg = State()
    waiting_block_reason = State()

# --- RAG API CLIENT ---
class RAGClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url
        self.headers = {"X-API-Key": api_key} if api_key else {}
    
    def chat(self, message: str, user_id: str, group: str = None) -> dict:
        try:
            resp = requests.post(
                f"{self.base_url}/chat",
                json={"message": message, "user_id": user_id, "group": group},
                headers=self.headers,
                timeout=120
            )
            if resp.status_code == 403:
                return {"error": "blocked", "detail": resp.json().get("detail")}
            if resp.status_code == 400:
                return {"error": "violation", "detail": resp.json().get("detail")}
            return resp.json()
        except Exception as e:
            return {"error": "connection", "detail": str(e)}
    
    def get_user(self, user_id: str) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/users/{user_id}", headers=self.headers)
            return resp.json()
        except:
            return {}
    
    def create_user(self, user_id: str, username: str) -> dict:
        try:
            resp = requests.post(
                f"{self.base_url}/users",
                json={"user_id": user_id, "username": username, "platform": "telegram"},
                headers=self.headers
            )
            return resp.json()
        except:
            return {}
    
    def set_group(self, user_id: str, group: str) -> bool:
        try:
            resp = requests.put(f"{self.base_url}/users/{user_id}/group?group={group}", headers=self.headers)
            return resp.status_code == 200
        except:
            return False
    
    def get_groups(self) -> list:
        try:
            resp = requests.get(f"{self.base_url}/groups", headers=self.headers)
            return resp.json().get("groups", [])
        except:
            return ["ИВТ-21", "ПИ-22"]  # Fallback
    
    def get_stats(self) -> dict:
        try:
            resp = requests.get(f"{self.base_url}/admin/stats", headers=self.headers)
            return resp.json()
        except:
            return {}
    
    def get_users_list(self) -> list:
        try:
            resp = requests.get(f"{self.base_url}/admin/users", headers=self.headers)
            return resp.json().get("users", [])
        except:
            return []
    
    def block_user(self, user_id: str, reason: str) -> bool:
        try:
            resp = requests.post(f"{self.base_url}/admin/users/{user_id}/block?reason={reason}", headers=self.headers)
            return resp.status_code == 200
        except:
            return False
    
    def unblock_user(self, user_id: str) -> bool:
        try:
            resp = requests.post(f"{self.base_url}/admin/users/{user_id}/unblock", headers=self.headers)
            return resp.status_code == 200
        except:
            return False

rag = RAGClient(RAG_API_URL, RAG_API_KEY)

# --- ВСПОМОГАТЕЛЬНЫЕ ---
def is_admin(username: str) -> bool:
    if not username or not ADMIN_USERNAME:
        return False
    return username.lower() == ADMIN_USERNAME

# --- КЛАВИАТУРЫ ---
def get_main_menu(username: str = None) -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="📅 Указать группу", callback_data="menu_group")],
        [InlineKeyboardButton(text="💬 Начать диалог", callback_data="menu_chat")],
        [InlineKeyboardButton(text="🆘 Тех. поддержка", callback_data="menu_support")]
    ]
    if is_admin(username):
        buttons.append([InlineKeyboardButton(text="🔧 Админ-панель", callback_data="menu_admin")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)

def get_chat_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text="🔚 Завершить диалог")]],
        resize_keyboard=True
    )

def get_admin_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📊 Статистика", callback_data="admin_stats")],
        [InlineKeyboardButton(text="👥 Пользователи", callback_data="admin_users")],
        [InlineKeyboardButton(text="◀️ Назад", callback_data="back_menu")]
    ])

# --- БОТ ---
storage = MemoryStorage()
bot = Bot(token=os.getenv("TELEGRAM_TOKEN"))
dp = Dispatcher(storage=storage)

user_groups_cache = {}  # Локальный кэш групп
pending_actions = {}

WELCOME = """👋 <b>Добро пожаловать!</b>

Я — справочный бот университета. Помогу с:
• 📅 Расписанием экзаменов
• 🏛 Информацией о кафедрах
• 📋 Вопросами о деканате и библиотеке

<b>Как пользоваться:</b>
1. Укажи группу для расписания
2. Начни диалог и задавай вопросы
3. При проблемах — обратись в поддержку"""

# ==================== ГЛАВНОЕ МЕНЮ ====================

@dp.message(CommandStart())
async def cmd_start(msg: types.Message, state: FSMContext):
    await state.clear()
    rag.create_user(str(msg.from_user.id), msg.from_user.username)
    await msg.answer(WELCOME, reply_markup=get_main_menu(msg.from_user.username), parse_mode="HTML")

@dp.callback_query(F.data == "back_menu")
async def cb_back_menu(cb: CallbackQuery, state: FSMContext):
    await state.clear()
    await cb.message.edit_text(WELCOME, reply_markup=get_main_menu(cb.from_user.username), parse_mode="HTML")

async def show_menu(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer(WELCOME, reply_markup=get_main_menu(msg.from_user.username), parse_mode="HTML")

# ==================== ВЫБОР ГРУППЫ ====================

@dp.callback_query(F.data == "menu_group")
async def cb_menu_group(cb: CallbackQuery):
    groups = rag.get_groups()
    current = user_groups_cache.get(cb.from_user.id, "не указана")
    
    buttons = [[InlineKeyboardButton(text=g, callback_data=f"group_{g}")] for g in groups]
    buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="back_menu")])
    
    await cb.message.edit_text(
        f"📅 <b>Выбор группы</b>\n\nТекущая: <b>{current}</b>",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons),
        parse_mode="HTML"
    )

@dp.callback_query(F.data.startswith("group_"))
async def cb_select_group(cb: CallbackQuery):
    group = cb.data.replace("group_", "")
    user_groups_cache[cb.from_user.id] = group
    rag.set_group(str(cb.from_user.id), group)
    await cb.answer(f"✅ Группа {group} сохранена!", show_alert=True)
    await cb.message.edit_text(WELCOME, reply_markup=get_main_menu(cb.from_user.username), parse_mode="HTML")

# ==================== ДИАЛОГ ====================

@dp.callback_query(F.data == "menu_chat")
async def cb_menu_chat(cb: CallbackQuery, state: FSMContext):
    await state.set_state(DialogStates.in_chat)
    await cb.message.delete()
    await cb.message.answer(
        "💬 <b>Диалог начат!</b>\n\nЗадавай вопросы. Для выхода нажми кнопку.",
        reply_markup=get_chat_keyboard(),
        parse_mode="HTML"
    )

@dp.message(F.text == "🔚 Завершить диалог")
async def end_chat(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer("✅ Диалог завершён.", reply_markup=ReplyKeyboardRemove())
    await show_menu(msg, state)

@dp.message(DialogStates.in_chat)
async def handle_chat(msg: types.Message, state: FSMContext):
    await bot.send_chat_action(msg.chat.id, "typing")
    
    group = user_groups_cache.get(msg.from_user.id)
    result = rag.chat(msg.text, str(msg.from_user.id), group)
    
    if result.get("error") == "blocked":
        await msg.answer("🚫 Ваш аккаунт заблокирован.", reply_markup=ReplyKeyboardRemove())
        await state.clear()
        return
    
    if result.get("error") == "violation":
        await msg.answer(f"⚠️ {result.get('detail', 'Нарушение правил')}")
        return
    
    if result.get("error"):
        await msg.answer(f"❌ Ошибка: {result.get('detail', 'Неизвестная ошибка')}")
        return
    
    await msg.answer(result.get("response", "Нет ответа"))

# ==================== ПОДДЕРЖКА ====================

@dp.callback_query(F.data == "menu_support")
async def cb_menu_support(cb: CallbackQuery, state: FSMContext):
    await state.set_state(DialogStates.waiting_support_msg)
    await cb.message.edit_text(
        "🆘 <b>Тех. поддержка</b>\n\nОпишите проблему. Для отмены: /cancel",
        parse_mode="HTML"
    )

@dp.message(DialogStates.waiting_support_msg)
async def handle_support(msg: types.Message, state: FSMContext):
    # В реальном проекте здесь отправка в систему тикетов
    await state.clear()
    await msg.answer("✅ Обращение отправлено! Администратор рассмотрит его.")
    await show_menu(msg, state)

@dp.message(Command("cancel"))
async def cmd_cancel(msg: types.Message, state: FSMContext):
    await state.clear()
    await msg.answer("❌ Отменено.", reply_markup=ReplyKeyboardRemove())
    await show_menu(msg, state)

# ==================== АДМИН ====================

@dp.callback_query(F.data == "menu_admin")
async def cb_menu_admin(cb: CallbackQuery):
    if not is_admin(cb.from_user.username):
        await cb.answer("⛔ Нет доступа", show_alert=True)
        return
    
    stats = rag.get_stats()
    await cb.message.edit_text(
        f"🔧 <b>Админ-панель</b>\n\n"
        f"👥 Пользователей: {stats.get('total_users', 0)}\n"
        f"🚫 Заблокировано: {stats.get('blocked_users', 0)}\n"
        f"📚 Документов: {sum(stats.get('collections', {}).values())}",
        reply_markup=get_admin_menu(),
        parse_mode="HTML"
    )

@dp.callback_query(F.data == "admin_stats")
async def cb_admin_stats(cb: CallbackQuery):
    if not is_admin(cb.from_user.username):
        return
    
    stats = rag.get_stats()
    colls = stats.get("collections", {})
    
    await cb.message.edit_text(
        f"📊 <b>Статистика</b>\n\n"
        f"👥 Пользователей: {stats.get('total_users', 0)}\n"
        f"🚫 Заблокировано: {stats.get('blocked_users', 0)}\n"
        f"⚠️ С предупреждениями: {stats.get('users_with_warnings', 0)}\n\n"
        f"<b>Документы:</b>\n"
        f"• Расписания: {colls.get('schedules', 0)}\n"
        f"• Кафедры: {colls.get('departments', 0)}\n"
        f"• Общее: {colls.get('general_info', 0)}",
        reply_markup=InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_admin")]
        ]),
        parse_mode="HTML"
    )

@dp.callback_query(F.data == "admin_users")
async def cb_admin_users(cb: CallbackQuery):
    if not is_admin(cb.from_user.username):
        return
    
    users = rag.get_users_list()
    blocked = [u for u in users if u.get("is_blocked")]
    
    if not blocked:
        text = "✅ Нет заблокированных пользователей"
        kb = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="◀️ Назад", callback_data="menu_admin")]
        ])
    else:
        text = f"🚫 <b>Заблокированные ({len(blocked)})</b>\n\nНажмите для разблокировки:"
        buttons = [
            [InlineKeyboardButton(text=f"@{u.get('username', u['user_id'])}", callback_data=f"unblock_{u['user_id']}")]
            for u in blocked[:10]
        ]
        buttons.append([InlineKeyboardButton(text="◀️ Назад", callback_data="menu_admin")])
        kb = InlineKeyboardMarkup(inline_keyboard=buttons)
    
    await cb.message.edit_text(text, reply_markup=kb, parse_mode="HTML")

@dp.callback_query(F.data.startswith("unblock_"))
async def cb_unblock(cb: CallbackQuery):
    if not is_admin(cb.from_user.username):
        return
    
    user_id = cb.data.replace("unblock_", "")
    if rag.unblock_user(user_id):
        await cb.answer("✅ Разблокирован", show_alert=True)
    await cb_admin_users(cb)

@dp.message(Command("admin"))
async def cmd_admin(msg: types.Message):
    if not is_admin(msg.from_user.username):
        await msg.answer("⛔ Нет доступа")
        return
    
    stats = rag.get_stats()
    await msg.answer(
        f"🔧 <b>Админ-панель</b>\n\n"
        f"👥 Пользователей: {stats.get('total_users', 0)}\n"
        f"🚫 Заблокировано: {stats.get('blocked_users', 0)}",
        reply_markup=get_admin_menu(),
        parse_mode="HTML"
    )

# ==================== FALLBACK ====================

@dp.message()
async def fallback(msg: types.Message, state: FSMContext):
    user = rag.get_user(str(msg.from_user.id))
    if user.get("is_blocked"):
        await msg.answer("🚫 Аккаунт заблокирован.")
        return
    await msg.answer("ℹ️ Нажми /start для меню.", reply_markup=ReplyKeyboardRemove())

if __name__ == "__main__":
    print(f"Bot started! Admin: @{ADMIN_USERNAME}")
    asyncio.run(dp.start_polling(bot))
