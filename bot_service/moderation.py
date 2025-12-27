"""
Модуль модерации: определение нецензурной лексики и опасного поведения
"""

import re
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

class ViolationType(Enum):
    PROFANITY = "profanity"           # Нецензурная лексика
    AGGRESSION = "aggression"         # Агрессивное поведение
    DANGEROUS = "dangerous"           # Опасный контент
    SPAM = "spam"                     # Спам

@dataclass
class Ticket:
    id: int
    user_id: int
    username: str
    violation_type: ViolationType
    message_text: str
    timestamp: datetime
    resolved: bool = False
    resolution: Optional[str] = None

@dataclass
class UserRecord:
    user_id: int
    username: str
    warnings: int = 0
    is_blocked: bool = False
    blocked_reason: Optional[str] = None
    blocked_at: Optional[datetime] = None

class ModerationSystem:
    def __init__(self):
        self.tickets: list[Ticket] = []
        self.users: dict[int, UserRecord] = {}
        self.ticket_counter = 0
        
        # Паттерны для определения нарушений (базовый список, можно расширить)
        self.profanity_patterns = [
            r'\b(бля|блять|блядь|блядина|блядство)\b',
            r'\b(хуй|хуя|хуе|хуё|хуи)\b',
            r'\b(пизд|пизда|пиздец|пиздёж)\b',
            r'\b(ебать|ебан|ебла|ебло|ёб|еб)\b',
            r'\b(сука|сучка|сучар)\b',
            r'\b(мудак|мудила|мудень)\b',
            r'\b(дерьмо|говно|срань)\b',
            r'\b(залупа|членосос|хер)\b',
            r'\b(fuck|shit|bitch|asshole)\b',
        ]
        
        self.aggression_patterns = [
            r'\b(убью|убить|убей|сдохни|сдохнешь)\b',
            r'\b(урою|закопаю|порешу|прибью)\b',
            r'\b(ненавижу тебя|тварь|мразь|урод)\b',
            r'\b(угрожаю|угроза|отомщу)\b',
        ]
        
        self.dangerous_patterns = [
            r'\b(бомба|взрыв|взорвать|теракт)\b',
            r'\b(наркотик|героин|кокаин|мет)\b',
            r'\b(оружие|пистолет|автомат)\b',
            r'\b(суицид|самоубийство|повеситься)\b',
        ]
    
    def get_or_create_user(self, user_id: int, username: str) -> UserRecord:
        """Получить или создать запись пользователя"""
        if user_id not in self.users:
            self.users[user_id] = UserRecord(user_id=user_id, username=username or "unknown")
        return self.users[user_id]
    
    def check_message(self, text: str) -> Optional[ViolationType]:
        """Проверить сообщение на нарушения"""
        text_lower = text.lower()
        
        # Проверка на нецензурную лексику
        for pattern in self.profanity_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return ViolationType.PROFANITY
        
        # Проверка на агрессию
        for pattern in self.aggression_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return ViolationType.AGGRESSION
        
        # Проверка на опасный контент
        for pattern in self.dangerous_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return ViolationType.DANGEROUS
        
        return None
    
    def create_ticket(self, user_id: int, username: str, violation_type: ViolationType, message_text: str) -> Ticket:
        """Создать тикет о нарушении"""
        self.ticket_counter += 1
        ticket = Ticket(
            id=self.ticket_counter,
            user_id=user_id,
            username=username or "unknown",
            violation_type=violation_type,
            message_text=message_text[:500],  # Ограничиваем длину
            timestamp=datetime.now()
        )
        self.tickets.append(ticket)
        
        # Увеличиваем счётчик предупреждений
        user = self.get_or_create_user(user_id, username)
        user.warnings += 1
        
        return ticket
    
    def get_open_tickets(self) -> list[Ticket]:
        """Получить открытые тикеты"""
        return [t for t in self.tickets if not t.resolved]
    
    def get_ticket_by_id(self, ticket_id: int) -> Optional[Ticket]:
        """Получить тикет по ID"""
        for ticket in self.tickets:
            if ticket.id == ticket_id:
                return ticket
        return None
    
    def resolve_ticket(self, ticket_id: int, resolution: str) -> bool:
        """Закрыть тикет"""
        ticket = self.get_ticket_by_id(ticket_id)
        if ticket:
            ticket.resolved = True
            ticket.resolution = resolution
            return True
        return False
    
    def block_user(self, user_id: int, reason: str) -> bool:
        """Заблокировать пользователя"""
        if user_id in self.users:
            user = self.users[user_id]
            user.is_blocked = True
            user.blocked_reason = reason
            user.blocked_at = datetime.now()
            return True
        return False
    
    def unblock_user(self, user_id: int) -> bool:
        """Разблокировать пользователя"""
        if user_id in self.users:
            user = self.users[user_id]
            user.is_blocked = False
            user.blocked_reason = None
            user.blocked_at = None
            return True
        return False
    
    def is_user_blocked(self, user_id: int) -> bool:
        """Проверить, заблокирован ли пользователь"""
        if user_id in self.users:
            return self.users[user_id].is_blocked
        return False
    
    def get_user_stats(self, user_id: int) -> Optional[UserRecord]:
        """Получить статистику пользователя"""
        return self.users.get(user_id)
    
    def get_all_blocked_users(self) -> list[UserRecord]:
        """Получить список заблокированных пользователей"""
        return [u for u in self.users.values() if u.is_blocked]
    
    def get_violation_type_name(self, vtype: ViolationType) -> str:
        """Получить название типа нарушения"""
        names = {
            ViolationType.PROFANITY: "🤬 Нецензурная лексика",
            ViolationType.AGGRESSION: "😡 Агрессия",
            ViolationType.DANGEROUS: "⚠️ Опасный контент",
            ViolationType.SPAM: "📢 Спам"
        }
        return names.get(vtype, "Неизвестно")

# Глобальный экземпляр системы модерации
moderation = ModerationSystem()
