from langchain.memory.chat_memory import BaseChatMessageHistory
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from app.db.database import agents_collection
from app.services.agents_services import get_or_create_conversation  # Importamos la función
from typing import List
from datetime import datetime


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, agent_name: str, session_id: str):
        self.agent_name = agent_name
        self.session_id = session_id
        self.messages = None # Usamos None para saber si ya se cargaron los mensajes

    async def _load_messages(self):
        """Carga los últimos 20 mensajes desde MongoDB de forma asíncrona."""
        if self.messages is not None:
            return  # Si ya se cargaron, no volver a hacerlo

        # 🔹 Asegurar que la conversación existe antes de cargar mensajes
        await get_or_create_conversation(self.agent_name, self.session_id)

        agent_doc = await agents_collection.find_one(
            {'name': self.agent_name, 'conversations.uuid': self.session_id},
            {'conversations.$': 1}  # Solo traemos la conversación relevante
        )

        if not agent_doc or 'conversations' not in agent_doc:
            self.messages = []
            return

        conversation = agent_doc['conversations'][0]  # Solo obtenemos la conversación filtrada
        sorted_messages = sorted(
            conversation.get('messages', []),
            key=lambda x: x['time'],
            reverse=True
        )[:20]  # Limitar a 20 mensajes

        self.messages = [
            HumanMessage(content=msg['text']) if msg['sender'] == 'user' else AIMessage(content=msg['text'])
            for msg in reversed(sorted_messages)  # Revertimos para mantener orden cronológico
        ]

    async def aget_messages(self) -> List[BaseMessage]:
        """Devuelve los mensajes cargados en el historial."""
        if self.messages is None:
            await self._load_messages()
        return self.messages

    async def add_message(self, message: BaseMessage):
        """Agrega un mensaje al historial y lo almacena en MongoDB."""
        # 🔹 Asegurar que la conversación existe antes de guardar un mensaje
        await get_or_create_conversation(self.agent_name, self.session_id)

        if self.messages is None:
            await self._load_messages()

        self.messages.append(message)

        sender = 'user' if isinstance(message, HumanMessage) else 'bot'
        message_data = {
            'sender': sender,
            'text': message.content,
            'time': datetime.utcnow()
        }

        await agents_collection.update_one(
            {'name': self.agent_name, 'conversations.uuid': self.session_id},
            {'$push': {'conversations.$.messages': message_data}},
            upsert=True
        )

    async def clear(self):
        """Vacía el historial de mensajes tanto en memoria como en MongoDB."""
        # 🔹 Asegurar que la conversación existe antes de limpiarla
        await get_or_create_conversation(self.agent_name, self.session_id)

        self.messages = []  # También vaciamos la lista en memoria
        await agents_collection.update_one(
            {'name': self.agent_name, 'conversations.uuid': self.session_id},
            {'$set': {'conversations.$.messages': []}}
        )
