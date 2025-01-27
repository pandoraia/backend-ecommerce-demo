from langchain.memory.chat_memory import BaseChatMessageHistory
from langchain.schema import BaseMessage, AIMessage, HumanMessage
from app.db.database import agents_collection
from typing import List
from datetime import datetime
from app.services.agents_services import get_or_create_conversation
from pymongo import DESCENDING


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, agent_name: str, uuid: str):
        self.agent_name = agent_name
        self.uuid = uuid
        self.messages = []

    async def _load_messages(self):
        """Carga los últimos 20 mensajes desde MongoDB de forma asíncrona."""
        agent_doc = await agents_collection.find_one({'name': self.agent_name})
        messages = []
        if agent_doc:
            # Buscar la conversación con el UUID correspondiente
            conversation = next(
                (conv for conv in agent_doc.get('conversations', []) if conv['uuid'] == self.uuid), None)
            if conversation:
                # Obtener los últimos 10 mensajes ordenados por 'time' descendente
                sorted_messages = sorted(
                    conversation.get('messages', []),
                    key=lambda x: x['time'],
                    reverse=True
                )[:20]  # Limitar a 10 mensajes
                # Invertir para obtener en orden cronológico
                for msg_data in sorted_messages[::-1]:
                    if msg_data['sender'] == 'user':
                        messages.append(HumanMessage(content=msg_data['text']))
                    elif msg_data['sender'] == 'bot':
                        messages.append(AIMessage(content=msg_data['text']))
        self.messages = messages

    async def aget_messages(self) -> List[BaseMessage]:
        """Devuelve los mensajes cargados en el historial."""
        if not self.messages:
            await self._load_messages()
        return self.messages

    async def add_message(self, message: BaseMessage):
        """Agrega un mensaje al historial y lo almacena en MongoDB."""
        # Asegurarse de que el agente y la conversación existen
        await get_or_create_conversation(self.agent_name, self.uuid)

        self.messages.append(message)
        sender = 'user' if isinstance(message, HumanMessage) else 'bot'
        message_data = {
            'sender': sender,
            'text': message.content,
            'time': datetime.utcnow()
        }
        await agents_collection.update_one(
            {'name': self.agent_name, 'conversations.uuid': self.uuid},
            {'$push': {'conversations.$.messages': message_data}},
            upsert=True
        )

    def clear(self):
        """Vacía el historial de mensajes tanto en memoria como en MongoDB."""
        self.messages = []
        agents_collection.update_one(
            {'name': self.agent_name, 'conversations.uuid': self.uuid},
            {'$set': {'conversations.$.messages': []}}
        )
