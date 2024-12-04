# app/services/agents_service.py
from typing import Dict, Any,List
from app.db.database import agents_collection
from datetime import datetime

async def get_or_create_agent(name: str) -> Dict[str, Any]:
    """Obtiene o crea un documento de agente en MongoDB."""
    agent_doc = await agents_collection.find_one({'name': name})
    if not agent_doc:
        agent_doc = {
            'name': name,
            'conversations': []
        }
        result = await agents_collection.insert_one(agent_doc)
        agent_doc['_id'] = result.inserted_id
    return agent_doc


async def get_or_create_conversation(agent_name: str, uuid: str) -> Dict[str, Any]:
    """Obtiene o crea una conversación para un agente específico."""
    agent_doc = await get_or_create_agent(agent_name)
    conversations = agent_doc.get('conversations', [])
    conversation = next((conv for conv in conversations if conv['uuid'] == uuid), None)

    if not conversation:
        conversation = {
            'uuid': uuid,
            'messages': []
        }
        await agents_collection.update_one(
            {'_id': agent_doc['_id']},
            {'$push': {'conversations': conversation}}
        )
    return conversation


async def add_message_to_conversation(agent_name: str, uuid: str, sender: str, text: str):
    """Agrega un mensaje a una conversación en MongoDB."""
    # Asegurarse de que el agente y la conversación existen
    await get_or_create_conversation(agent_name, uuid)

    message = {
        'sender': sender,
        'text': text,
        'time': datetime.utcnow()
    }

    await agents_collection.update_one(
        {'name': agent_name, 'conversations.uuid': uuid},
        {'$push': {'conversations.$.messages': message}}
    )


async def get_conversation_messages(agent_name: str, uuid: str) -> List[Dict[str, Any]]:
    """Obtiene todos los mensajes de una conversación específica."""
    agent_doc = await get_or_create_agent(agent_name)
    conversation = next(
        (conv for conv in agent_doc.get('conversations', []) if conv['uuid'] == uuid), None)
    if conversation:
        return conversation.get('messages', [])
    else:
        return []
