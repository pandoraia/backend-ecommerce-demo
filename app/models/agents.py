# app/models.py

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class Message(BaseModel):
    sender: str  # "user" or "bot"
    text: str
    time: datetime

class Conversation(BaseModel):
    uuid: str
    messages: List[Message] = []

class Agent(BaseModel):
    name: str
    conversations: List[Conversation] = []
