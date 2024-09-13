# app/models/user.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class Admin(BaseModel):
    name: str
    email: EmailStr
    hashed_password: str
    profile_image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
