from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class RegisterAdmin(BaseModel):
    name: str
    email: str
    password: str
    profile_image_url: Optional[str] = None

class AdminResponse(BaseModel):
    id: str
    name: str
    email: str
    profile_image_url: Optional[str] = None
    created_at: datetime

class Admin(BaseModel):  # Añadir esta clase si es necesario
    id: str
    name: str
    email: str
    profile_image_url: Optional[str] = None
    created_at: datetime
