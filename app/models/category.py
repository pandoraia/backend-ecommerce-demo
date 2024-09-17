from pydantic import BaseModel
from typing import List, Optional, Dict
from bson import ObjectId

class Translations(BaseModel):
    en: str
    es: str
    fr: str
    de: str

class CategoryBase(BaseModel):
    categoryName: Translations
    subcategories: List[Dict[str, Translations]]

class CategoryCreate(CategoryBase):
    pass

class Category(CategoryBase):
    id: Optional[str]

    class Config:
        orm_mode = True
        json_encoders = {
            ObjectId: str  # Convert ObjectId to string
        }
