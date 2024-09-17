# models.py
from pydantic import BaseModel
from typing import List, Optional
from bson import ObjectId

class CategoryBase(BaseModel):
    categoryName: str
    subcategories: List[str]

class CategoryCreate(CategoryBase):
    pass

class Category(CategoryBase):
    id: Optional[str]

    class Config:
        orm_mode = True
        json_encoders = {
            ObjectId: str  # Convert ObjectId to string
        }
