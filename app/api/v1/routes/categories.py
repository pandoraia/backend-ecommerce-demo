# routes.py
from fastapi import APIRouter, HTTPException
from typing import List
from app.models.category import CategoryCreate, Category
from app.services.category_service  import create_category, get_categories

router = APIRouter()

@router.post("/", response_model=Category)
async def create_category_endpoint(category: CategoryCreate):
    return await create_category(category)

@router.get("/", response_model=List[Category])
async def get_categories_endpoint():
    return await get_categories()
