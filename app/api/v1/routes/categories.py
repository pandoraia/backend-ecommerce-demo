# routes.py
from fastapi import APIRouter, HTTPException
from typing import List
from app.models.category import CategoryCreate, Category
from app.services.category_service import create_categories, get_categories

router = APIRouter()

@router.post("/", response_model=List[Category])
async def create_category_endpoint(categories: List[CategoryCreate]):
    try:
        created_categories = await create_categories(categories)
        return created_categories
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[Category])
async def get_categories_endpoint():
    return await get_categories()
