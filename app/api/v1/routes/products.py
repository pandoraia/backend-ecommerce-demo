# app/api/v1/routes/products.py
from fastapi import APIRouter, HTTPException
from app.models.product import Product
from app.services.product_service import get_all_products, create_product

router = APIRouter()

@router.get("/")
async def list_products():
    products = await get_all_products()
    if not products:
        raise HTTPException(status_code=404, detail="No products found")
    return products


@router.post("/", response_model=Product)
async def add_product(product: Product):
    try:
        new_product = await create_product(product)
        return new_product
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))