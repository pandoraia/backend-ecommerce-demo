# app/api/v1/routes/products.py
from fastapi import APIRouter, HTTPException, Depends
from app.models.product import Product, ProductOut
from app.core.auth import get_current_admin
from app.services.product_service import get_all_products, create_product, get_product_by_slug
from typing import List
router = APIRouter()

@router.get("/", response_model=List[ProductOut])
async def list_products():
    products = await get_all_products()
    if not products:
        raise HTTPException(status_code=404, detail="No products found")
    return products


@router.get("/{slug}", response_model=ProductOut)
async def get_product(slug: str):
    product = await get_product_by_slug(slug)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@router.post("/", response_model=Product)
async def add_product(
    product: Product,
    # current_admin: dict = Depends(get_current_admin)
):
    try:
        new_product = await create_product(product)
        return new_product
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

