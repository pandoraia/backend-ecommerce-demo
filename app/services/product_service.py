# app/services/product_service.py
from app.db.database import db
from app.models.product import Product

async def get_all_products():
    products = await db["products"].find().to_list(100)
    return products
