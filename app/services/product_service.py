# app/services/product_service.py
from app.db.database import db
from app.models.product import Product
from typing import Dict, Any
from app.db.database import products_collection

async def get_all_products():
    products = await db["products"].find().to_list(100)
    return products



async def create_product(product: Product) -> Dict[str, Any]:
    # Convierte el modelo Pydantic a un diccionario
    product_dict = product.dict()
    
    # Inserta el producto en la base de datos
    result = await products_collection.insert_one(product_dict)
    
    # Recupera el producto creado usando el ID del resultado
    created_product = await products_collection.find_one({"_id": result.inserted_id})
    return created_product