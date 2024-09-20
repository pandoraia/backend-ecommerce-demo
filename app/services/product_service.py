# app/services/product_service.py
from app.db.database import db
from app.models.product import Product, ProductOut
from typing import Dict, Any, List, Optional
from app.db.database import products_collection
from app.services.vectorization_service import generate_product_embeddings  # Importar la función de embeddings


def product_serializer(product: Dict[str, Any]) -> ProductOut:
    # Asegúrate de que `_id` se convierta en una cadena y se agregue como `id`
    if '_id' in product:
        product['id'] = str(product.pop('_id'))
    return ProductOut(**product)

async def get_all_products() -> List[ProductOut]:
    products = await products_collection.find().to_list(100)
    # Verifica si `_id` está presente en cada producto antes de serializar
    return [product_serializer(product) for product in products]

async def get_product_by_slug(slug: str) -> Optional[ProductOut]:
    """Obtiene un producto por su slug."""
    product = await products_collection.find_one({"slug": slug})
    if product:
        return product_serializer(product)
    return None

async def create_product(product: Product) -> Dict[str, Any]:
    # Convierte el modelo Pydantic a un diccionario
    product_dict = product.dict()

    # Generar embeddings utilizando OpenAI
    embeddings = await generate_product_embeddings(product)

    # Agregar los embeddings al diccionario del producto
    product_dict['embeddings'] = embeddings

    # Inserta el producto en la base de datos
    result = await products_collection.insert_one(product_dict)

    # Recupera el producto creado usando el ID del resultado
    created_product = await products_collection.find_one({"_id": result.inserted_id})
    return created_product