# app/api/v1/routes/chatbot.py
from fastapi import APIRouter
from app.services.search_service import SearchService
from app.services.product_service import get_product_by_slug

router = APIRouter()


@router.post("/query")
async def handle_user_query(query: str):
    # Step 1: Search for products using the user's query
    recommended_products = await SearchService.search_products_by_query(query)

    # Verificar si se encontraron productos
        # Verificar si se encontraron productos
    if not recommended_products:
        return {"response": "Sorry, no products found.", "products": []}

    # Step 2: Generar la respuesta final utilizando el producto principal y los complementarios
    response = await SearchService.generate_answer(recommended_products, query)

    return {"response": response, "products": recommended_products}
