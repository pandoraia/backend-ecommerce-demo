# app/api/v1/routes/chatbot.py
from fastapi import APIRouter
from app.services.search_service import SearchService

router = APIRouter()


@router.post("/query")
async def handle_user_query(query: str):
    # Generar la respuesta y obtener los productos
    result = await SearchService.generate_answer(query)
    response = result["response"]
    products = result["products"]

    return {
        "response": response,
        "products": products
    }
