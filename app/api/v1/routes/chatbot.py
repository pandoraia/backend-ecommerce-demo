# app/api/v1/routes/chatbot.py
from fastapi import APIRouter
from app.services.search_service import SearchService
from bs4 import BeautifulSoup

router = APIRouter()


def validate_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return str(soup)


@router.post("/query")
async def handle_user_query(query: str):
    # Generar la respuesta y obtener los productos
    result = await SearchService.generate_answer(query)
    response = result["response"]
    products = result["products"]

    response = validate_html(response)

    return {
        "response": response,
        "products": products
    }
