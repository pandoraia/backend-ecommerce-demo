# app/api/v1/routes/chatbot.py
from fastapi import APIRouter
from app.services.search_service import SearchService
from bs4 import BeautifulSoup
from pydantic import BaseModel

router = APIRouter()


class UserQuery(BaseModel):
    query: str
    userUUID: str


def validate_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    return str(soup)


@router.post("/query")
async def handle_user_query(user_query: UserQuery):
    query = user_query.query
    user_uuid = user_query.userUUID
    # Generar la respuesta y obtener los productos
    result = await SearchService.generate_answer(query, session_id=user_uuid)
    response = result["response"]
    products = result["products"]

    response = validate_html(response)

    return {
        "response": response,
        "products": products
    }
