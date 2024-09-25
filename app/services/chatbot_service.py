# /app/services/chatbot_service.py
from app.services.search_service import SearchService

class ChatbotService:
    
    @staticmethod
    async def process_user_query(query_text: str):
        """
        Procesa la consulta del usuario y devuelve una respuesta adecuada.
        """
        recommended_products = await SearchService.search_products_by_query(query_text)

        if recommended_products:
            return recommended_products
        else:
            return "Lo siento, no he encontrado productos relacionados con tu consulta."
