# # app/api/v1/routes/chatbot.py
# from fastapi import APIRouter
# from app.services.search_service import SearchService
# from app.services.product_service import get_product_by_slug

# router = APIRouter()


# @router.post("/query")
# async def handle_user_query(query: str):
#     # Step 1: Search for products using the user's query
#     principal_product, secondary_products = await SearchService.search_products_by_query(query)

#     # Step 2: Verificar si se encontró un producto principal
#     if not principal_product:
#         return {"response": "Sorry, no products found.", "products": []}

#     # Step 3: Generar la respuesta final utilizando el producto principal y los complementarios
#     response = await SearchService.generate_answer(principal_product, secondary_products, query)

#     return {
#         "response": response,
#         "products": {
#             "principal": principal_product,
#             "secondary": secondary_products
#         }
#     }
