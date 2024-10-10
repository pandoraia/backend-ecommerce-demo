#app/api/v1/routes/chatbot.py
from fastapi import APIRouter
from app.services.search_service import SearchService
from app.services.product_service import get_product_by_slug

router = APIRouter()

@router.post("/query")
async def handle_user_query(query: str):
    # Step 1: Search for products using the user's query
    recommended_products = await SearchService.search_products_by_query(query)
    
    # Verificar si se encontraron productos
    if not recommended_products or "main_product" not in recommended_products:
        return {"response": "Sorry, no products found.", "products": []}
    
    # Extraer el producto principal y los productos complementarios
    main_product = recommended_products["main_product"]
    complementary_products = recommended_products.get("complementary_products", [])
    
    # Convertir productos complementarios a un formato adecuado (string o lista)
    complementary_products_str = ", ".join([f"{p['name']} ({p['description']})" for p in complementary_products[:2]])  # Limitar a 2 productos
    
    # Step 2: Generar la respuesta final utilizando el producto principal y los complementarios
    response = await SearchService.generate_answer(main_product["name"], complementary_products_str, query)
    
    return {"response": response, "products": recommended_products}
