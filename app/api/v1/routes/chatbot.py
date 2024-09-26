from fastapi import APIRouter
from app.services.search_service import SearchService
from app.services.product_service import get_product_by_slug

router = APIRouter()

@router.post("/query")
async def handle_user_query(query: str):
    # Step 1: Search for products using the user's query
    recommended_products = await SearchService.search_products_by_query(query)
    # product = await get_product_by_slug(slug)
    
    # Extract the top recommended product and complementary products
    if len(recommended_products) > 0:
        recommended_product = recommended_products[0]["id"]
        complementary_products = ", ".join([p["id"] for p in recommended_products[1:3]])  # Two complementary products
    else:
        return {"response": "Sorry, no products found.", "products": []}

    # Step 2: Generate the final answer using the recommended and complementary products
    response = await SearchService.generate_answer(recommended_product, complementary_products, query)
    
    return {"response": response, "products": recommended_products}
