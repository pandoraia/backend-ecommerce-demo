# services.py
from typing import List
from app.models.category import CategoryCreate, Category
from app.db.database import categories_collection

async def create_category(category: CategoryCreate) -> Category:
    category_dict = category.dict()
    result = await categories_collection.insert_one(category_dict)
    new_category = await categories_collection.find_one({"_id": result.inserted_id})
    # Convert _id to id
    new_category['id'] = str(new_category.pop('_id'))
    return Category(**new_category)

async def get_categories() -> List[Category]:
    categories = []
    async for category in categories_collection.find():
        category_dict = dict(category)
        # Convert _id to id
        category_dict['id'] = str(category_dict.pop('_id'))
        categories.append(Category(**category_dict))
    return categories
