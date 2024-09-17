from typing import List
from app.models.category import CategoryCreate, Category
from app.db.database import categories_collection

async def create_categories(categories: List[CategoryCreate]) -> List[Category]:
    category_dicts = [category.dict() for category in categories]
    result = await categories_collection.insert_many(category_dicts)
    inserted_ids = result.inserted_ids

    new_categories = []
    async for category in categories_collection.find({"_id": {"$in": inserted_ids}}):
        category_dict = dict(category)
        category_dict['id'] = str(category_dict.pop('_id'))
        new_categories.append(Category(**category_dict))
    return new_categories

async def get_categories() -> List[Category]:
    categories = []
    async for category in categories_collection.find():
        category_dict = dict(category)
        category_dict['id'] = str(category_dict.pop('_id'))
        categories.append(Category(**category_dict))
    return categories
