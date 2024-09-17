from typing import List, Dict, Optional
from pydantic import BaseModel, constr

class Translation(BaseModel):
    name: str
    description: str
    category: str
    subCategory: str

class VariantAttribute(BaseModel):
    key: str
    value: str

class Variant(BaseModel):
    translations: Dict[str, List[VariantAttribute]]
    stock: Optional[int] = None

class Tags(BaseModel):
    en: List[str]
    es: List[str]
    fr: List[str]
    de: List[str]

class Product(BaseModel):
    slug: str
    translations: Dict[str, Translation]
    price: float
    brand: str
    sku: str
    stock: Optional[int] = None
    images: List[str]
    variants: List[Variant]
    tags: Tags
    rating: Optional[float] = None
    reviews: Optional[List[Dict]] = None
