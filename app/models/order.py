from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict

# 📌 Modelo para las variantes del producto (ej. color, peso, sabor)
class ProductVariant(BaseModel):
    translations: Dict[str, List[Dict[str, str]]]  # Traducciones de las variantes

# 📌 Modelo de producto dentro de la orden
class Product(BaseModel):
    slug: str
    translations: Dict[str, Dict[str, str]]  # Nombre del producto en varios idiomas
    price: float
    variant: List[ProductVariant]
    image: str

# 📌 Modelo de pasos completados en la orden
class OrderStep(BaseModel):
    step: int
    date: datetime

# 📌 Modelo de una orden
class Order(BaseModel):
    id: Optional[str] = None
    user_id: str  # Relación con un usuario
    order_date: Optional[datetime] = None
    total_price: float
    status: str = Field(..., pattern="^(Pending|Processing|Shipped|Delivered|Cancelled)$")
    products: List[Product]  # Lista de productos en la orden
    completed_steps: Optional[List[OrderStep]] = []  # Historial de pasos de la orden (opcional)
