
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List
# 📌 Modelo para una orden

class Order(BaseModel):
    id: str
    user_id: str  # Referencia al usuario que hizo la orden
    order_date: datetime
    total_price: float
    status: str = Field(..., pattern="^(Pending|Processing|Shipped|Delivered|Cancelled)$") 
    products: List[dict]  # Lista de productos en la orden
