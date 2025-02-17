from fastapi import APIRouter, Query
from typing import List, Optional
from app.models.order import Order
from app.services.order_service import create_order, get_orders, get_order, get_orders_by_user, update_order

router = APIRouter()

# ✅ Crear una nueva orden
@router.post("/orders", response_model=Order)
async def create_order_endpoint(order: Order):
    return await create_order(order)

# ✅ Obtener todas las órdenes (opcionalmente por usuario)
@router.get("/orders", response_model=List[Order])
def get_orders_endpoint(user_id: Optional[str] = Query(None, description="Filtrar por usuario")):
    return get_orders(user_id)

# ✅ Obtener una orden por ID
@router.get("/orders/{order_id}", response_model=Order)
def get_order_endpoint(order_id: str):
    return get_order(order_id)

# ✅ Obtener todas las órdenes de un usuario
@router.get("/users/{user_id}/orders", response_model=List[Order])
def get_orders_by_user_endpoint(user_id: str):
    return get_orders_by_user(user_id)

# ✅ Editar una orden
@router.put("/orders/{order_id}", response_model=Order)
def update_order_endpoint(order_id: str, user_id: str, status: Optional[str] = None, products: Optional[List[dict]] = None, total_price: Optional[float] = None):
    return update_order(order_id, user_id, status, products, total_price)
