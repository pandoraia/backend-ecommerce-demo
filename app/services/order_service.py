import uuid
from datetime import datetime
from fastapi import HTTPException
from typing import List, Optional
from app.models.order import Order
from app.db.database import user_collection, orders_collection
from bson import ObjectId

# ✅ Crear una nueva orden
async def create_order(order: Order) -> Order:
    # 📌 Verifica si el usuario existe en MongoDB
    user = await user_collection.find_one({"_id": ObjectId(order.user_id)})
    
    if not user:
        raise HTTPException(status_code=400, detail="Usuario no válido")

    # Generar un ID único y asignar fecha actual
    order.id = str(uuid.uuid4())
    order.order_date = datetime.now()

    # Insertar en la colección de órdenes
    await orders_collection.insert_one(order.dict())

    return order

# ✅ Obtener todas las órdenes (opcionalmente filtradas por usuario)
def get_orders(user_id: Optional[str] = None) -> List[Order]:
    if user_id:
        return [order for order in orders_collection if order.user_id == user_id]
    return orders_collection

# ✅ Obtener una orden específica por ID
def get_order(order_id: str) -> Order:
    order = next((o for o in orders_collection if o.id == order_id), None)
    if order is None:
        raise HTTPException(status_code=404, detail="Orden no encontrada")
    return order

# ✅ Obtener todas las órdenes de un usuario
def get_orders_by_user(user_id: str) -> List[Order]:
    if user_id not in user_collection:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return [order for order in orders_collection if order.user_id == user_id]

# ✅ Editar una orden (solo el usuario dueño puede modificarla)
def update_order(order_id: str, user_id: str, status: Optional[str] = None, products: Optional[List[dict]] = None, total_price: Optional[float] = None) -> Order:
    order = next((o for o in orders_collection if o.id == order_id), None)

    if order is None:
        raise HTTPException(status_code=404, detail="Orden no encontrada")

    if order.user_id != user_id:
        raise HTTPException(status_code=403, detail="No tienes permiso para modificar esta orden")

    if status:
        order.status = status
    if products:
        order.products = products
    if total_price:
        order.total_price = total_price

    return order
