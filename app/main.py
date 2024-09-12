# app/main.py
from fastapi import FastAPI
from app.db.database import connect_to_mongo, close_mongo_connection
from app.api.v1.routes import products, users, orders

app = FastAPI()

# Incluir las rutas de los endpoints
app.include_router(products.router, prefix="/api/v1/products", tags=["Products"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(orders.router, prefix="/api/v1/orders", tags=["Orders"])

@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()
