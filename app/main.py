# app/main.py
from fastapi import FastAPI
from app.db.database import connect_to_mongo, close_mongo_connection
from app.api.v1.routes import products, users, orders, admin
from app.core import auth
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",  # La URL de tu frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Incluir las rutas de los endpoints
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(products.router, prefix="/api/v1/products", tags=["Products"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(orders.router, prefix="/api/v1/orders", tags=["Orders"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])

@app.get("/")
async def ping():
    return {"message": "Pong"}
# get testing server

@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()
