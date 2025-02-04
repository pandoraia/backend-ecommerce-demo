# app/main.py
from fastapi import FastAPI
from app.db.database import connect_to_mongo, close_mongo_connection
from app.api.v1.routes import products, users, orders, admin, categories, chatbot
from app.core import auth, user_auth
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3001",# La URL de tu frontend
    "https://pandorai.ch",
    "https://ecomerce-demo-pandorai.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    # Permite todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    allow_headers=["*"],  # Permite todos los encabezados
)

# Incluir las rutas de los endpoints
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(user_auth.router, prefix="/user_auth", tags=["user_auth"])
app.include_router(products.router, prefix="/api/v1/products", tags=["Products"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(orders.router, prefix="/api/v1/orders", tags=["Orders"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(categories.router, prefix="/api/v1/categories", tags=["Categories"])
app.include_router(chatbot.router, prefix="/chatbot", tags=["Chatbot"])



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
