1. Entorno de desarrollo:
Primero, vamos a asegurarnos de que tengas el entorno adecuado configurado.

a. Crear un entorno virtual:
Es una buena práctica usar entornos virtuales para aislar las dependencias de tu proyecto.

bash
Copiar código
# Crear un entorno virtual
python -m venv env

# Activar el entorno virtual
# Para Linux/MacOS:
source env/bin/activate
# Para Windows:
env\Scripts\activate
b. Instalar dependencias iniciales:
Instalemos FastAPI, Uvicorn (servidor ASGI), motor (el driver de MongoDB asíncrono) y pydantic para validaciones.

bash
Copiar código
# Instalar FastAPI, Uvicorn, motor y otras dependencias
pip install fastapi uvicorn motor pydantic

# (Opcional) Si prefieres usar Python >=3.11 para mejores prestaciones en concurrencia, puedes asegurarte de estar usando esa versión de Python.
2. Estructura del proyecto:
Organizaremos tu proyecto de manera modular para que sea escalable y fácil de mantener.

bash
Copiar código
my_ecommerce/
│
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── products.py  # Endpoints para productos
│   │   │   │   ├── users.py     # Endpoints para usuarios
│   │   │   │   └── orders.py    # Endpoints para órdenes
│   │   │   └── dependencies.py  # Dependencias como seguridad, autenticación, etc.
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py  # Configuración del proyecto (variables de entorno, etc.)
│   │   └── security.py  # Gestión de seguridad (JWT, OAuth, etc.)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── product.py  # Modelos de productos
│   │   ├── user.py     # Modelos de usuarios
│   │   ├── order.py    # Modelos de órdenes
│   ├── services/
│   │   ├── __init__.py
│   │   ├── product_service.py  # Lógica relacionada a productos
│   │   ├── user_service.py     # Lógica relacionada a usuarios
│   │   └── order_service.py    # Lógica relacionada a órdenes
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py  # Conexión a MongoDB
│   │   └── models.py    # Iniciación de modelos
│   ├── main.py  # Punto de entrada principal de la app
│   └── utils/
│       ├── __init__.py
│       ├── logger.py  # Manejador de logs
│       └── helpers.py  # Funciones auxiliares
│
├── tests/
│   ├── __init__.py
│   ├── test_products.py  # Pruebas de productos
│   ├── test_users.py     # Pruebas de usuarios
│   └── test_orders.py    # Pruebas de órdenes
│
├── .env  # Archivo de variables de entorno (como la conexión a MongoDB)
├── .gitignore  # Ignorar carpetas como `env/` y archivos innecesarios
├── requirements.txt  # Listado de dependencias
└── README.md  # Documentación inicial del proyecto


Descripción de la estructura:
app/main.py: Este es el punto de entrada de la aplicación donde arrancas FastAPI y Uvicorn.
app/api: Aquí irán todas tus rutas (endpoints) organizadas por versiones (en este caso v1).
app/models: Aquí defines los esquemas o modelos que representarán tus datos (por ejemplo, User, Product, Order).
app/services: Aquí se encuentra la lógica del negocio para cada parte de tu aplicación. Los servicios se encargan de realizar operaciones más complejas con los datos, como interactuar con la base de datos.
app/db: Contiene la configuración de la conexión a MongoDB utilizando motor y la inicialización de modelos de MongoDB.
app/core: Aquí puedes poner configuraciones globales y temas relacionados con seguridad, como la configuración de JWT u OAuth.
app/utils: Archivos de utilidad como logs y funciones auxiliares.
tests: Aquí irán todas tus pruebas unitarias.
3. Configuración del proyecto:
a. Conexión a MongoDB (app/db/database.py)
En este archivo se creará la conexión con MongoDB utilizando motor.

python
Copiar código
# app/db/database.py
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import errors
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ecommerce")

client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB_NAME]

async def connect_to_mongo():
    try:
        client.server_info()
        print("Conectado a MongoDB")
    except errors.ServerSelectionTimeoutError:
        print("Error conectándose a MongoDB")

async def close_mongo_connection():
    client.close()
b. Configuración principal (app/core/config.py)
Aquí puedes cargar las variables de entorno y configurar el proyecto.

python
Copiar código
# app/core/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My Ecommerce API"
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "ecommerce"

    class Config:
        env_file = ".env"

settings = Settings()
c. Inicialización de FastAPI (app/main.py)
Este archivo será el punto de entrada de tu aplicación.

python
Copiar código
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
d. Definición de rutas (app/api/v1/routes/products.py)
Ejemplo básico de una ruta para productos:

python
Copiar código
# app/api/v1/routes/products.py
from fastapi import APIRouter, HTTPException
from app.models.product import Product
from app.services.product_service import get_all_products

router = APIRouter()

@router.get("/")
async def list_products():
    products = await get_all_products()
    if not products:
        raise HTTPException(status_code=404, detail="No products found")
    return products
e. Servicio para manejar lógica de productos (app/services/product_service.py)
python
Copiar código
# app/services/product_service.py
from app.db.database import db
from app.models.product import Product

async def get_all_products():
    products = await db["products"].find().to_list(100)
    return products
4. Variables de entorno:
Crea un archivo .env para manejar las variables de entorno como la conexión a MongoDB.

makefile
Copiar código
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=ecommerce
5. Iniciar el servidor:
Para correr la aplicación:

bash
Copiar código
uvicorn app.main:app --reload
Esto iniciará tu servidor FastAPI en modo de recarga automática.
