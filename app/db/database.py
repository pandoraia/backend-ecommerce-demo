
from motor.motor_asyncio import AsyncIOMotorClient
import certifi
from app.core.config import settings  # Importar las configuraciones desde config.py

client = AsyncIOMotorClient(
    settings.mongo_uri,
    tlsCAFile=certifi.where()
)
db = client[settings.mongo_db_name] 
admin_collection = db['admins']
products_collection = db['products']
categories_collection = db['categories']

async def connect_to_mongo():
    try:
        client.server_info()
        print(f"Conectado a MongoDB en {settings.mongo_uri}")
    except Exception as e:
        print(f"Error conectándose a MongoDB: {e}")

async def close_mongo_connection():
    client.close()
