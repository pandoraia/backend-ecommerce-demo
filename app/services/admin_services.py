# app/services/admin_services.py
from datetime import datetime
from fastapi import HTTPException, status
from passlib.context import CryptContext
from app.db.database import admin_collection

# Configuración de hashing de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

async def register_admin(admin_data):
    # Verifica si el usuario ya existe de manera asíncrona
    existing_admin = await admin_collection.find_one({"email": admin_data.email})
    if existing_admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin already exists"
        )

    # Hashea la contraseña
    hashed_password = get_password_hash(admin_data.password)

    # Crea un nuevo usuario
    new_admin = {
        "name": admin_data.name,
        "email": admin_data.email,
        "hashed_password": hashed_password,
        "profile_image_url": admin_data.profile_image_url,
        "created_at": datetime.utcnow()
    }

    # Inserta el usuario en la base de datos de manera asíncrona
    insert_result = await admin_collection.insert_one(new_admin)
    return str(insert_result.inserted_id)

async def get_all_admins():
    # Obtener todos los administradores
    admins = await admin_collection.find().to_list(None)
    return [{"id": str(admin["_id"]), "name": admin["name"], "email": admin["email"], 
             "profile_image_url": admin.get("profile_image_url"), "created_at": admin["created_at"]} 
            for admin in admins]
