from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from passlib.context import CryptContext
from datetime import datetime
from typing import Optional, List
from app.db.database import admin_collection
from app.core.auth import get_current_admin  # Importamos la función que valida el token

router = APIRouter()

# Configuración de hashing de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Pydantic model para el registro de usuario
class RegisterAdmin(BaseModel):
    name: str
    email: str
    password: str
    profile_image_url: Optional[str] = None

# Pydantic model para la respuesta de un administrador
class AdminResponse(BaseModel):
    id: str
    name: str
    email: str
    profile_image_url: Optional[str] = None
    created_at: datetime

def get_password_hash(password):
    return pwd_context.hash(password)

# Crear una dependencia que devuelva la colección de admin
def get_admin_collection():
    return admin_collection

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_admin(
    admin: RegisterAdmin, 
    current_admin: dict = Depends(get_current_admin),  # Valida el token del administrador
    admin_collection=Depends(get_admin_collection)     # Obtener la colección de admin
):
    print(f"Trying to register admin with email: {admin.email}")

    # Verifica si el usuario ya existe de manera asíncrona
    existing_admin = await admin_collection.find_one({"email": admin.email})
    print(f"Existing admin: {existing_admin}")
    
    if existing_admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Admin already exists"
        )

    # Hashea la contraseña
    hashed_password = get_password_hash(admin.password)

    # Crea un nuevo usuario
    new_admin = {
        "name": admin.name,
        "email": admin.email,
        "hashed_password": hashed_password,
        "profile_image_url": admin.profile_image_url,
        "created_at": datetime.utcnow()
    }

    print(f"Inserting new admin: {new_admin}")
    # Inserta el usuario en la base de datos de manera asíncrona
    insert_result = await admin_collection.insert_one(new_admin)
    admin_id = insert_result.inserted_id

    return {
        "message": "Admin created successfully",
        "admin_id": str(admin_id)
    }

@router.get("/admins", response_model=List[AdminResponse])
async def get_all_admins(
    current_admin: dict = Depends(get_current_admin),  # Verificar el token del administrador
    admin_collection=Depends(get_admin_collection)     # Obtener la colección de admin
):
    # Obtener todos los administradores
    admins = await admin_collection.find().to_list(None)
    
    # Transformar los documentos de MongoDB en la respuesta de Pydantic
    return [{"id": str(admin["_id"]), "name": admin["name"], "email": admin["email"], 
             "profile_image_url": admin.get("profile_image_url"), "created_at": admin["created_at"]} 
            for admin in admins]