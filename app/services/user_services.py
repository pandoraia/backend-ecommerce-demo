from datetime import datetime
from fastapi import HTTPException, status
from passlib.context import CryptContext
from app.db.database import user_collection
from app.models.user_models import UserResponse
# Configuración de hashing de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password):
    return pwd_context.hash(password)

async def register_user(user_data):
    # Verifica si el usuario ya existe
    existing_user = await user_collection.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already exists"
        )

    # Valida que las contraseñas coincidan
    if user_data.password != user_data.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match"
        )

    # Hashea la contraseña
    hashed_password = get_password_hash(user_data.password)

    # Crea el usuario con los datos proporcionados
    new_user = {
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "email": user_data.email,
        "hashed_password": hashed_password,
        "country_code": user_data.country_code,
        "phone": user_data.phone,
        "birth_date": user_data.birth_date,
        "accept_terms": user_data.accept_terms,
        "marketing_preferences": user_data.marketing_preferences,
        "created_at": datetime.utcnow()
    }

    # Inserta el usuario en la base de datos
    insert_result = await user_collection.insert_one(new_user)
    return str(insert_result.inserted_id)

async def get_user_by_email(email: str) -> UserResponse:
    # Implementación de ejemplo con MongoDB
    user = await user_collection.find_one({"email": email})
    if not user:
        return None
    return UserResponse(
        id=str(user["_id"]),
        first_name=user["first_name"],
        last_name=user["last_name"],
        email=user["email"],
        country_code=user.get("country_code"),
        phone=user.get("phone"),
        birth_date=user.get("birth_date"),
        created_at=user["created_at"]
    )


async def get_all_users():
    # Obtener todos los usuarios
    users = await user_collection.find().to_list(None)
    return [
        {
            "id": str(user["_id"]),
            "first_name": user["first_name"],
            "last_name": user["last_name"],
            "email": user["email"],
            "country_code": user["country_code"],
            "phone": user["phone"],
            "birth_date": user["birth_date"],
            "created_at": user["created_at"],
        }
        for user in users
    ]
    
 