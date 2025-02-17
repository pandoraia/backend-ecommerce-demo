from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from app.services.user_services import register_user, get_all_users, get_user_by_email, update_user
from app.models.user_models import RegisterUser, UserResponse, UpdateUser

from typing import List, Optional
router = APIRouter()

@router.post("/register", response_model=UserResponse, status_code=201)
async def register_user_endpoint(user_data: RegisterUser):
    """
    Endpoint para registrar un nuevo usuario.
    """
    user_id = await register_user(user_data)
    return {
        "id": user_id,
        "first_name": user_data.first_name,
        "last_name": user_data.last_name,
        "email": user_data.email,
        "country_code": user_data.country_code,
        "phone": user_data.phone,
        "birth_date": user_data.birth_date,
        "created_at": datetime.utcnow(),
    }


@router.get("/", response_model=List[UserResponse])
async def get_users_endpoint(email: Optional[str] = Query(None)):
    """
    Endpoint para obtener usuarios. Filtra por email si se proporciona.
    """
    if email:
        user = await get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        return [user]
    return await get_all_users()

@router.patch("/{user_id}", response_model=UserResponse)
async def update_user_endpoint(user_id: str, user_data: UpdateUser):
    """
    Endpoint para actualizar la información de un usuario.
    """
    updated_user = await update_user(user_id, user_data)
    if not updated_user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    return updated_user
