from fastapi import APIRouter, HTTPException, Depends
from fastapi import status  # Añadir esta línea
from typing import List
from app.models.admin import RegisterAdmin, AdminResponse
from app.core.auth import get_current_admin
from app.services.admin_services import register_admin, get_all_admins

router = APIRouter()

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register_admin_route(
    admin: RegisterAdmin,
    current_admin: dict = Depends(get_current_admin)
):
    admin_id = await register_admin(admin)
    return {
        "message": "Admin created successfully",
        "admin_id": admin_id
    }

@router.get("/admins", response_model=List[AdminResponse])
async def get_all_admins_route(
    current_admin: dict = Depends(get_current_admin)
):
    return await get_all_admins()
