from pydantic import BaseModel, EmailStr, Field
from datetime import datetime
from typing import Optional, List

# 📌 Modelo para una dirección
class Address(BaseModel):
    type: str = Field(..., pattern="^(shipping|billing)$")  # Solo puede ser 'shipping' o 'billing'
    street: str = Field(..., min_length=5, max_length=100)
    city: str = Field(..., min_length=2, max_length=50)
    state: str = Field(..., min_length=2, max_length=50)
    zip_code: str = Field(..., min_length=3, max_length=10)
    country: str = Field(..., min_length=2, max_length=50)


# Modelo para la creación de un usuario
class RegisterUser(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)
    country_code: str = "+52"  # Valor por defecto
    phone: str = Field(..., pattern=r"^\d{10}$", description="Número de teléfono de 10 dígitos")
    birth_date: datetime
    accept_terms: bool
    marketing_preferences: bool = True
    addresses: Optional[List[Address]] = []

# Respuesta al cliente
class UserResponse(BaseModel):
    id: str
    first_name: str
    last_name: str
    email: str
    country_code: str
    phone: str
    birth_date: datetime
    created_at: datetime
    addresses: Optional[List[Address]] = []
