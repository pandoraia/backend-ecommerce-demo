# app/api/user_auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.db.database import user_collection  # Importar colección de usuarios # Modelo de usuario
from datetime import datetime, timedelta
from bson import ObjectId
import os

router = APIRouter()

# Configuración (puedes usar diferentes credenciales si lo prefieres)
SECRET_KEY = os.getenv("USER_SECRET_KEY", "usersupersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 semana para usuarios

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="user_auth/token")

def verify_user_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_user_password_hash(password: str):
    return pwd_context.hash(password)

async def authenticate_user(email: str, password: str):
    user = await user_collection.find_one({"email": email})
    if not user:
        return False
    if not verify_user_password(password, user['hashed_password']):
        return False
    return user

def create_user_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@router.post("/token")
async def login_for_user_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo electrónico o contraseña incorrectos",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_user_access_token(
        data={"sub": str(user['_id']), "email": user['email']}, expires_delta=access_token_expires
    )
    
    # Imprimir el token generado
    print(f"Generated access token: {access_token}")
    return {"access_token": access_token, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = {"id": user_id}
    except JWTError:
        raise credentials_exception

    user = await user_collection.find_one({"_id": ObjectId(token_data["id"])})
    if user is None:
        raise credentials_exception
    return user
    