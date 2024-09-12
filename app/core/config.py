# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "My Ecommerce API"
    mongo_uri: str
    mongo_db_name: str

    class Config:
        env_file = ".env"

settings = Settings()