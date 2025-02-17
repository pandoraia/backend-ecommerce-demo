# app/core/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Configuración general
    app_name: str = "My Ecommerce API"

    # Configuración de MongoDB
    mongo_uri: str
    mongo_db_name: str
    stripe_secret_key: str
    stripe_public_key: str
    stripe_webhook_secret: str
    # Configuración de OpenAI
    openai_api_key: str

    # Configuración de Pinecone
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index: str

    # Configuración de LangChain (LangSmith)
    langchain_tracing_v2: bool  # True/False
    langchain_endpoint: str
    langchain_api_key: str
    langchain_project: str

    class Config:
        env_file = ".env"  # Archivo desde donde se cargan las variables


# Instancia global para usar en toda la app
settings = Settings()
