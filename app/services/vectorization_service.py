import time
import os
import re
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from app.models.product import Product
from app.core.config import settings
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Inicializar Pinecone usando la nueva clase Pinecone
pc = Pinecone(api_key=settings.pinecone_api_key)

# Nombre del índice
index_name = settings.pinecone_index

# Comprobar si el índice ya existe
existing_indexes = pc.list_indexes()

# Acceder al índice creado o ya existente
index = pc.Index(index_name)

# Crear el almacén de vectores de Langchain para Pinecone
embedder = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
vector_store = PineconeVectorStore(index=index, embedding=embedder)

async def split_text_optimally(text: str) -> list:
    """
    Divide el texto de forma óptima basándose en saltos de línea, encabezados y longitud de caracteres.
    """
    sections = re.split(r'\n\n+', text.strip())  # Dividir en bloques usando saltos de línea dobles

    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,   # Tamaño del chunk
        chunk_overlap=20, # Superposición entre chunks
        length_function=len
    )

    chunks = []
    for section in sections:
        if len(section) > 200:  # Si la sección es muy grande, dividirla en subpartes
            sub_chunks = section_splitter.split_text(section)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section)

    return chunks

async def generate_product_embeddings(product: Product) -> None:
    for lang, translation in product.translations.items():
        name_text = translation.name
        description_text = translation.description
        description_vect_text = translation.destovect

        # Generar embeddings para el nombre y la descripción del producto
        name_embedding = embedder.embed_query(name_text)
        description_embedding = embedder.embed_query(description_text)

        # Dividir el texto en segmentos más pequeños de forma óptima
        description_vect_segments = await split_text_optimally(description_vect_text)
        description_vect_embeddings = embedder.embed_documents(description_vect_segments)
        print(product)
        # Metadata para relacionar con MongoDB (usamos el _id del producto)
        metadata = {
            "product_slug": str(product.slug),  # Asegúrate de tener `product.slug` en MongoDB
            "language": lang,
        }

        # Subir los embeddings a Pinecone

        # 1. Embedding del nombre del producto
        index.upsert(vectors=[
            (f"{product.slug}_name_{lang}", name_embedding, metadata)
        ])

        # 2. Embedding de la descripción del producto
        index.upsert(vectors=[
            (f"{product.slug}_description_{lang}", description_embedding, metadata)
        ])

        # 3. Embeddings de los segmentos de la descripción
        for i, segment_embedding in enumerate(description_vect_embeddings):
            segment_metadata = {**metadata, "segment": i}
            index.upsert(vectors=[
                (f"{product.slug}_description_segment_{lang}_{i}", segment_embedding, segment_metadata)
            ])
