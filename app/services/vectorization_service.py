from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Import actualizado
from app.models.product import Product
from app.core.config import settings
from typing import Dict, Any
import re

async def split_text_optimally(text: str) -> list:
    """
    Divide el texto de forma óptima basándose en saltos de línea, encabezados y longitud de caracteres.
    """
    # Dividir en bloques usando saltos de línea dobles como delimitadores de secciones
    sections = re.split(r'\n\n+', text.strip())
    
    # Patrón para identificar encabezados (líneas con mayúscula inicial y sin puntuación final)
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,   # Tamaño del chunk
        chunk_overlap=20, # Superposición entre chunks
        length_function=len
    )

    # Dividir cada sección en subpartes más pequeñas respetando su estructura
    chunks = []
    for section in sections:
        # Si la sección es muy grande, la dividimos en partes más pequeñas
        if len(section) > 200:
            sub_chunks = section_splitter.split_text(section)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section)
    
    return chunks

async def generate_product_embeddings(product: Product) -> Dict[str, Dict[str, list]]:
    embeddings = {}
    embedder = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)

    for lang, translation in product.translations.items():
        name_text = translation.name
        description_text = translation.description
        description_vect_text = translation.destovect

        # Generar embeddings para el nombre del producto (usar embed_query)
        name_embedding = embedder.embed_query(name_text)

        # Generar embeddings para la descripción del producto (usar embed_query)
        description_embedding = embedder.embed_query(description_text)

        # Dividir el texto en segmentos más pequeños de forma óptima
        description_vect_segments = await split_text_optimally(description_vect_text)

        # Generar embeddings para cada segmento de description_vect (usar embed_documents sin await)
        description_vect_embeddings = embedder.embed_documents(description_vect_segments)

        # Almacenar los embeddings
        embeddings[lang] = {
            'name_embedding': name_embedding,
            'description_embedding': description_embedding,
            'description_vect_embeddings': description_vect_embeddings
        }

    return embeddings
