

import re
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from app.models.product import Product
from app.core.config import settings
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

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
    Divide el texto en segmentos más pequeños para capturar mejor el contexto.
    """
    sections = re.split(r'\n\n+', text.strip())
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
        length_function=len
    )

    chunks = []
    for section in sections:
        if len(section) > 200:
            sub_chunks = section_splitter.split_text(section)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section)

    return chunks

async def generate_product_embeddings(product: Product) -> None:
    """
    Genera embeddings para un producto y sus traducciones en diferentes idiomas.
    """
    for lang, translation in product.translations.items():
        # Datos del producto por idioma
        name_text = translation.name
        description_text = translation.description
        full_text_sections = [
            translation.destovect,  # Secciones detalladas del producto
            f"Category: {translation.category}",
            f"SubCategory: {translation.subCategory}"
        ]
        price_text = f"Price: {product.price}"

        # Generar embeddings para el nombre del producto y la descripción general
        name_embedding = embedder.embed_query(name_text)
        description_embedding = embedder.embed_query(description_text)
        price_embedding = embedder.embed_query(price_text)

        # Crear una lista para almacenar todos los embeddings bajo el mismo vector_id
        vectors_to_upsert = []

        # Agregar embedding del nombre del producto con metadatos
        metadata_name = {
            "product_slug": product.slug,
            "language": lang,
            "type": "name"
        }
        vectors_to_upsert.append((f"{product.slug}_{lang}_name", name_embedding, metadata_name))

        # Agregar embedding de la descripción general con metadatos
        metadata_description = {
            "product_slug": product.slug,
            "language": lang,
            "type": "description"
        }
        vectors_to_upsert.append((f"{product.slug}_{lang}_description", description_embedding, metadata_description))
        
        metadata_price = {
            "product_slug": product.slug,
            "language": lang,
            "type": "price"
        }
        vectors_to_upsert.append((f"{product.slug}_{lang}_price", price_embedding, metadata_price))


        # Procesar y dividir las secciones detalladas del producto para embeddings más específicos
        for i, section in enumerate(full_text_sections):
            section_segments = await split_text_optimally(section)
            segment_embeddings = embedder.embed_documents(section_segments)

            # Agregar cada segmento como un embedding individual con metadatos específicos y manteniendo el orden
            for j, segment_embedding in enumerate(segment_embeddings):
                segment_metadata = {
                    "product_slug": product.slug,
                    "language": lang,
                    "type": "section_segment",
                    "section_index": i,
                    "segment_index": j
                }
                vectors_to_upsert.append((f"{product.slug}_{lang}_segment_{i}_{j}", segment_embedding, segment_metadata))
                # Agregar embeddings para la categoría y subcategoría con metadatos
        category_embedding = embedder.embed_query(translation.category)
        subcategory_embedding = embedder.embed_query(translation.subCategory)

        # Agregar embedding de la categoría con metadatos
        metadata_category = {
            "product_slug": product.slug,
            "language": lang,
            "type": "category"
        }
        vectors_to_upsert.append((f"{product.slug}_{lang}_category", category_embedding, metadata_category))

        # Agregar embedding de la subcategoría con metadatos
        metadata_subcategory = {
            "product_slug": product.slug,
            "language": lang,
            "type": "subcategory"
        }
        vectors_to_upsert.append((f"{product.slug}_{lang}_subcategory", subcategory_embedding, metadata_subcategory))


        # Subir los embeddings al índice de Pinecone asegurando el orden coherente
        index.upsert(vectors=vectors_to_upsert)

        print(f"Product embeddings for {product.slug} in language {lang} have been successfully upserted.")

